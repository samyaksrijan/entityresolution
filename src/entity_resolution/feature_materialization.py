"""Stream candidate Parquet into atomic, resume-safe pair-feature shards.

Example: python -m entity_resolution.feature_materialization --mode train \
  --candidates artifacts/production_candidates/train_max_score \
  --output artifacts/pair_features/train_max_score
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from entity_resolution.config import load_data_paths
from entity_resolution.feature_schema import OUTPUT_SCHEMA, SCHEMA_VERSION
from entity_resolution.pair_features import SourceRecord, compute_pair_features

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "pair_features.yaml"
REQUIRED_CANDIDATE_COLUMNS = (
    "s1_id",
    "target_id",
    "target_source",
    "channel",
    "retrieval_score",
    "channel_rank",
    "exact_match",
    "run_id",
)
OPTIONAL_CANDIDATE_COLUMNS = ("support_json", "fusion_score")
SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _candidate_inputs(root: Path) -> tuple[list[Path], dict[Path, str]]:
    expected: dict[Path, str] = {}
    if root.is_file():
        files = [root]
    else:
        if not root.is_dir():
            raise ValueError(f"candidate input does not exist: {root}")
        files = sorted(root.glob("part-*.parquet"))
        manifest = root / "manifest.json"
        if manifest.is_file():
            state = json.loads(manifest.read_text(encoding="utf-8"))
            if not state.get("complete"):
                raise ValueError("candidate manifest is incomplete")
            listed = [root / item["file"] for item in state["completed_shards"]]
            if files != listed:
                raise ValueError("candidate manifest and shard listing disagree")
            expected = {
                path: item["sha256"]
                for path, item in zip(files, state["completed_shards"], strict=True)
            }
    fingerprints = {}
    for path in files:
        try:
            parquet = pq.ParquetFile(path)
            names = set(parquet.schema_arrow.names)
        except (OSError, pa.ArrowException) as exc:
            raise ValueError(f"corrupt candidate shard: {path}") from exc
        if not set(REQUIRED_CANDIDATE_COLUMNS) <= names or names - set(
            (*REQUIRED_CANDIDATE_COLUMNS, *OPTIONAL_CANDIDATE_COLUMNS)
        ):
            raise ValueError(f"incompatible candidate schema: {path}")
        digest = _sha256(path)
        if path in expected and expected[path] != digest:
            raise ValueError(f"corrupt candidate shard: {path}")
        fingerprints[path] = digest
    return files, fingerprints


def _build_index(path: Path, source_paths: dict[str, Path]) -> None:
    temporary = path.with_suffix(".sqlite.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with sqlite3.connect(temporary) as db:
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("PRAGMA cache_size=-16384")
            db.execute(
                "CREATE TABLE records(id TEXT PRIMARY KEY,source TEXT,"
                "name TEXT,address TEXT,country TEXT)"
            )
            for source, source_path in source_paths.items():
                with source_path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.reader(handle, delimiter="\t")
                    if tuple(next(reader, ())) != SOURCE_COLUMNS:
                        raise ValueError(f"invalid source TSV header: {source_path}")
                    pending = []
                    for line, row in enumerate(reader, 2):
                        if len(row) != 4 or not row[0].startswith(source + "-") or len(row[0]) <= 3:
                            raise ValueError(f"invalid {source} record at line {line}")
                        pending.append((row[0], source, row[1], row[2], row[3]))
                        if len(pending) == 1000:
                            db.executemany("INSERT INTO records VALUES (?,?,?,?,?)", pending)
                            pending.clear()
                    if pending:
                        db.executemany("INSERT INTO records VALUES (?,?,?,?,?)", pending)
                db.commit()
        os.replace(temporary, path)
    except (sqlite3.IntegrityError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"invalid source data: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _lookup(db: sqlite3.Connection, ids: set[str]) -> dict[str, SourceRecord]:
    records = {}
    ordered = sorted(ids)
    for start in range(0, len(ordered), 500):
        group = ordered[start : start + 500]
        marker = ",".join("?" for _ in group)
        for entity_id, source, name, address, country in db.execute(
            f"SELECT id,source,name,address,country FROM records WHERE id IN ({marker})", group
        ):
            records[entity_id] = SourceRecord(source, name, address, country)
    if len(records) != len(ids):
        raise ValueError(f"candidate references {len(ids) - len(records)} absent source records")
    return records


def _write_shard(table: pa.Table, path: Path) -> None:
    temporary = path.with_suffix(".parquet.tmp")
    try:
        pq.write_table(table, temporary, compression="zstd", row_group_size=65536)
        check = pq.read_table(temporary)
        if (
            not check.schema.equals(OUTPUT_SCHEMA, check_metadata=False)
            or check.num_rows != table.num_rows
        ):
            raise ValueError("feature shard verification failed")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize(
    *,
    mode: str,
    candidates: Path,
    output: Path,
    data_paths_config: Path,
    batch_size: int = 1000,
    shard_size: int = 1000,
    resume: bool = False,
) -> dict[str, Any]:
    """Build features with memory bounded by one candidate batch and SQLite lookup."""
    if mode not in {"train", "test"} or batch_size < 1 or shard_size < 1:
        raise ValueError("invalid mode or batch/shard size")
    paths = load_data_paths(data_paths_config)
    source_paths = {f"S{i}": paths[f"{mode}_source{i}"] for i in (1, 2, 3)}
    raw_root = Path(os.path.commonpath([str(path.parent) for path in paths.values.values()]))
    if output.resolve() == raw_root or raw_root in output.resolve().parents:
        raise ValueError("output may not be beneath organizer raw data")
    input_files, candidate_hashes = _candidate_inputs(candidates)
    if output.resolve() == candidates.resolve() or output.resolve() in candidates.resolve().parents:
        raise ValueError("output must be separate from candidate input")
    identity = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "candidate_files": [
            {"path": str(p.resolve()), "sha256": candidate_hashes[p]} for p in input_files
        ],
        "source_files": {
            key: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for key, path in source_paths.items()
        },
        "batch_size": batch_size,
        "shard_size": shard_size,
        "implementation": {
            name: _sha256(Path(__file__).with_name(name))
            for name in (
                "feature_schema.py",
                "pair_features.py",
                "feature_materialization.py",
                "views.py",
            )
        },
    }
    schema_record = [[field.name, str(field.type)] for field in OUTPUT_SCHEMA]
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    import fcntl

    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("feature output is locked") from exc
        if manifest.exists():
            if not resume:
                raise ValueError("feature run exists; use --resume")
            state = json.loads(manifest.read_text(encoding="utf-8"))
            if state.get("identity") != identity or state.get("schema") != schema_record:
                raise ValueError("incompatible feature manifest")
        else:
            if resume or list(output.glob("part-*.parquet")):
                raise ValueError("missing feature manifest or orphan shards")
            state = {
                "identity": identity,
                "complete": False,
                "shards": [],
                "schema": schema_record,
            }
            _atomic_json(manifest, state)
        for number, item in enumerate(state["shards"]):
            if item["file"] != f"part-{number:06d}.parquet":
                raise ValueError("noncontiguous feature manifest")
            path = output / item["file"]
            if not path.is_file() or _sha256(path) != item["sha256"]:
                raise ValueError("corrupt completed feature shard")
            parquet = pq.ParquetFile(path)
            if (
                not parquet.schema_arrow.equals(OUTPUT_SCHEMA, check_metadata=False)
                or parquet.metadata.num_rows != item["rows"]
            ):
                raise ValueError("incompatible completed feature shard")
        known = {item["file"] for item in state["shards"]}
        orphaned = {path.name for path in output.glob("part-*.parquet")} - known
        permitted = {f"part-{len(known):06d}.parquet"} if not state["complete"] else set()
        if orphaned - permitted:
            raise ValueError("unexpected unmanifested feature shard")
        if state["complete"]:
            return state
        index = output / "records.sqlite"
        if state.get("index_sha256") and (
            not index.is_file() or _sha256(index) != state["index_sha256"]
        ):
            raise ValueError("corrupt source index")
        if not index.is_file() or "index_sha256" not in state:
            _build_index(index, source_paths)
        if "index_sha256" not in state:
            state["index_sha256"] = _sha256(index)
            _atomic_json(manifest, state)
        seen_path = output / ".seen.sqlite.tmp"
        seen_path.unlink(missing_ok=True)
        with (
            ExitStack() as cleanup,
            sqlite3.connect(index) as db,
            sqlite3.connect(seen_path) as seen,
        ):
            cleanup.callback(seen_path.unlink, missing_ok=True)
            cleanup.callback(db.close)
            cleanup.callback(seen.close)
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("corrupt source index")
            seen.execute(
                "CREATE TABLE pairs(s1 TEXT,target TEXT,source TEXT,PRIMARY KEY(s1,target,source))"
            )
            unit = 0
            for path in input_files:
                parquet = pq.ParquetFile(path)
                for batch in parquet.iter_batches(batch_size=min(batch_size, shard_size)):
                    frame = batch.to_pandas()
                    run_ids = set(frame["run_id"])
                    if len(run_ids) != 1 or ("run_id" in state and state["run_id"] not in run_ids):
                        raise ValueError("candidate run ID changes across input batches")
                    if "run_id" not in state:
                        state["run_id"] = next(iter(run_ids))
                        _atomic_json(manifest, state)
                    keys = list(
                        frame[["s1_id", "target_id", "target_source"]].itertuples(
                            index=False, name=None
                        )
                    )
                    try:
                        seen.executemany("INSERT INTO pairs VALUES (?,?,?)", keys)
                    except sqlite3.IntegrityError as exc:
                        raise ValueError("duplicate candidate pair across input shards") from exc
                    if unit < len(state["shards"]):
                        if len(frame) != state["shards"][unit]["rows"]:
                            raise ValueError("resume input batch changed")
                        unit += 1
                        continue
                    ids = set(frame["s1_id"]) | set(frame["target_id"])
                    records = _lookup(db, ids)
                    features = compute_pair_features(frame, records)
                    table = pa.Table.from_pandas(
                        features, schema=OUTPUT_SCHEMA, preserve_index=False, safe=True
                    )
                    name = f"part-{unit:06d}.parquet"
                    _write_shard(table, output / name)
                    state["shards"].append(
                        {
                            "file": name,
                            "sha256": _sha256(output / name),
                            "rows": len(frame),
                            "input": path.name,
                        }
                    )
                    _atomic_json(manifest, state)
                    unit += 1
            if unit != len(state["shards"]):
                raise ValueError("resume contains more output than input")
        state["complete"] = True
        state["rows"] = sum(item["rows"] for item in state["shards"])
        _atomic_json(manifest, state)
        index.unlink(missing_ok=True)
        return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-paths", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    state = materialize(
        mode=args.mode,
        candidates=args.candidates,
        output=args.output,
        data_paths_config=args.data_paths or Path(config["data_paths_config"]),
        batch_size=args.batch_size or int(config["batch_size"]),
        shard_size=args.shard_size or int(config["shard_size"]),
        resume=args.resume,
    )
    print(
        json.dumps(
            {"complete": state["complete"], "rows": state["rows"], "shards": len(state["shards"])},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
