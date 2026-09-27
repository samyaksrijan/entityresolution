"""Strict feature validation and disk-backed, bounded feature staging."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from entity_resolution.config import DataPaths
from entity_resolution.feature_schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    OUTPUT_SCHEMA,
    SCHEMA_VERSION,
    feature_names,
)

KEYS = ["s1_id", "target_id", "target_source"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def contract():
    schema = [[f.name, str(f.type)] for f in OUTPUT_SCHEMA]
    return {
        "feature_names": list(feature_names()),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "schema_version": SCHEMA_VERSION,
        "schema_fingerprint": hashlib.sha256(json.dumps(schema).encode()).hexdigest(),
    }


def protect_output(output: Path, paths: DataPaths, inputs=()):
    output = output.resolve()
    raw_root = Path(os.path.commonpath([str(p.parent) for p in paths.values.values()]))
    if output == raw_root or raw_root in output.parents:
        raise ValueError("output may not be beneath organizer raw data")
    for root in inputs:
        root = Path(root).resolve()
        if output == root or root in output.parents or output in root.parents:
            raise ValueError("output must be separate from inputs")


def source_ids(paths: DataPaths, mode: str):
    result = {}
    for number in (1, 2, 3):
        source = f"S{number}"
        ids = set()
        with paths[f"{mode}_source{number}"].open(newline="") as stream:
            reader = csv.reader(stream, delimiter="\t")
            if next(reader, []) != ["entity_id", "business_name", "business_address", "country"]:
                raise ValueError("invalid source TSV header")
            for row in reader:
                if len(row) != 4 or not row[0].startswith(source + "-") or len(row[0]) <= 3:
                    raise ValueError("invalid source ownership")
                if row[0] in ids:
                    raise ValueError("duplicate source ID")
                ids.add(row[0])
        result[source] = ids
    return result


def feature_inputs(root: Path, mode: str):
    files = [root] if root.is_file() else sorted(root.glob("part-*.parquet"))
    manifest = root / "manifest.json"
    state = None
    if manifest.is_file():
        state = json.loads(manifest.read_text())
        if not state.get("complete"):
            raise ValueError("incomplete feature manifest")
        if state["identity"]["mode"] != mode:
            raise ValueError("feature manifest mode mismatch")
        if state["identity"]["schema_version"] != SCHEMA_VERSION:
            raise ValueError("feature schema version mismatch")
        if [p.name for p in files] != [s["file"] for s in state["shards"]]:
            raise ValueError("feature manifest shard mismatch")
    if not files and state is None:
        raise ValueError(f"no feature Parquet shards in {root}")
    fingerprints = []
    for i, path in enumerate(files):
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(OUTPUT_SCHEMA, check_metadata=False):
            raise ValueError(f"feature order/schema mismatch: {path}")
        digest = sha256(path)
        if state and (
            digest != state["shards"][i]["sha256"]
            or parquet.metadata.num_rows != state["shards"][i]["rows"]
        ):
            raise ValueError(f"corrupt feature shard: {path}")
        fingerprints.append({"path": str(path.resolve()), "sha256": digest})
    return files, fingerprints


def model_inputs(frame):
    # Explicit projection from the fixed output contract into the fixed model contract.
    if list(frame.columns) != OUTPUT_SCHEMA.names:
        raise ValueError("feature order/schema mismatch")
    return frame.loc[:, list(feature_names())]


def labels(frame, truth):
    return np.array(
        [
            int(t in truth[s])
            for s, t in frame[["s1_id", "target_id"]].itertuples(index=False, name=None)
        ],
        dtype=np.int8,
    )


def open_store(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA cache_size=-16384")
    db.execute("PRAGMA temp_store=FILE")
    return db


def stage_features(db, files, ids, batch_size=2048, truth=None, selected_ids=None):
    db.execute("DROP TABLE IF EXISTS pairs")
    db.execute(
        "CREATE TABLE pairs(s1 TEXT,target TEXT,source TEXT,payload TEXT,label INTEGER,"
        "score REAL,fold INTEGER,PRIMARY KEY(s1,target,source))"
    )
    run_id = None
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
            frame = batch.to_pandas()
            if frame.isna().any().any():
                raise ValueError("missing feature or identifier values")
            if not np.isfinite(frame[list(NUMERIC_FEATURES)].to_numpy()).all():
                raise ValueError("non-finite numeric feature")
            pending = []
            for row in frame.to_dict("records"):
                s1, target, source = (row[k] for k in KEYS)
                if s1 not in ids["S1"] or source not in ("S2", "S3") or target not in ids[source]:
                    raise ValueError("unknown ID or invalid source ownership")
                if not row["run_id"] or (run_id is not None and row["run_id"] != run_id):
                    raise ValueError("inconsistent run IDs")
                run_id = row["run_id"]
                # Validate the entire input, including rows outside a smoke subset.
                label = int(target in truth[s1]) if truth is not None else None
                keep = selected_ids is None or s1 in selected_ids
                pending.append((s1, target, source, json.dumps(row) if keep else None, label))
            try:
                db.executemany(
                    "INSERT INTO pairs(s1,target,source,payload,label) VALUES(?,?,?,?,?)", pending
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("duplicate candidate pair across shards") from exc
            db.commit()
    if selected_ids is not None:
        db.execute("CREATE TEMP TABLE selected_s1(id TEXT PRIMARY KEY)")
        db.executemany("INSERT INTO selected_s1 VALUES(?)", [(s,) for s in sorted(selected_ids)])
        db.execute("DELETE FROM pairs WHERE s1 NOT IN (SELECT id FROM selected_s1)")
        db.commit()
    return run_id


def frames(db, where="1", params=(), batch_size=2048):
    cursor = db.execute(
        f"SELECT payload FROM pairs WHERE {where} ORDER BY s1,target,source", params
    )
    while rows := cursor.fetchmany(batch_size):
        yield pd.DataFrame([json.loads(r[0]) for r in rows], columns=OUTPUT_SCHEMA.names)


def candidate_diagnostics(db, truth):
    retrieved = {s: [] for s in truth}
    for s, t in db.execute("SELECT s1,target FROM pairs WHERE label=1 ORDER BY s1,target"):
        retrieved[s].append(t)
    total = sum(map(len, truth.values()))
    found = sum(map(len, retrieved.values()))
    from entity_resolution.metrics import evaluate_macro_f0_5_efficient

    oracle, _ = evaluate_macro_f0_5_efficient(truth, retrieved)
    sources = {}
    for source in ("S2", "S3"):
        n = sum(t.startswith(source + "-") for ts in truth.values() for t in ts)
        r = sum(t.startswith(source + "-") for ts in retrieved.values() for t in ts)
        sources[source] = {
            "truth_pairs": n,
            "retrieved_pairs": r,
            "missed_pairs": n - r,
            "candidate_recall": r / n if n else None,
        }
    return {
        "total_truth_pairs": total,
        "retrieved_truth_pairs": found,
        "missed_truth_pairs": total - found,
        "pair_candidate_recall": found / total if total else None,
        "s1_full_truth_coverage": sum(set(truth[s]) == set(retrieved[s]) for s in truth)
        / len(truth),
        "no_match_s1": sum(not ts for ts in truth.values()),
        "candidate_free_s1": len(truth)
        - db.execute("SELECT COUNT(DISTINCT s1) FROM pairs").fetchone()[0],
        "candidate_oracle_macro_f0_5": oracle,
        "by_source": sources,
    }
