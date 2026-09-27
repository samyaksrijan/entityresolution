"""Disk-cached, label-free train/test retrieval with validated atomic shards.

Deadline fallback: --mode test --profile fast_submission (also exports to output/<run_id>).
Sparse example: --mode test --profile safe_k40
Resume: repeat that command with --resume (same run ID and configuration).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sqlite3
import subprocess
import time
from collections import Counter, OrderedDict
from datetime import UTC, datetime
from itertools import groupby, islice
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from scipy import sparse
from sparse_dot_topn import sp_matmul_topn

from entity_resolution.candidate_fusion import _atomic_json, _sha256, fuse_budgeted
from entity_resolution.candidate_generation import SOURCE_COLUMNS
from entity_resolution.config import load_data_paths
from entity_resolution.exact_index import content_tie_key, normalized_key
from entity_resolution.sparse_retrieval import (
    SparseRetrievalConfig,
    SparseTopNRetriever,
    _merge_topk,
    fit_vectorizer,
)
from entity_resolution.views import normalize_field

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/production_candidates.yaml"
SCHEMA_VERSION = 1
SCHEMA = pa.schema(
    [
        ("s1_id", pa.string()),
        ("target_id", pa.string()),
        ("target_source", pa.string()),
        ("channel", pa.string()),
        ("retrieval_score", pa.float32()),
        ("channel_rank", pa.int16()),
        ("exact_match", pa.bool_()),
        ("run_id", pa.string()),
        ("support_json", pa.string()),
        ("fusion_score", pa.float32()),
    ]
)
EVIDENCE_COLUMNS = [*SCHEMA.names[:8], "target_content_key"]


def configuration_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat()


def code_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def read_batches(path: Path, size: int):
    """Project only source fields; TSV parsing never guesses the separator."""
    yield from pd.read_csv(
        path, sep="\t", usecols=SOURCE_COLUMNS, dtype=str, keep_default_na=False, chunksize=size
    )


def _validate_source(frame: pd.DataFrame, source: str) -> None:
    ids = frame["entity_id"]
    if not ids.str.startswith(source + "-").all() or (ids.str.len() <= 3).any():
        raise ValueError(f"invalid identifier ownership: {source}")
    if ids.duplicated().any():
        raise ValueError("duplicate source identifier")


def _protect_outputs(output: Path, paths) -> None:
    # Common parent of all configured dataset files is the immutable organizer root.
    raw_root = Path(
        os.path.commonpath([str(p.parent) for k, p in paths.values.items() if isinstance(k, str)])
    )
    if output.resolve() == raw_root or raw_root in output.resolve().parents:
        raise ValueError("output may not be beneath the raw data root")


def validate_shard(path: Path, run_id: str, expected_rows: int | None = None) -> pa.Table:
    table = pq.read_table(path)
    if not table.schema.equals(SCHEMA, check_metadata=False):
        raise ValueError("candidate schema mismatch")
    if expected_rows is not None and table.num_rows != expected_rows:
        raise ValueError("candidate row count mismatch")
    if any(column.null_count for column in table.columns):
        raise ValueError("candidate fields may not be null")
    frame = table.to_pandas()
    if not frame["s1_id"].str.startswith("S1-").all():
        raise ValueError("invalid query identifier")
    for source, rows in frame.groupby("target_source"):
        if source not in {"S2", "S3"} or not rows["target_id"].str.startswith(source + "-").all():
            raise ValueError("candidate source ownership mismatch")
    if not frame["run_id"].eq(run_id).all():
        raise ValueError("run ID mismatch")
    if frame.duplicated(["s1_id", "target_id", "target_source"]).any():
        raise ValueError("duplicate candidate")
    if (frame["channel_rank"] < 1).any() or not np.isfinite(
        frame[["retrieval_score", "fusion_score"]].to_numpy()
    ).all():
        raise ValueError("invalid rank or score")
    ordered = frame.sort_values(["s1_id", "target_source", "target_id"], kind="stable")
    if not ordered.index.equals(frame.index):
        raise ValueError("unstable candidate ordering")
    for row in frame.itertuples(index=False):
        support = json.loads(row.support_json)
        if not support or len({v["channel"] for v in support}) != len(support):
            raise ValueError("invalid channel support")
        if row.exact_match != any(v["exact"] for v in support):
            raise ValueError("inconsistent exact provenance")
    return table


def atomic_shard(frame: pd.DataFrame, path: Path, run_id: str) -> None:
    temporary = path.with_suffix(".parquet.tmp")
    table = pa.Table.from_pandas(frame, schema=SCHEMA, preserve_index=False, safe=True)
    pq.write_table(table, temporary, compression="zstd", row_group_size=65536)
    validate_shard(temporary, run_id, len(frame))
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def volume_statistics(counts: Counter[int]) -> dict[str, float | int]:
    size = sum(counts.values())
    if not size:
        return {"unique_query_count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "maximum": 0}

    def quantile(q: float) -> float:
        position = (size - 1) * q
        lo, hi = int(np.floor(position)), int(np.ceil(position))
        values, cumulative = [], 0
        for count, frequency in sorted(counts.items()):
            previous, cumulative = cumulative, cumulative + frequency
            for rank in (lo, hi):
                if previous <= rank < cumulative:
                    values.append((rank, count))
        by_rank = dict(values)
        return float(by_rank[lo] + (by_rank[hi] - by_rank[lo]) * (position - lo))

    return {
        "unique_query_count": size,
        "mean": sum(k * v for k, v in counts.items()) / size,
        "median": quantile(0.5),
        "p95": quantile(0.95),
        "maximum": max(counts),
    }


def _build_record_database(database, paths, mode, config, *, exact_only=False):
    """Build the shared validated ID registry; fast mode indexes only Unicode exact keys."""
    temporary = database.with_suffix(".sqlite.tmp")
    temporary.unlink(missing_ok=True)
    with sqlite3.connect(temporary) as db:
        db.execute("PRAGMA cache_size=-32768")
        db.execute("PRAGMA temp_store=FILE")
        db.execute(
            "CREATE TABLE records (id TEXT PRIMARY KEY, source TEXT, country TEXT, "
            "name TEXT, address TEXT, folded_name TEXT, folded_address TEXT, tie TEXT)"
        )
        for source in ("S1", "S2", "S3"):
            for frame in read_batches(paths[f"{mode}_source{source[-1]}"], config["batch_size"]):
                _validate_source(frame, source)
                rows = []
                for row in frame.itertuples(index=False):
                    n, a = (
                        normalize_field(row.business_name),
                        normalize_field(row.business_address),
                    )
                    rows.append(
                        (
                            row.entity_id,
                            source,
                            row.country,
                            n.unicode_preserving,
                            a.unicode_preserving,
                            "" if exact_only else n.accent_folded,
                            "" if exact_only else a.accent_folded,
                            ""
                            if exact_only
                            else content_tie_key(
                                row.business_name, row.business_address, row.country
                            ),
                        )
                    )
                try:
                    db.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?)", rows)
                except sqlite3.IntegrityError as exc:
                    raise ValueError("duplicate source identifier across batches") from exc
                db.commit()
        fields = (
            ("name", "address")
            if exact_only
            else ("name", "address", "folded_name", "folded_address")
        )
        for field in fields:
            db.execute(f"CREATE INDEX idx_{field} ON records(source,country,{field},tie)")
        if not exact_only:
            db.execute("CREATE INDEX idx_order ON records(source,country,tie)")
    os.replace(temporary, database)


def _exact_rows(database, queries, run_id, maximum, *, exact_only=False):
    """Shared exact semantics; never form missing/null-like keys."""
    if queries.empty:
        return []
    output = []
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        for query in queries.itertuples(index=False):
            for source in ("S2", "S3"):
                for field in ("name", "address"):
                    value = getattr(query, "business_" + field)
                    for view, column in (
                        ("unicode_preserving", field),
                        ("accent_folded", "folded_" + field),
                    ):
                        if exact_only and view == "accent_folded":
                            continue
                        key, missing, null_like = normalized_key(value, view)
                        if missing or null_like or not key:
                            continue
                        cursor = db.execute(
                            f"SELECT id,tie FROM records WHERE source=? "
                            f"AND country=? AND {column}=? ORDER BY tie,rowid",
                            (source, query.country, key),
                        )
                        for target, tie in cursor:
                            output.append(
                                (
                                    query.entity_id,
                                    target,
                                    source,
                                    f"exact_{field}_{view}",
                                    1.0,
                                    1,
                                    True,
                                    run_id,
                                    tie,
                                )
                            )
                            if len(output) > maximum:
                                raise MemoryError("exact evidence cap exceeded; reduce shard size")
    return output


class ExactOnlyRetriever:
    """Linear TSV/SQLite preparation and indexed exact probes; no sparse dependency."""

    def __init__(self, root: Path, config: dict[str, Any]):
        self.root, self.config = root, config
        self.database = root / "records.sqlite"
        self.manifest = root / "index.json"

    def build(self, paths, mode: str, fingerprints: dict[str, str]) -> None:
        identity = {
            "inputs": fingerprints,
            "config": configuration_hash(self.config),
            "schema_version": SCHEMA_VERSION,
            "backend": "exact_only",
        }
        if self.manifest.exists():
            state = json.loads(self.manifest.read_text())
            if state["identity"] != identity:
                raise ValueError("incompatible exact index manifest")
            if _sha256(self.database) != state["files"][self.database.name]:
                raise ValueError("corrupt exact index")
            return
        self.root.mkdir(parents=True, exist_ok=True)
        _build_record_database(self.database, paths, mode, self.config, exact_only=True)
        _atomic_json(
            self.manifest,
            {
                "identity": identity,
                "files": {self.database.name: _sha256(self.database)},
                "blocks": [],
            },
        )

    def retrieve(self, queries: pd.DataFrame, run_id: str) -> pd.DataFrame:
        return pd.DataFrame(
            _exact_rows(
                self.database,
                queries,
                run_id,
                self.config["max_evidence_rows"],
                exact_only=True,
            ),
            columns=EVIDENCE_COLUMNS,
        )


def _fuse_exact(evidence, queries, policy, run_id):
    """All exact pairs survive; skip per-query fuzzy quality gates and budget sorting."""
    observations = {}
    for row in evidence.itertuples(index=False):
        if not row.exact_match or not row.channel.startswith("exact_"):
            raise ValueError("non-exact evidence in exact-only backend")
        observations.setdefault((row.s1_id, row.target_source, row.target_id), set()).add(
            row.channel
        )
    output = []
    for (query, source, target), channels in sorted(observations.items()):
        support = [{"channel": c, "score": 1.0, "rank": 1, "exact": True} for c in sorted(channels)]
        output.append(
            (
                query,
                target,
                source,
                support[0]["channel"],
                1.0,
                1,
                True,
                run_id,
                json.dumps(support, separators=(",", ":")),
                len(channels) / (float(policy.get("rrf_k", 60)) + 1),
            )
        )
    return pd.DataFrame(output, columns=SCHEMA.names), {
        "adaptive_queries": 0,
        "queries": len(queries),
        "exact_only": True,
    }


class CachedRetriever:
    """SQLite exact lookups plus reusable float32 sparse target blocks on disk.

    Memory scales with one target block and one query batch, never total corpus size.
    Fit samples and result buffers have explicit caps. One process owns a run directory.
    """

    def __init__(self, root: Path, config: dict[str, Any]):
        self.root, self.config = root, config
        self.database = root / "records.sqlite"
        self.manifest = root / "index.json"
        self._cache_limit = int(config.get("memory_cache_mb", 0)) * 2**20
        self._block_cache: OrderedDict[str, tuple[Any, pd.DataFrame, int]] = OrderedDict()
        self._cached_bytes = 0

    def _load_block(self, block: dict[str, Any], field: str):
        """Optional byte-bounded LRU; immutable matrices can stay warm on the cloud CPU."""
        key = block[field]
        if key in self._block_cache:
            self._block_cache.move_to_end(key)
            matrix, metadata, _ = self._block_cache[key]
            return matrix, metadata
        matrix = sparse.load_npz(self.root / key)
        metadata = pd.read_parquet(self.root / block["metadata"])
        if not self._cache_limit:
            return matrix, metadata
        size = sum(array.nbytes for array in (matrix.data, matrix.indices, matrix.indptr))
        size += int(metadata.memory_usage(index=True, deep=True).sum())
        if size <= self._cache_limit:
            while self._cached_bytes + size > self._cache_limit:
                _, (_, _, evicted) = self._block_cache.popitem(last=False)
                self._cached_bytes -= evicted
            self._block_cache[key] = (matrix, metadata, size)
            self._cached_bytes += size
        return matrix, metadata

    def build(self, paths, mode: str, fingerprints: dict[str, str]) -> None:
        identity = {
            "inputs": fingerprints,
            "config": configuration_hash(self.config),
            "schema_version": SCHEMA_VERSION,
        }
        if self.manifest.exists():
            state = json.loads(self.manifest.read_text())
            if state["identity"] != identity:
                raise ValueError("incompatible index manifest")
            for name, digest in state["files"].items():
                if not (self.root / name).is_file() or _sha256(self.root / name) != digest:
                    raise ValueError(f"corrupt index artifact: {name}")
            return
        self.root.mkdir(parents=True, exist_ok=True)
        # An index interrupted before its commit marker is rebuilt, never trusted.
        _build_record_database(self.database, paths, mode, self.config)
        files, blocks = {}, []
        with sqlite3.connect(self.database) as db:
            for source in ("S2", "S3"):
                vectorizers = {}
                for field in ("name", "address"):
                    spec = SparseRetrievalConfig(field=field, **self.config["sparse"])
                    asset_root = self.config.get("vectorizer_root")
                    if asset_root:
                        saved = SparseTopNRetriever.load(
                            Path(asset_root) / f"{source}_{field}_vectorizer.joblib"
                        )
                        for key in (
                            "field",
                            "view",
                            "ngram_min",
                            "ngram_max",
                            "min_df",
                            "max_features",
                        ):
                            if getattr(saved.config, key) != getattr(spec, key):
                                raise ValueError(
                                    f"incompatible fitted vectorizer: {source}/{field}/{key}"
                                )
                        vectorizer = saved.vectorizer
                    else:
                        fit = [
                            r[0]
                            for r in db.execute(
                                f"SELECT {field} FROM records WHERE source=? AND {field}!='' "
                                "ORDER BY rowid LIMIT ?",
                                (source, self.config["fit_sample_size"]),
                            )
                        ]
                        try:
                            vectorizer = (
                                fit_vectorizer(fit, spec) if len(fit) >= spec.min_df else None
                            )
                        except ValueError as exc:
                            if "empty vocabulary" not in str(exc) and "no terms remain" not in str(
                                exc
                            ):
                                raise
                            vectorizer = None
                    vectorizers[field] = vectorizer
                    name = f"{source}_{field}.joblib"
                    joblib.dump(vectorizer, self.root / name)
                    files[name] = _sha256(self.root / name)
                countries = db.execute(
                    "SELECT DISTINCT country FROM records WHERE source=? ORDER BY country",
                    (source,),
                )
                for (country,) in countries:
                    cursor = db.execute(
                        "SELECT id,name,address,tie FROM records WHERE source=? "
                        "AND country=? ORDER BY tie,rowid",
                        (source, country),
                    )
                    while rows := cursor.fetchmany(self.config["index_block_size"]):
                        block = {"source": source, "country": country, "number": len(blocks)}
                        meta_name = f"block-{len(blocks):06d}.parquet"
                        metadata = pd.DataFrame(rows, columns=["id", "name", "address", "tie"])
                        metadata["address_missing"] = metadata["address"].eq("")
                        metadata[["id", "tie", "address_missing"]].to_parquet(
                            self.root / meta_name, index=False
                        )
                        block["metadata"] = meta_name
                        files[meta_name] = _sha256(self.root / meta_name)
                        for field, position in (("name", 1), ("address", 2)):
                            vectorizer = vectorizers[field]
                            if vectorizer is None:
                                continue
                            matrix = vectorizer.transform([r[position] for r in rows]).astype(
                                np.float32
                            )
                            name = f"block-{len(blocks):06d}-{field}.npz"
                            sparse.save_npz(self.root / name, matrix)
                            block[field] = name
                            files[name] = _sha256(self.root / name)
                        blocks.append(block)
        files[self.database.name] = _sha256(self.database)
        _atomic_json(self.manifest, {"identity": identity, "files": files, "blocks": blocks})

    def retrieve(self, queries: pd.DataFrame, run_id: str) -> pd.DataFrame:
        if queries.empty:
            return pd.DataFrame(columns=EVIDENCE_COLUMNS)
        maximum = int(self.config["max_evidence_rows"])
        output = _exact_rows(self.database, queries, run_id, maximum)
        output.extend(self.sparse_rows(queries, run_id))
        if len(output) > maximum:
            raise MemoryError("evidence cap exceeded; reduce shard size")
        return pd.DataFrame(output, columns=EVIDENCE_COLUMNS)

    def sparse_rows(
        self,
        queries: pd.DataFrame,
        run_id: str,
        *,
        missing_address_only: bool = False,
        top_k: int | None = None,
        threshold: float | None = None,
    ):
        """Reuse cached matrices for base and missing-address-target name rescue."""
        queries = queries.reset_index(drop=True)
        state = json.loads(self.manifest.read_text())
        maximum = int(self.config["max_evidence_rows"])
        output = []
        threshold = self.config["sparse"]["threshold"] if threshold is None else threshold
        for source in ("S2", "S3"):
            for field in ("name",) if missing_address_only else ("name", "address"):
                k = top_k or int(self.config.get(f"{field}_top_k", self.config["sparse"]["top_k"]))
                vectorizer = joblib.load(self.root / f"{source}_{field}.joblib")
                if vectorizer is None:
                    continue
                texts = [
                    normalize_field(v).unicode_preserving for v in queries["business_" + field]
                ]
                for country, group in queries.groupby("country", sort=True):
                    positions = group.index.to_numpy()
                    qmatrix = vectorizer.transform([texts[i] for i in positions]).astype(np.float32)
                    best = [[] for _ in positions]
                    for block in state["blocks"]:
                        if (
                            block["source"] != source
                            or block["country"] != country
                            or field not in block
                        ):
                            continue
                        matrix, metadata = self._load_block(block, field)
                        if missing_address_only:
                            mask = metadata["address_missing"].to_numpy(dtype=bool)
                            matrix, metadata = matrix[mask], metadata.loc[mask]
                            if not len(metadata):
                                continue
                        matches = sp_matmul_topn(
                            qmatrix,
                            matrix.T,
                            top_n=k + 1,
                            threshold=threshold,
                            sort=True,
                            n_threads=self.config["threads"],
                        ).tocsr()
                        target_ids, ties = metadata["id"].to_numpy(), metadata["tie"].to_numpy()
                        for i in range(len(positions)):
                            start, end = matches.indptr[i : i + 2]
                            scores = matches.data[start:end]
                            indices = matches.indices[start:end]
                            if len(scores) > k and scores[k - 1] == scores[k]:
                                # Resolve only cutoff ties using a ONE-query sparse row.
                                # This is bounded by index_block_size, never a dense
                                # query-by-corpus product. Content decides tied membership.
                                full = (qmatrix[i : i + 1] @ matrix.T).tocsr()
                                keep = full.data >= scores[k - 1]
                                scores, indices = full.data[keep], full.indices[keep]
                            additions = [
                                (float(score), ties[index], target_ids[index])
                                for score, index in zip(scores, indices, strict=True)
                            ]
                            best[i] = _merge_topk(best[i], additions, k)
                    for position, matches in zip(positions, best, strict=True):
                        for rank, (score, tie, target) in enumerate(matches, 1):
                            output.append(
                                (
                                    queries.iloc[position].entity_id,
                                    target,
                                    source,
                                    (
                                        "missing_address_name_char"
                                        if missing_address_only
                                        else f"{field}_char_tfidf"
                                    ),
                                    score,
                                    rank,
                                    False,
                                    run_id,
                                    tie,
                                )
                            )
                    if len(output) > maximum:
                        raise MemoryError("evidence cap exceeded; reduce shard size")
        return output


def _refresh_totals(state: dict[str, Any]) -> None:
    counts, sources = Counter(), Counter()
    exact = rows = 0
    for shard in state["completed_shards"]:
        counts.update({int(k): v for k, v in shard["count_histogram"].items()})
        sources.update(shard["source_counts"])
        exact += shard["exact_match_count"]
        rows += shard["row_count"]
    state.update(
        volume_statistics(counts),
        source_counts=dict(sources),
        exact_match_count=exact,
        candidate_rows=rows,
    )


def generate(
    config: dict[str, Any],
    *,
    mode: str,
    run_id: str,
    output: Path,
    data_paths_config: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """No label arguments or label reads in either mode; test loads only test sources."""
    if mode not in {"train", "test"} or not run_id:
        raise ValueError("mode must be train/test and run ID must be nonempty")
    allowed = {
        "batch_size",
        "shard_size",
        "threads",
        "index_block_size",
        "fit_sample_size",
        "max_evidence_rows",
        "sparse",
        "policy",
        "rescue",
        "vectorizer_root",
        "name_top_k",
        "address_top_k",
        "memory_cache_mb",
        "retrieval_backend",
    }
    if set(config) - allowed:
        raise ValueError("unknown configuration keys (labels are not accepted)")
    if (
        not isinstance(config.get("memory_cache_mb", 0), int)
        or config.get("memory_cache_mb", 0) < 0
    ):
        raise ValueError("memory_cache_mb must be a nonnegative integer")
    for key in (
        "batch_size",
        "shard_size",
        "threads",
        "index_block_size",
        "fit_sample_size",
        "max_evidence_rows",
    ):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 < config["sparse"]["top_k"] <= 32767:
        raise ValueError("sparse top_k must fit int16")
    for field in ("name", "address"):
        value = config.get(f"{field}_top_k", config["sparse"]["top_k"])
        if not isinstance(value, int) or not 0 < value <= 32767:
            raise ValueError("field top_k must fit int16")
    if config["shard_size"] > config["batch_size"]:
        raise ValueError("shard_size may not exceed batch_size")
    permitted = {
        "sparse": {"ngram_min", "ngram_max", "min_df", "max_features", "threshold", "top_k"},
        "policy": {
            "preserve_channel_union",
            "top_k",
            "expanded_k",
            "adaptive",
            "rrf_k",
            "source_quotas",
            "channel_quotas",
            "channel_weights",
        },
        "rescue": {"enabled", "top_k", "threshold", "gate"},
    }
    for section, keys in permitted.items():
        if set(config.get(section, {})) - keys:
            raise ValueError(f"unknown {section} configuration keys; labels are not accepted")
    sparse_config = config["sparse"]
    if not (
        1 <= sparse_config["ngram_min"] <= sparse_config["ngram_max"]
        and sparse_config["min_df"] >= 1
        and sparse_config["max_features"] >= 1
        and 0 <= sparse_config["threshold"] <= 1
    ):
        raise ValueError("invalid sparse configuration")
    fuse_budgeted(
        pd.DataFrame(columns=EVIDENCE_COLUMNS),
        pd.DataFrame(columns=SOURCE_COLUMNS),
        config["policy"],
        run_id=run_id,
    )
    rescue = config.get("rescue", {})
    if rescue.get("enabled", False) and not (
        0 < rescue.get("top_k", 0) <= 32767
        and 0 <= rescue.get("threshold", -1) <= 1
        and rescue.get("gate", "weak_query") == "weak_query"
    ):
        raise ValueError("invalid rescue configuration")
    backend = config.get("retrieval_backend", "sparse")
    if backend not in {"sparse", "exact_only"}:
        raise ValueError("unknown retrieval backend")
    if backend == "exact_only" and (
        rescue.get("enabled", False)
        or set(config["policy"]) - {"top_k", "rrf_k", "preserve_channel_union"}
    ):
        raise ValueError("exact-only backend does not accept rescue or fuzzy fusion policies")
    paths = load_data_paths(data_paths_config)
    _protect_outputs(output, paths)
    fingerprints = {f"{mode}_source{i}": _sha256(paths[f"{mode}_source{i}"]) for i in (1, 2, 3)}
    if backend == "sparse" and config.get("vectorizer_root"):
        for source in ("S2", "S3"):
            for field in ("name", "address"):
                asset = Path(config["vectorizer_root"]) / f"{source}_{field}_vectorizer.joblib"
                fingerprints[f"vectorizer_{source}_{field}"] = _sha256(asset)
    identity = {
        "run_id": run_id,
        "configuration_hash": configuration_hash(config),
        "input_fingerprints": fingerprints,
        "schema_version": SCHEMA_VERSION,
        "implementation_hash": configuration_hash(
            {
                name: _sha256(Path(__file__).with_name(name))
                for name in (
                    "production_candidates.py",
                    "candidate_fusion.py",
                    "sparse_retrieval.py",
                    "exact_index.py",
                    "views.py",
                    "retrieval_rescue.py",
                )
            }
        ),
        "mode": mode,
        "runtime_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "sparse-dot-topn", "pyarrow", "pandas")
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    # Exclusive OS lock is released after interruption; concurrent writers cannot interleave.
    import fcntl

    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("run directory already locked by another writer") from exc
        if manifest.exists():
            if not resume:
                raise ValueError("run exists; use --resume")
            state = json.loads(manifest.read_text())
            if any(state.get(k) != value for k, value in identity.items()):
                raise ValueError("incompatible manifest")
            for number, shard in enumerate(state["completed_shards"]):
                if shard["file"] != f"part-{number:06d}.parquet":
                    raise ValueError("noncontiguous shard manifest")
                path = output / shard["file"]
                if not path.is_file() or _sha256(path) != shard["sha256"]:
                    raise ValueError("corrupt or missing completed shard")
                validate_shard(path, run_id, shard["row_count"])
            known = {shard["file"] for shard in state["completed_shards"]}
            orphaned = {path.name for path in output.glob("part-*.parquet")} - known
            # A crash after rename but before manifest commit may leave exactly the next
            # shard. Replay it from immutable inputs; never silently adopt its bytes.
            permitted_orphan = (
                {f"part-{len(known):06d}.parquet"} if not state["complete"] else set()
            )
            if orphaned - permitted_orphan:
                raise ValueError("unexpected unmanifested shard")
            if state["complete"]:
                _refresh_totals(state)
                return state
        else:
            if list(output.glob("part-*.parquet")):
                raise ValueError("orphan shards without manifest")
            state = {
                **identity,
                "code_revision": code_revision(),
                "configuration": config,
                "started_at": now(),
                "ended_at": None,
                "complete": False,
                "completed_shards": [],
                "data_paths_config": str(data_paths_config.resolve()),
            }
            _atomic_json(manifest, state)
        retriever_class = ExactOnlyRetriever if backend == "exact_only" else CachedRetriever
        retriever = retriever_class(output / "index", config)
        retriever.build(paths, mode, fingerprints)
        offset = sum(s["query_count"] for s in state["completed_shards"])
        seen = 0
        # Fixed shard boundaries are independent of parser and retrieval batch sizes.
        for queries in read_batches(paths[f"{mode}_source1"], config["shard_size"]):
            if seen < offset:
                seen += len(queries)
                continue
            queries = queries.reset_index(drop=True)
            _validate_source(queries, "S1")
            started = time.perf_counter()
            evidence = retriever.retrieve(queries, run_id)
            if config.get("rescue", {}).get("enabled", False):
                from entity_resolution.retrieval_rescue import retrieve_missing_address

                extra, rescue_stats = retrieve_missing_address(
                    retriever, queries, evidence, run_id, config["rescue"]
                )
                evidence = pd.concat([evidence, extra], ignore_index=True)
            else:
                rescue_stats = {"activated_queries": 0}
            if backend == "exact_only":
                selected, gating = _fuse_exact(evidence, queries, config["policy"], run_id)
            else:
                selected, gating = fuse_budgeted(evidence, queries, config["policy"], run_id=run_id)
            filename = f"part-{len(state['completed_shards']):06d}.parquet"
            atomic_shard(selected, output / filename, run_id)
            counts = selected.groupby("s1_id").size().reindex(queries.entity_id, fill_value=0)
            shard = {
                "file": filename,
                "sha256": _sha256(output / filename),
                "row_count": len(selected),
                "query_count": len(queries),
                "query_ids_sha256": configuration_hash(queries.entity_id.tolist()),
                "count_histogram": dict(Counter(map(int, counts))),
                "volume": volume_statistics(Counter(map(int, counts))),
                "source_counts": selected.target_source.value_counts().to_dict(),
                "exact_match_count": int(selected.exact_match.sum()),
                "wall_seconds": time.perf_counter() - started,
                "rss_mb_at_end": psutil.Process().memory_info().rss / 2**20,
                "finished_at": now(),
                "gating": gating,
                "rescue": rescue_stats,
            }
            state["completed_shards"].append(shard)
            _refresh_totals(state)
            _atomic_json(manifest, state)
            seen += len(queries)
        state.update(complete=True, ended_at=now())
        _refresh_totals(state)
        _atomic_json(manifest, state)
        return state


def export_fallback(output: Path, submission_root: Path, run_id: str) -> Path:
    """Conservative exact-name AND exact-address fallback through official APIs.

    Pair maps live on disk. The existing writer sorts all S1 IDs in memory, so this
    optional final export needs O(number of S1 IDs) RAM, not O(candidate pairs).
    This is a format-valid fallback, not a calibrated competition-quality model.
    """
    from collections.abc import Mapping, Set

    from entity_resolution.decode import Decoder
    from entity_resolution.submission import write_submission

    state = json.loads((output / "manifest.json").read_text())
    if not state["complete"] or state["mode"] != "test":
        raise ValueError("submission requires a complete test run")
    database = output / "index/records.sqlite"
    index_state = json.loads((output / "index/index.json").read_text())
    if _sha256(database) != index_state["files"][database.name]:
        raise ValueError("corrupt source index")
    config_paths = state["data_paths_config"]
    _protect_outputs(submission_root, load_data_paths(Path(config_paths)))
    with sqlite3.connect(output / "submission.sqlite") as db:
        db.execute("ATTACH DATABASE ? AS records", (str(database),))
        db.executescript(
            "DROP TABLE IF EXISTS pairs; DROP TABLE IF EXISTS matches; "
            "CREATE TABLE pairs(s1 TEXT,target TEXT,source TEXT,score REAL); "
            "CREATE TABLE matches(s1 TEXT,target TEXT);"
        )
        for shard in state["completed_shards"]:
            path = output / shard["file"]
            if _sha256(path) != shard["sha256"]:
                raise ValueError("corrupt completed shard")
            table = validate_shard(path, state["run_id"], shard["row_count"])
            rows = []
            for row in table.to_pandas().itertuples(index=False):
                channels = {v["channel"] for v in json.loads(row.support_json) if v["exact"]}
                score = float(
                    any(c.startswith("exact_name_") for c in channels)
                    and any(c.startswith("exact_address_") for c in channels)
                )
                rows.append((row.s1_id, row.target_id, row.target_source, score))
            db.executemany("INSERT INTO pairs VALUES(?,?,?,?)", rows)
        db.execute("CREATE INDEX pairs_s1 ON pairs(s1)")
        db.execute("CREATE INDEX pairs_target ON pairs(target)")
        decoder = Decoder(global_threshold=1.0, enforce_target_ownership=True, score_margin=0.01)
        # Keep each ownership group complete while amortizing DataFrame/decoder setup.
        # The authoritative decoder still resolves all cross-query target conflicts.
        cursor = db.execute(
            "SELECT s1,target,source,score FROM pairs WHERE score=1 ORDER BY target,s1"
        )
        pending = []

        def decode_batch(rows):
            frame = pd.DataFrame(rows, columns=["s1_id", "target_id", "target_source", "score"])
            decoded = decoder.decode(frame, set(frame.s1_id))
            db.executemany(
                "INSERT INTO matches VALUES(?,?)",
                ((s1, t) for s1, targets in decoded.items() for t in targets),
            )

        for _, group in groupby(cursor, key=lambda row: row[1]):
            rows = list(islice(group, 100001))
            if len(rows) > 100000:
                raise MemoryError("pathological exact-owner group exceeds export memory guard")
            if pending and len(pending) + len(rows) > 10000:
                decode_batch(pending)
                pending = []
            pending.extend(rows)
        if pending:
            decode_batch(pending)
        db.execute("CREATE INDEX matches_s1 ON matches(s1)")
        db.commit()

        class IdSet(Set):
            def __init__(self, source):
                self.condition = "source='S1'" if source == "S1" else "source IN ('S2','S3')"

            def __contains__(self, value):
                return (
                    db.execute(
                        "SELECT 1 FROM records.records WHERE id=? AND " + self.condition, (value,)
                    ).fetchone()
                    is not None
                )

            def __iter__(self):
                return (
                    r[0]
                    for r in db.execute("SELECT id FROM records.records WHERE " + self.condition)
                )

            def __len__(self):
                return db.execute(
                    "SELECT COUNT(*) FROM records.records WHERE " + self.condition
                ).fetchone()[0]

        class PairMap(Mapping):
            def __init__(self, table):
                self.table = table

            def __iter__(self):
                return (r[0] for r in db.execute(f"SELECT DISTINCT s1 FROM {self.table}"))

            def __len__(self):
                return db.execute(f"SELECT COUNT(DISTINCT s1) FROM {self.table}").fetchone()[0]

            def __getitem__(self, key):
                return [
                    r[0] for r in db.execute(f"SELECT target FROM {self.table} WHERE s1=?", (key,))
                ]

        return write_submission(
            IdSet("S1"),
            IdSet("target"),
            PairMap("pairs"),
            PairMap("matches"),
            submission_root,
            run_id=run_id,
        )


def load_profile(path: Path, profile: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    config = {**payload["runtime"], "policy": payload["profiles"][profile]}
    config.update(payload.get("profile_runtime", {}).get(profile, {}))
    config["rescue"] = payload.get("profile_rescue", {}).get(profile, {"enabled": False})
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--data-paths", type=Path, default=DEFAULT_CONFIG.with_name("data_paths.yaml")
    )
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument(
        "--profile",
        choices=[
            "fast_submission",
            "safe_k40",
            "emergency_k20",
            "stretch_adaptive",
            "competitive_adaptive",
            "max_score",
        ],
        default="safe_k40",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--submission-output",
        type=Path,
        help=(
            "Export exact fallback via existing decoder/writer; "
            "fast test profile defaults to output"
        ),
    )
    parser.add_argument(
        "--enable-rescue",
        action="store_true",
        help="Enable the bounded missing-address name rescue",
    )
    parser.add_argument(
        "--rescue-config",
        type=Path,
        help="Explicit rescue override with --enable-rescue; otherwise retain profile defaults",
    )
    parser.add_argument(
        "--vectorizer-root", type=Path, help="Directory of existing S2/S3 name/address vectorizers"
    )
    parser.add_argument("--threads", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--memory-cache-mb", type=int, help="Bounded in-memory sparse block cache")
    parser.add_argument(
        "--top-k", type=int, help="Override final budget; clears reservations/adaptation"
    )
    args = parser.parse_args()
    config = load_profile(args.config, args.profile)
    if args.vectorizer_root is not None:
        config["vectorizer_root"] = str(args.vectorizer_root)
    if args.enable_rescue:
        if args.rescue_config is not None or not config["rescue"].get("enabled", False):
            rescue_path = args.rescue_config or DEFAULT_CONFIG.with_name("retrieval_rescue.yaml")
            config["rescue"] = yaml.safe_load(rescue_path.read_text())["rescue"]
        config["rescue"]["enabled"] = True
    for key in ("threads", "batch_size", "shard_size", "memory_cache_mb"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.top_k is not None:
        config["policy"] = {"top_k": args.top_k}
    run_id = args.run_id or f"{args.mode}_{args.profile}"
    state = generate(
        config,
        mode=args.mode,
        run_id=run_id,
        output=args.output or Path("artifacts/production_candidates") / run_id,
        data_paths_config=args.data_paths,
        resume=args.resume,
    )
    if args.profile == "fast_submission" and args.mode == "test" and not args.submission_output:
        args.submission_output = Path("output")
    if args.submission_output:
        export_fallback(
            args.output or Path("artifacts/production_candidates") / run_id,
            args.submission_output,
            run_id,
        )
    print(
        json.dumps(
            {key: state[key] for key in ("run_id", "complete", "candidate_rows", "mean")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
