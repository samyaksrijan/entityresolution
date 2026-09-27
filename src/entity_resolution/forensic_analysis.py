# ruff: noqa: E501
"""Chunked, leakage-safe forensic analysis for the supplied competition data.

Run from the repository root with::

    python -m entity_resolution.forensic_analysis

The command never writes below the configured organizer-data root. Large intermediate
arrays and the temporary SQLite join database are removed after successful completion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz.fuzz import ratio
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from entity_resolution.config import DataPaths, load_data_paths
from entity_resolution.io import GROUND_TRUTH_COLUMNS, SOURCE_COLUMNS, validate_all
from entity_resolution.views import (
    digit_sequence,
    punctuation_normalized,
    word_tokens,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 20260927
CHUNK_SIZE = 100_000
NULL_LIKE = {"null", "none", "nan", "n/a", "na", "nil", "<null>", "missing"}
SOURCE_KEYS = (
    "train_source1",
    "train_source2",
    "train_source3",
    "test_source1",
    "test_source2",
    "test_source3",
)
SCRIPT_NAMES = ("Latin", "Devanagari", "OtherLetter", "Mixed", "NoLetters")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _stable_u64(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "little")


def _vector_normalize(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.normalize("NFKC")
        .str.casefold()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _series_hash(series: pd.Series) -> np.ndarray:
    return pd.util.hash_pandas_object(series, index=False, categorize=True).to_numpy(np.uint64)


def _frame_hash(frame: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(frame, index=False, categorize=True).to_numpy(np.uint64)


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


@dataclass
class ResourceMonitor:
    started: float = dc_field(default_factory=time.perf_counter)
    peak_mb: float = dc_field(default_factory=_rss_mb)

    def sample(self) -> None:
        self.peak_mb = max(self.peak_mb, _rss_mb())

    @property
    def runtime_seconds(self) -> float:
        return time.perf_counter() - self.started


def _row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(block.count(b"\n") for block in iter(lambda: handle.read(8 * 1024 * 1024), b"")) - 1


def _read_chunks(path: Path, *, chunksize: int = CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(
        path,
        sep="\t",
        dtype={column: "string" for column in SOURCE_COLUMNS},
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8",
        chunksize=chunksize,
    )


def _script(value: str) -> str:
    scripts: set[str] = set()
    for char in value:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if "LATIN" in name:
            scripts.add("Latin")
        elif "DEVANAGARI" in name:
            scripts.add("Devanagari")
        else:
            scripts.add("OtherLetter")
    if not scripts:
        return "NoLetters"
    if len(scripts) > 1:
        return "Mixed"
    return next(iter(scripts))


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0}
    array = np.asarray(values)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": int(array.max()),
    }


def _duplicate_metrics(values: np.memmap, rows: int) -> dict[str, int]:
    ordered = np.sort(np.asarray(values[:rows]))
    if not len(ordered):
        return {"unique": 0, "duplicate_excess": 0, "duplicate_groups": 0, "rows_in_duplicate_groups": 0}
    boundaries = np.flatnonzero(ordered[1:] != ordered[:-1]) + 1
    counts = np.diff(np.r_[0, boundaries, len(ordered)])
    duplicate = counts[counts > 1]
    return {
        "unique": int(len(counts)),
        "duplicate_excess": int(rows - len(counts)),
        "duplicate_groups": int(len(duplicate)),
        "rows_in_duplicate_groups": int(duplicate.sum()),
    }


def profile_source(
    key: str,
    path: Path,
    work_dir: Path,
    monitor: ResourceMonitor,
) -> tuple[dict[str, Any], Path]:
    """Profile one source exactly for row-level rates and with deterministic token sampling."""
    rows = _row_count(path)
    hash_path = work_dir / f"{key}_hashes.dat"
    hashes = np.memmap(hash_path, mode="w+", dtype=np.uint64, shape=(rows, 6))
    countries: Counter[str] = Counter()
    token_counter: Counter[str] = Counter()
    script_counter: Counter[str] = Counter()
    sampled_rows = 0
    lengths: dict[str, list[int]] = {"name": [], "address": []}
    token_counts: dict[str, list[int]] = {"name": [], "address": []}
    totals = Counter()
    memory_bytes = 0
    offset = 0
    started = time.perf_counter()

    for chunk in _read_chunks(path):
        size = len(chunk)
        memory_bytes += int(chunk.memory_usage(index=True, deep=True).sum())
        name = chunk["business_name"]
        address = chunk["business_address"]
        name_stripped = name.str.strip()
        address_stripped = address.str.strip()
        name_norm = _vector_normalize(name)
        address_norm = _vector_normalize(address)

        for label, values, stripped in (
            ("name", name, name_stripped),
            ("address", address, address_stripped),
        ):
            empty = values.eq("")
            whitespace = ~empty & stripped.eq("")
            null_like = stripped.str.casefold().isin(NULL_LIKE)
            totals[f"{label}_empty"] += int(empty.sum())
            totals[f"{label}_whitespace_only"] += int(whitespace.sum())
            totals[f"{label}_null_like"] += int(null_like.sum())
            totals[f"{label}_non_ascii"] += int(values.str.contains(r"[^\x00-\x7f]", regex=True).sum())
            totals[f"{label}_digits"] += int(values.str.contains(r"\d", regex=True).sum())
        totals["both_missing"] += int((name_stripped.eq("") & address_stripped.eq("")).sum())
        countries.update(chunk["country"].tolist())

        hashes[offset : offset + size, 0] = _series_hash(name)
        hashes[offset : offset + size, 1] = _series_hash(address)
        hashes[offset : offset + size, 2] = _series_hash(name_norm)
        hashes[offset : offset + size, 3] = _series_hash(address_norm)
        hashes[offset : offset + size, 4] = _frame_hash(chunk)
        hashes[offset : offset + size, 5] = np.fromiter(
            (_stable_u64(value) for value in chunk["entity_id"]), dtype=np.uint64, count=size
        )

        selected = (_series_hash(chunk["entity_id"]) % 100) < 2
        sample = chunk.loc[selected]
        sampled_rows += len(sample)
        for raw_name, raw_address in sample[["business_name", "business_address"]].itertuples(
            index=False, name=None
        ):
            for label, value in (("name", raw_name), ("address", raw_address)):
                tokens = word_tokens(value)
                token_counter.update(f"{label}:{token}" for token in tokens)
                token_counts[label].append(len(tokens))
                lengths[label].append(len(value))
                script_counter[f"{label}:{_script(value)}"] += 1
                folded = unicodedata.normalize("NFKD", value)
                if any(unicodedata.combining(char) for char in folded):
                    totals[f"{label}_accented_sample"] += 1
        offset += size
        monitor.sample()

    hashes.flush()
    if offset != rows:
        raise RuntimeError(f"row-count drift for {key}: expected {rows}, read {offset}")
    id_hash_path = work_dir / f"{key}_id_hashes.npy"
    ordered_ids = np.sort(np.asarray(hashes[:, 5]))
    np.save(id_hash_path, ordered_ids)

    duplicate_labels = {
        "exact_name": 0,
        "exact_address": 1,
        "normalized_name": 2,
        "normalized_address": 3,
        "record": 4,
    }
    duplicates = {
        label: _duplicate_metrics(hashes[:, column], rows)
        for label, column in duplicate_labels.items()
    }
    common_tokens = [
        {"field_token": token, "count_in_2pct_sample": count}
        for token, count in token_counter.most_common(30)
    ]
    rare_tokens = sum(count == 1 for count in token_counter.values())
    result: dict[str, Any] = {
        "key": key,
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "rows": rows,
        "dataframe_memory_bytes": memory_bytes,
        "missing": {
            "name": totals["name_empty"] + totals["name_whitespace_only"],
            "address": totals["address_empty"] + totals["address_whitespace_only"],
            "both": totals["both_missing"],
        },
        "missing_rates": {
            "name": (totals["name_empty"] + totals["name_whitespace_only"]) / rows,
            "address": (totals["address_empty"] + totals["address_whitespace_only"]) / rows,
            "both": totals["both_missing"] / rows,
        },
        "empty_whitespace_null_like": dict(totals),
        "countries": dict(sorted(countries.items())),
        "sample": {
            "method": "deterministic pandas-hash(entity_id) modulo 100 < 2",
            "rows": sampled_rows,
            "name_length": _distribution(lengths["name"]),
            "address_length": _distribution(lengths["address"]),
            "name_token_count": _distribution(token_counts["name"]),
            "address_token_count": _distribution(token_counts["address"]),
            "scripts": dict(sorted(script_counter.items())),
            "common_tokens": common_tokens,
            "rare_token_types": rare_tokens,
            "accent_rates": {
                "name": totals["name_accented_sample"] / max(sampled_rows, 1),
                "address": totals["address_accented_sample"] / max(sampled_rows, 1),
            },
        },
        "full_rates": {
            "name_non_ascii": totals["name_non_ascii"] / rows,
            "address_non_ascii": totals["address_non_ascii"] / rows,
            "name_has_digit": totals["name_digits"] / rows,
            "address_has_digit": totals["address_digits"] / rows,
        },
        "duplicates": duplicates,
        "runtime_seconds": time.perf_counter() - started,
    }
    del hashes, ordered_ids
    hash_path.unlink()
    return result, id_hash_path


def _hash_membership(values: np.ndarray, ordered: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(ordered, values)
    valid = positions < len(ordered)
    result = np.zeros(len(values), dtype=bool)
    result[valid] = ordered[positions[valid]] == values[valid]
    return result


def _issue(
    issues: list[dict[str, Any]],
    *,
    severity: str,
    dataset: str,
    category: str,
    count: int,
    detail: str,
) -> None:
    issues.append(
        {
            "severity": severity,
            "dataset": dataset,
            "category": category,
            "count": count,
            "detail": detail,
        }
    )


def inspect_truth(
    path: Path,
    id_hash_paths: dict[str, Path],
    issues: list[dict[str, Any]],
    work_dir: Path,
    monitor: ResourceMonitor,
) -> dict[str, Any]:
    """Stream ground truth and validate all references and ownership."""
    s1_ids = np.load(id_hash_paths["train_source1"], mmap_mode="r")
    s2_ids = np.load(id_hash_paths["train_source2"], mmap_mode="r")
    s3_ids = np.load(id_hash_paths["train_source3"], mmap_mode="r")
    rows = _row_count(path)
    cardinality = Counter()
    per_source = Counter()
    singleton = 0
    both_sources = 0
    unknown_s1 = 0
    unknown_target = 0
    duplicate_within = 0
    s1_as_target = 0
    edge_count = 0
    seen_truth_s1: set[int] = set()
    s1_batch: list[int] = []
    s2_batch: list[int] = []
    s3_batch: list[int] = []

    def flush_membership() -> None:
        nonlocal unknown_s1, unknown_target
        if s1_batch:
            values = np.asarray(s1_batch, dtype=np.uint64)
            unknown_s1 += int((~_hash_membership(values, s1_ids)).sum())
            s1_batch.clear()
        if s2_batch:
            values = np.asarray(s2_batch, dtype=np.uint64)
            unknown_target += int((~_hash_membership(values, s2_ids)).sum())
            s2_batch.clear()
        if s3_batch:
            values = np.asarray(s3_batch, dtype=np.uint64)
            unknown_target += int((~_hash_membership(values, s3_ids)).sum())
            s3_batch.clear()

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            s1 = row[GROUND_TRUTH_COLUMNS[0]]
            raw = row[GROUND_TRUTH_COLUMNS[1]]
            owner = _stable_u64(s1)
            if owner in seen_truth_s1:
                _issue(
                    issues,
                    severity="error",
                    dataset="train_ground_truth",
                    category="duplicate_source1_id",
                    count=1,
                    detail="A source1_entity_id occurs on multiple truth rows.",
                )
            seen_truth_s1.add(owner)
            s1_batch.append(owner)
            matches = [] if raw == "" else raw.split(",")
            if len(matches) != len(set(matches)):
                duplicate_within += len(matches) - len(set(matches))
            source_types = {match[:2] for match in matches}
            if "S2" in source_types and "S3" in source_types:
                both_sources += 1
            if not matches:
                singleton += 1
            n = len(matches)
            cardinality[str(n) if n < 4 else "4+"] += 1
            for match in matches:
                if match.startswith("S1-"):
                    s1_as_target += 1
                if match.startswith("S2-"):
                    per_source["S1-S2"] += 1
                    target_batch = s2_batch
                elif match.startswith("S3-"):
                    per_source["S1-S3"] += 1
                    target_batch = s3_batch
                else:
                    unknown_target += 1
                    continue
                hashed = _stable_u64(match)
                target_batch.append(hashed)
            edge_count += n
            if len(seen_truth_s1) % CHUNK_SIZE == 0:
                flush_membership()
                monitor.sample()

    flush_membership()

    for count, category in (
        (unknown_s1, "unknown_source1_reference"),
        (unknown_target, "unknown_target_reference"),
        (duplicate_within, "duplicate_id_within_truth_list"),
        (s1_as_target, "source1_id_used_as_target"),
    ):
        _issue(
            issues,
            severity="error" if count else "pass",
            dataset="train_ground_truth",
            category=category,
            count=count,
            detail="Measured by full streaming scan and hash membership against configured sources.",
        )
    return {
        "rows": rows,
        "positive_edges": edge_count,
        "singletons": singleton,
        "singleton_rate": singleton / rows,
        "cardinality": dict(cardinality),
        "per_source_edges": dict(per_source),
        "matched_in_both_sources": both_sources,
        "unknown_source1_references": unknown_s1,
        "unknown_target_references": unknown_target,
        "duplicate_ids_within_lists": duplicate_within,
        "source1_ids_as_targets": s1_as_target,
    }


def _sqlite_batches(rows: Iterable[tuple[Any, ...]], size: int = 25_000) -> Iterator[list[tuple[Any, ...]]]:
    batch: list[tuple[Any, ...]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_join_database(paths: DataPaths, database: Path, monitor: ResourceMonitor) -> sqlite3.Connection:
    """Create a temporary indexed store for full positive-pair joins."""
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        PRAGMA cache_size=-131072;
        CREATE TABLE records (
            entity_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            address TEXT NOT NULL,
            country TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE edges (s1_id TEXT NOT NULL, target_id TEXT NOT NULL, target_source TEXT NOT NULL);
        """
    )
    for key in ("train_source1", "train_source2", "train_source3"):
        for chunk in _read_chunks(paths[key], chunksize=50_000):
            connection.executemany(
                "INSERT INTO records VALUES (?, ?, ?, ?)",
                chunk.loc[:, SOURCE_COLUMNS].itertuples(index=False, name=None),
            )
            connection.commit()
            monitor.sample()
    with paths["train_ground_truth"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")

        def edge_rows() -> Iterator[tuple[str, str, str]]:
            for row in reader:
                for target in ([] if row["matched_entity_ids"] == "" else row["matched_entity_ids"].split(",")):
                    yield row["source1_entity_id"], target, target[:2]

        for batch in _sqlite_batches(edge_rows()):
            connection.executemany("INSERT INTO edges VALUES (?, ?, ?)", batch)
            connection.commit()
            monitor.sample()
    connection.execute("CREATE INDEX edges_s1_idx ON edges(s1_id)")
    connection.execute("CREATE INDEX edges_target_idx ON edges(target_id)")
    connection.commit()
    return connection


def _token_jaccard(left: str, right: str) -> float:
    a, b = set(word_tokens(left)), set(word_tokens(right))
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def _length_ratio(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    return min(len(left), len(right)) / max(len(left), len(right), 1)


def _pair_features(row: tuple[str, ...]) -> dict[str, Any]:
    s1_id, target_id, target_source, n1, a1, c1, n2, a2, c2 = row
    nn1, nn2 = punctuation_normalized(n1), punctuation_normalized(n2)
    aa1, aa2 = punctuation_normalized(a1), punctuation_normalized(a2)
    d1, d2 = digit_sequence(a1), digit_sequence(a2)
    digit_overlap = len(set(d1) & set(d2)) / max(len(set(d1) | set(d2)), 1)
    digit_conflict = bool(d1 and d2 and not set(d1).intersection(d2))
    return {
        "source1_entity_id": s1_id,
        "matched_entity_id": target_id,
        "target_source": target_source,
        "country": c1,
        "country_agreement": c1 == c2,
        "name_raw_exact": n1 == n2,
        "address_raw_exact": a1 == a2,
        "name_normalized_exact": nn1 == nn2,
        "address_normalized_exact": aa1 == aa2,
        "name_char_similarity": ratio(nn1, nn2) / 100.0,
        "address_char_similarity": ratio(aa1, aa2) / 100.0,
        "name_token_jaccard": _token_jaccard(n1, n2),
        "address_token_jaccard": _token_jaccard(a1, a2),
        "digit_overlap": digit_overlap,
        "digit_conflict": digit_conflict,
        "s1_name_missing": not n1.strip(),
        "target_name_missing": not n2.strip(),
        "s1_address_missing": not a1.strip(),
        "target_address_missing": not a2.strip(),
        "name_length_ratio": _length_ratio(n1, n2),
        "address_length_ratio": _length_ratio(a1, a2),
    }


def write_positive_analysis(
    connection: sqlite3.Connection,
    output: Path,
    monitor: ResourceMonitor,
) -> dict[str, Any]:
    """Materialize full positive-pair features and aggregate source/country differences."""
    query = """
        SELECT e.s1_id, e.target_id, e.target_source,
               s.name, s.address, s.country, t.name, t.address, t.country
        FROM edges e
        JOIN records s ON s.entity_id=e.s1_id
        JOIN records t ON t.entity_id=e.target_id
    """
    cursor = connection.execute(query)
    writer: pq.ParquetWriter | None = None
    aggregates: dict[str, Counter[str]] = defaultdict(Counter)
    counts = Counter()
    rows_written = 0
    while True:
        rows = cursor.fetchmany(25_000)
        if not rows:
            break
        features = [_pair_features(row) for row in rows]
        table = pa.Table.from_pylist(features)
        if writer is None:
            writer = pq.ParquetWriter(output, table.schema, compression="zstd")
        writer.write_table(table)
        for feature in features:
            group = f"{feature['target_source']}|{feature['country']}"
            counts[group] += 1
            for metric_field in (
                "name_raw_exact",
                "address_raw_exact",
                "name_normalized_exact",
                "address_normalized_exact",
                "digit_conflict",
            ):
                aggregates[group][metric_field] += int(feature[metric_field])
        rows_written += len(features)
        monitor.sample()
    if writer is not None:
        writer.close()
    rates = {
        group: {field: value / counts[group] for field, value in counter.items()}
        for group, counter in aggregates.items()
    }
    return {"rows": rows_written, "group_counts": dict(counts), "group_rates": rates}


def summarize_pair_parquet(path: Path, group_field: str) -> dict[str, Any]:
    """Stream aggregate boolean/numeric comparison features by a categorical field."""
    parquet = pq.ParquetFile(path)
    excluded = {"source1_entity_id", "matched_entity_id", group_field, "target_source", "country"}
    sums: dict[str, Counter[str]] = defaultdict(Counter)
    counts = Counter()
    columns = parquet.schema_arrow.names
    numeric = [
        field.name
        for field in parquet.schema_arrow
        if field.name not in excluded
        and (pa.types.is_boolean(field.type) or pa.types.is_floating(field.type))
    ]
    for batch in parquet.iter_batches(columns=[group_field, *numeric], batch_size=100_000):
        frame = batch.to_pandas()
        for group, values in frame.groupby(group_field, sort=False):
            label = str(group)
            counts[label] += len(values)
            for column in numeric:
                sums[label][column] += float(values[column].sum())
    return {
        "rows": parquet.metadata.num_rows,
        "group_field": group_field,
        "group_counts": dict(counts),
        "group_means": {
            group: {metric: value / counts[group] for metric, value in metrics.items()}
            for group, metrics in sums.items()
        },
        "columns": columns,
    }


def truth_structure_from_database(connection: sqlite3.Connection) -> dict[str, Any]:
    conflicts = connection.execute(
        "SELECT COUNT(*) FROM (SELECT target_id FROM edges GROUP BY target_id HAVING COUNT(DISTINCT s1_id)>1)"
    ).fetchone()[0]
    by_country = {
        row[0]: {"s1_entities": row[1], "positive_edges": row[2]}
        for row in connection.execute(
            """
            SELECT r.country, COUNT(DISTINCT r.entity_id), COUNT(e.target_id)
            FROM records r LEFT JOIN edges e ON r.entity_id=e.s1_id
            WHERE r.entity_id LIKE 'S1-%' GROUP BY r.country
            """
        )
    }
    by_missingness = {
        row[0]: {"s1_entities": row[1], "positive_edges": row[2]}
        for row in connection.execute(
            """
            SELECT CASE WHEN trim(r.name)='' AND trim(r.address)='' THEN 'both_missing'
                        WHEN trim(r.name)='' THEN 'name_missing'
                        WHEN trim(r.address)='' THEN 'address_missing'
                        ELSE 'complete' END,
                   COUNT(DISTINCT r.entity_id), COUNT(e.target_id)
            FROM records r LEFT JOIN edges e ON r.entity_id=e.s1_id
            WHERE r.entity_id LIKE 'S1-%' GROUP BY 1
            """
        )
    }
    return {
        "conflicting_target_ownership": int(conflicts),
        "by_country": by_country,
        "by_missingness": by_missingness,
    }


def _sample_s1(connection: sqlite3.Connection, limit: int) -> list[tuple[str, str, str, str]]:
    rows = connection.execute(
        "SELECT entity_id, name, address, country FROM records WHERE entity_id LIKE 'S1-%'"
    )
    selected = [row for row in rows if _stable_u64(row[0]) % 10_000 < 10]
    selected.sort(key=lambda row: _stable_u64(row[0]))
    return selected[:limit]


def _candidate_summary(candidate_sets: dict[str, set[str]], true_sets: dict[str, set[str]]) -> dict[str, Any]:
    counts = [len(candidate_sets.get(s1, set())) for s1 in true_sets]
    true_edges = sum(len(values) for values in true_sets.values())
    recalled = sum(len(candidate_sets.get(s1, set()) & values) for s1, values in true_sets.items())
    covered = sum(values <= candidate_sets.get(s1, set()) for s1, values in true_sets.items())
    return {
        "positive_edge_recall": recalled / max(true_edges, 1),
        "all_true_matches_per_s1_coverage": covered / max(len(true_sets), 1),
        "average_candidates": statistics.fmean(counts) if counts else 0.0,
        "median_candidates": statistics.median(counts) if counts else 0.0,
        "p95_candidates": float(np.quantile(counts, 0.95)) if counts else 0.0,
        "maximum_candidates": max(counts, default=0),
        "sample_s1": len(true_sets),
        "sample_positive_edges": true_edges,
    }


def exact_blocking_experiment(
    paths: DataPaths,
    connection: sqlite3.Connection,
    sample_limit: int,
    monitor: ResourceMonitor,
) -> tuple[dict[str, Any], dict[str, dict[str, set[str]]], dict[str, dict[str, set[str]]]]:
    """Evaluate exact and digit channels on sampled S1 queries against every target row."""
    sample = _sample_s1(connection, sample_limit)
    sample_ids = {row[0] for row in sample}
    truth: dict[str, dict[str, set[str]]] = {"S2": {}, "S3": {}}
    for source in truth:
        truth[source] = {s1: set() for s1 in sample_ids}
    placeholders = ",".join("?" for _ in sample_ids)
    for s1, target, source in connection.execute(
        f"SELECT s1_id,target_id,target_source FROM edges WHERE s1_id IN ({placeholders})",
        tuple(sample_ids),
    ):
        truth[source][s1].add(target)

    query_maps: dict[str, dict[str, set[str]]] = {}
    for channel in ("exact_normalized_name", "exact_normalized_address", "digit_token_overlap"):
        query_maps[channel] = defaultdict(set)
    for entity_id, name, address, _country in sample:
        if normalized := punctuation_normalized(name):
            query_maps["exact_normalized_name"][normalized].add(entity_id)
        if normalized := punctuation_normalized(address):
            query_maps["exact_normalized_address"][normalized].add(entity_id)
        for digits in set(digit_sequence(address)):
            query_maps["digit_token_overlap"][digits].add(entity_id)

    output: dict[str, Any] = {}
    candidates_by_source: dict[str, dict[str, dict[str, set[str]]]] = {}
    for source_number in (2, 3):
        source = f"S{source_number}"
        candidates_by_source[source] = {}
        channel_started = time.perf_counter()
        channel_candidates = {
            channel: {s1: set() for s1 in sample_ids} for channel in query_maps
        }
        for chunk in _read_chunks(paths[f"train_source{source_number}"]):
            for target_id, name, address in chunk[
                ["entity_id", "business_name", "business_address"]
            ].itertuples(index=False, name=None):
                keys = {
                    "exact_normalized_name": [punctuation_normalized(name)],
                    "exact_normalized_address": [punctuation_normalized(address)],
                    "digit_token_overlap": list(set(digit_sequence(address))),
                }
                for channel, values in keys.items():
                    for value in values:
                        if not value:
                            continue
                        for s1 in query_maps[channel].get(value, ()):
                            channel_candidates[channel][s1].add(target_id)
            monitor.sample()
        for channel, sets in channel_candidates.items():
            summary = _candidate_summary(sets, truth[source])
            summary.update(
                {
                    "runtime_seconds": time.perf_counter() - channel_started,
                    "peak_rss_mb": monitor.peak_mb,
                    "estimated_full_candidate_volume": int(
                        summary["average_candidates"]
                        * next(item["rows"] for item in _PROFILE_CACHE if item["key"] == "train_source1")
                    ),
                    "scope": "deterministic S1 sample queried against all target rows",
                }
            )
            output[f"{source}:{channel}"] = summary
            candidates_by_source[source][channel] = sets
        union = {
            s1: set().union(*(channel_candidates[channel][s1] for channel in channel_candidates))
            for s1 in sample_ids
        }
        union_summary = _candidate_summary(union, truth[source])
        union_summary.update(
            {
                "runtime_seconds": time.perf_counter() - channel_started,
                "peak_rss_mb": monitor.peak_mb,
                "estimated_full_candidate_volume": int(
                    union_summary["average_candidates"]
                    * next(item["rows"] for item in _PROFILE_CACHE if item["key"] == "train_source1")
                ),
                "scope": "deterministic S1 sample queried against all target rows",
            }
        )
        output[f"{source}:exact_digit_union"] = union_summary
        candidates_by_source[source]["exact_digit_union"] = union
    return output, candidates_by_source, truth


def _bounded_sparse_topk(
    vectorizer: TfidfVectorizer,
    queries: list[str],
    target_chunks: Iterable[tuple[list[str], list[str]]],
    *,
    threshold: float = 0.35,
    top_k: int = 20,
) -> list[set[str]]:
    query_matrix = vectorizer.transform(queries).astype(np.float32)
    best: list[list[tuple[float, str]]] = [[] for _ in queries]
    for target_ids, texts in target_chunks:
        target_matrix = vectorizer.transform(texts).astype(np.float32)
        similarities = query_matrix @ target_matrix.T
        if not sparse.issparse(similarities):
            raise RuntimeError("sparse retrieval unexpectedly produced a dense matrix")
        similarities = similarities.tocsr()
        for query_index in range(similarities.shape[0]):
            start, end = similarities.indptr[query_index : query_index + 2]
            scores = similarities.data[start:end]
            indices = similarities.indices[start:end]
            keep = scores >= threshold
            if keep.any():
                best[query_index].extend(
                    (float(score), target_ids[int(index)])
                    for score, index in zip(scores[keep], indices[keep], strict=True)
                )
                best[query_index] = sorted(best[query_index], reverse=True)[:top_k]
    return [{target_id for _score, target_id in matches} for matches in best]


def sparse_tfidf_experiment(
    paths: DataPaths,
    connection: sqlite3.Connection,
    monitor: ResourceMonitor,
    sample_limit: int = 250,
    fit_limit: int = 50_000,
) -> dict[str, Any]:
    """Bounded sparse retrieval against full targets; vocabulary fitting is sampled."""
    sample = _sample_s1(connection, sample_limit)
    sample_ids = [row[0] for row in sample]
    result: dict[str, Any] = {}
    truth: dict[str, dict[str, set[str]]] = {
        source: {s1: set() for s1 in sample_ids} for source in ("S2", "S3")
    }
    placeholders = ",".join("?" for _ in sample_ids)
    for s1, target, source in connection.execute(
        f"SELECT s1_id,target_id,target_source FROM edges WHERE s1_id IN ({placeholders})",
        tuple(sample_ids),
    ):
        truth[source][s1].add(target)

    channel_specs = {
        "name_char_tfidf": ("business_name", "char_wb", (3, 5)),
        "address_char_tfidf": ("business_address", "char_wb", (3, 5)),
        "name_word_tfidf": ("business_name", "word", (1, 2)),
        "address_word_tfidf": ("business_address", "word", (1, 2)),
    }
    for source_number in (2, 3):
        source = f"S{source_number}"
        path = paths[f"train_source{source_number}"]
        for channel, (source_field, analyzer, ngram_range) in channel_specs.items():
            started = time.perf_counter()
            fit_texts: list[str] = []
            for chunk in _read_chunks(path):
                selected = (_series_hash(chunk["entity_id"]) % 100) == 0
                fit_texts.extend(chunk.loc[selected, source_field].tolist())
                if len(fit_texts) >= fit_limit:
                    break
            fit_texts = fit_texts[:fit_limit]
            vectorizer = TfidfVectorizer(
                analyzer=analyzer,
                ngram_range=ngram_range,
                min_df=2,
                max_features=40_000,
                dtype=np.float32,
                strip_accents="unicode",
                lowercase=True,
            )
            vectorizer.fit(fit_texts)
            query_texts = [
                row[1] if source_field == "business_name" else row[2] for row in sample
            ]

            def chunks(
                target_path: Path = path, target_field: str = source_field
            ) -> Iterator[tuple[list[str], list[str]]]:
                for chunk in _read_chunks(target_path, chunksize=20_000):
                    yield chunk["entity_id"].tolist(), chunk[target_field].tolist()
                    monitor.sample()

            retrieved = _bounded_sparse_topk(vectorizer, query_texts, chunks())
            candidate_sets = dict(zip(sample_ids, retrieved, strict=True))
            summary = _candidate_summary(candidate_sets, truth[source])
            summary.update(
                {
                    "runtime_seconds": time.perf_counter() - started,
                    "peak_rss_mb": monitor.peak_mb,
                    "estimated_full_candidate_volume": int(
                        summary["average_candidates"]
                        * next(item["rows"] for item in _PROFILE_CACHE if item["key"] == "train_source1")
                    ),
                    "scope": (
                        "deterministic S1 sample against all targets; vocabulary fit on a "
                        f"deterministic sample of {len(fit_texts):,}; cosine>=0.35, top-20 cap"
                    ),
                }
            )
            result[f"{source}:{channel}"] = summary
            del vectorizer
    return result


def _write_quality_issues(path: Path, issues: list[dict[str, Any]]) -> None:
    pd.DataFrame(issues).to_csv(path, sep="\t", index=False)


def _fmt_pct(value: float) -> str:
    return f"{100 * value:.3f}%"


def write_profile_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Dataset profile",
        "",
        f"Generated with deterministic seed `{SEED}`. Row-level missingness, ASCII/digit rates, "
        "countries, and hash-equivalent duplicate counts are full scans. Lengths, token counts, "
        "scripts, accents, and token frequencies use the documented deterministic 2% sample. "
        "A 64-bit pandas hash was used for duplicate equivalence; collision risk is negligible "
        "but non-zero.",
        "",
        "## Source profiles",
        "",
        "| Split/source | Rows | File MiB | DataFrame MiB | Missing name | Missing address | Both missing | Countries |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in payload["sources"]:
        lines.append(
            f"| {item['key']} | {item['rows']:,} | {item['file_bytes']/2**20:.1f} | "
            f"{item['dataframe_memory_bytes']/2**20:.1f} | {_fmt_pct(item['missing_rates']['name'])} | "
            f"{_fmt_pct(item['missing_rates']['address'])} | {_fmt_pct(item['missing_rates']['both'])} | "
            f"{item['countries']} |"
        )
    lines.extend(
        [
            "",
            "## Country partitions",
            "",
            "| Split/source | Country | Rows | Missing name | Missing address | Non-ASCII name | Address digits | Sample mean name/address length |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, countries in payload.get("country_profiles", {}).items():
        for country, item in countries.items():
            lines.append(
                f"| {key} | {country} | {item['rows']:,} | {_fmt_pct(item['name_missing_rate'])} | "
                f"{_fmt_pct(item['address_missing_rate'])} | {_fmt_pct(item['name_non_ascii_rate'])} | "
                f"{_fmt_pct(item['address_has_digit_rate'])} | "
                f"{item['sample']['name_length']['mean']:.2f}/{item['sample']['address_length']['mean']:.2f} |"
            )
    lines.extend(["", "## Detailed measurements", ""])
    for item in payload["sources"]:
        sample = item["sample"]
        lines.extend(
            [
                f"### {item['key']}",
                "",
                f"Sampled rows: {sample['rows']:,}. Name length mean/median/p95: "
                f"{sample['name_length']['mean']:.2f}/{sample['name_length']['median']:.1f}/"
                f"{sample['name_length']['p95']:.1f}; address: {sample['address_length']['mean']:.2f}/"
                f"{sample['address_length']['median']:.1f}/{sample['address_length']['p95']:.1f}.",
                "",
                f"Name token mean: {sample['name_token_count']['mean']:.2f}; address token mean: "
                f"{sample['address_token_count']['mean']:.2f}. Non-ASCII name/address: "
                f"{_fmt_pct(item['full_rates']['name_non_ascii'])}/"
                f"{_fmt_pct(item['full_rates']['address_non_ascii'])}. Digit-bearing name/address: "
                f"{_fmt_pct(item['full_rates']['name_has_digit'])}/"
                f"{_fmt_pct(item['full_rates']['address_has_digit'])}.",
                "",
                f"Script counts in sample: `{sample['scripts']}`. Accent rates in sample: "
                f"name {_fmt_pct(sample['accent_rates']['name'])}, address "
                f"{_fmt_pct(sample['accent_rates']['address'])}.",
                "",
                f"Duplicate metrics: `{item['duplicates']}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Ground-truth structure",
            "",
            f"`{payload['truth']}`",
            "",
            "## Positive-pair analysis",
            "",
            f"`{payload['positive_pairs']}`",
            "",
            "## Negative-pair comparison samples",
            "",
            f"`{payload.get('negative_pair_samples', {})}`. Samples are deterministic verified "
            "nonmatches and contain identifiers plus comparison features, not raw business text.",
            "",
            "## Train/test shift",
            "",
            *[f"- {line}" for line in payload["shift_findings"]],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_blocking_markdown(path: Path, blocking: dict[str, Any], resources: dict[str, Any]) -> None:
    lines = [
        "# Blocking feasibility",
        "",
        "All target sources were retrieved separately. Exact/digit channels use a deterministic "
        "S1 sample against every target row. TF-IDF uses sparse matrices only, transforms targets "
        "in 20,000-row chunks, and retains cosine >= 0.35 with a top-20 cap. TF-IDF vocabulary "
        "fitting and recall are sampled feasibility measurements, not full-corpus guarantees.",
        "",
        "| Source/channel | Edge recall | All-matches coverage | Avg cand. | Median | p95 | Max | Runtime s | Peak MiB | Est. full volume |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for channel, item in sorted(blocking.items()):
        lines.append(
            f"| {channel} | {_fmt_pct(item['positive_edge_recall'])} | "
            f"{_fmt_pct(item['all_true_matches_per_s1_coverage'])} | "
            f"{item['average_candidates']:.2f} | {item['median_candidates']:.1f} | "
            f"{item['p95_candidates']:.1f} | {item['maximum_candidates']} | "
            f"{item['runtime_seconds']:.1f} | {item['peak_rss_mb']:.1f} | "
            f"{item['estimated_full_candidate_volume']:,} |"
        )
    lines.extend(
        [
            "",
            "## Resource guard",
            "",
            f"Available memory at start: {resources['available_memory_start_mb']:.1f} MiB; "
            f"observed process peak: {resources['peak_rss_mb']:.1f} MiB. No dense all-pairs "
            "matrix or Cartesian product was constructed. The full unsampled TF-IDF query set "
            "was not attempted: even a single dense 2,206,821 x 5,034,616 float32 matrix would "
            "require about 40.9 TiB. The safe alternative is the measured chunked sample followed "
            "by an indexed ANN implementation in the next milestone.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _shift_findings(profiles: list[dict[str, Any]]) -> list[str]:
    by_key = {item["key"]: item for item in profiles}
    findings: list[str] = []
    for source in (1, 2, 3):
        train = by_key[f"train_source{source}"]
        test = by_key[f"test_source{source}"]
        findings.append(
            f"S{source} rows: train {train['rows']:,}, test {test['rows']:,} "
            f"({test['rows']/train['rows']:.3f}x); missing-name shift "
            f"{_fmt_pct(test['missing_rates']['name']-train['missing_rates']['name'])} points and "
            f"missing-address shift {_fmt_pct(test['missing_rates']['address']-train['missing_rates']['address'])} points."
        )
        train_dup = train["duplicates"]["normalized_name"]["duplicate_excess"] / train["rows"]
        test_dup = test["duplicates"]["normalized_name"]["duplicate_excess"] / test["rows"]
        findings.append(
            f"S{source} sampled mean name/address length shift: "
            f"{test['sample']['name_length']['mean']-train['sample']['name_length']['mean']:+.2f}/"
            f"{test['sample']['address_length']['mean']-train['sample']['address_length']['mean']:+.2f} chars; "
            f"non-ASCII name/address shift "
            f"{_fmt_pct(test['full_rates']['name_non_ascii']-train['full_rates']['name_non_ascii'])}/"
            f"{_fmt_pct(test['full_rates']['address_non_ascii']-train['full_rates']['address_non_ascii'])}; "
            f"digit-bearing name/address shift "
            f"{_fmt_pct(test['full_rates']['name_has_digit']-train['full_rates']['name_has_digit'])}/"
            f"{_fmt_pct(test['full_rates']['address_has_digit']-train['full_rates']['address_has_digit'])}; "
            f"normalized-name duplicate-density shift {_fmt_pct(test_dup-train_dup)}."
        )
    train_countries = set().union(
        *(set(by_key[f"train_source{s}"]["countries"]) for s in (1, 2, 3))
    )
    test_countries = set().union(*(set(by_key[f"test_source{s}"]["countries"]) for s in (1, 2, 3)))
    findings.append(
        f"Country labels: train={sorted(train_countries)}, test={sorted(test_countries)}; "
        f"test-only={sorted(test_countries-train_countries)}. Unseen labels are measured shift, not integrity errors."
    )
    france_rows = sum(by_key[f"test_source{s}"]["countries"].get("France", 0) for s in (1, 2, 3))
    findings.append(f"France contributes {france_rows:,} test rows across sources (measured).")
    return findings


def detailed_shift(paths: DataPaths) -> dict[str, Any]:
    """Measure sampled token OOV and France-specific characteristics by source."""
    result: dict[str, Any] = {}
    for source in (1, 2, 3):
        train_vocab: set[str] = set()
        train_tokens: Counter[str] = Counter()
        for chunk in _read_chunks(paths[f"train_source{source}"]):
            selected = (_series_hash(chunk["entity_id"]) % 100) < 2
            for name, address in chunk.loc[
                selected, ["business_name", "business_address"]
            ].itertuples(index=False, name=None):
                tokens = (*word_tokens(name), *word_tokens(address))
                train_vocab.update(tokens)
                train_tokens.update(tokens)

        test_tokens: Counter[str] = Counter()
        oov_tokens = 0
        total_tokens = 0
        france = Counter()
        france_lengths: list[int] = []
        france_token_counts: list[int] = []
        france_scripts: Counter[str] = Counter()
        for chunk in _read_chunks(paths[f"test_source{source}"]):
            selected = (_series_hash(chunk["entity_id"]) % 100) < 2
            sample = chunk.loc[selected]
            for name, address, country in sample[
                ["business_name", "business_address", "country"]
            ].itertuples(index=False, name=None):
                tokens = (*word_tokens(name), *word_tokens(address))
                test_tokens.update(tokens)
                total_tokens += len(tokens)
                oov_tokens += sum(token not in train_vocab for token in tokens)
                if country == "France":
                    france["sample_rows"] += 1
                    france["name_missing"] += int(not name.strip())
                    france["address_missing"] += int(not address.strip())
                    france["name_non_ascii"] += int(any(ord(char) > 127 for char in name))
                    france["address_non_ascii"] += int(any(ord(char) > 127 for char in address))
                    france["name_has_digit"] += int(any(char.isdigit() for char in name))
                    france["address_has_digit"] += int(any(char.isdigit() for char in address))
                    france_lengths.extend((len(name), len(address)))
                    france_token_counts.extend((len(word_tokens(name)), len(word_tokens(address))))
                    france_scripts[f"name:{_script(name)}"] += 1
                    france_scripts[f"address:{_script(address)}"] += 1
        common_train = {token for token, _count in train_tokens.most_common(100)}
        common_test = {token for token, _count in test_tokens.most_common(100)}
        france_rows = france["sample_rows"]
        result[f"S{source}"] = {
            "sample_method": "deterministic 2% ID-hash sample in train and test",
            "train_vocabulary_size": len(train_vocab),
            "test_token_count": total_tokens,
            "test_oov_token_count": oov_tokens,
            "test_oov_token_rate": oov_tokens / max(total_tokens, 1),
            "top_100_token_overlap": len(common_train & common_test) / 100,
            "france_sample": {
                "rows": france_rows,
                "name_missing_rate": france["name_missing"] / max(france_rows, 1),
                "address_missing_rate": france["address_missing"] / max(france_rows, 1),
                "name_non_ascii_rate": france["name_non_ascii"] / max(france_rows, 1),
                "address_non_ascii_rate": france["address_non_ascii"] / max(france_rows, 1),
                "name_has_digit_rate": france["name_has_digit"] / max(france_rows, 1),
                "address_has_digit_rate": france["address_has_digit"] / max(france_rows, 1),
                "combined_text_length": _distribution(france_lengths),
                "combined_token_count": _distribution(france_token_counts),
                "scripts": dict(france_scripts),
            },
        }
    return result


def detailed_country_profiles(paths: DataPaths) -> dict[str, Any]:
    """Profile core text characteristics within each source/country partition."""
    output: dict[str, Any] = {}
    for key in SOURCE_KEYS:
        counters: dict[str, Counter[str]] = defaultdict(Counter)
        sampled: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: {"name_length": [], "address_length": [], "name_tokens": [], "address_tokens": []}
        )
        scripts: dict[str, Counter[str]] = defaultdict(Counter)
        for chunk in _read_chunks(paths[key]):
            for country, group in chunk.groupby("country", sort=False):
                country_key = str(country)
                count = len(group)
                name = group["business_name"]
                address = group["business_address"]
                counters[country_key]["rows"] += count
                counters[country_key]["name_missing"] += int(name.str.strip().eq("").sum())
                counters[country_key]["address_missing"] += int(address.str.strip().eq("").sum())
                counters[country_key]["name_non_ascii"] += int(
                    name.str.contains(r"[^\x00-\x7f]", regex=True).sum()
                )
                counters[country_key]["address_non_ascii"] += int(
                    address.str.contains(r"[^\x00-\x7f]", regex=True).sum()
                )
                counters[country_key]["name_has_digit"] += int(
                    name.str.contains(r"\d", regex=True).sum()
                )
                counters[country_key]["address_has_digit"] += int(
                    address.str.contains(r"\d", regex=True).sum()
                )
                counters[country_key]["unicode_replacement"] += int(
                    name.str.contains("\ufffd", regex=False).sum()
                    + address.str.contains("\ufffd", regex=False).sum()
                )
                counters[country_key]["control_character"] += int(
                    name.str.contains(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", regex=True).sum()
                    + address.str.contains(
                        r"[\x00-\x08\x0b\x0c\x0e-\x1f]", regex=True
                    ).sum()
                )
            selected = (_series_hash(chunk["entity_id"]) % 100) < 2
            for name, address, country in chunk.loc[
                selected, ["business_name", "business_address", "country"]
            ].itertuples(index=False, name=None):
                values = sampled[country]
                values["name_length"].append(len(name))
                values["address_length"].append(len(address))
                values["name_tokens"].append(len(word_tokens(name)))
                values["address_tokens"].append(len(word_tokens(address)))
                scripts[country][f"name:{_script(name)}"] += 1
                scripts[country][f"address:{_script(address)}"] += 1
        output[key] = {}
        for country, values in counters.items():
            rows = values["rows"]
            output[key][country] = {
                "rows": rows,
                "name_missing_rate": values["name_missing"] / rows,
                "address_missing_rate": values["address_missing"] / rows,
                "name_non_ascii_rate": values["name_non_ascii"] / rows,
                "address_non_ascii_rate": values["address_non_ascii"] / rows,
                "name_has_digit_rate": values["name_has_digit"] / rows,
                "address_has_digit_rate": values["address_has_digit"] / rows,
                "unicode_replacement_count": values["unicode_replacement"],
                "control_character_count": values["control_character"],
                "sample": {
                    field: _distribution(measurements)
                    for field, measurements in sampled[country].items()
                },
                "sample_scripts": dict(scripts[country]),
            }
    return output


def _figures(profiles: list[dict[str, Any]], truth: dict[str, Any], figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    labels = [item["key"] for item in profiles]
    missing_name = [100 * item["missing_rates"]["name"] for item in profiles]
    missing_address = [100 * item["missing_rates"]["address"] for item in profiles]
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.bar(x - 0.2, missing_name, 0.4, label="name")
    axis.bar(x + 0.2, missing_address, 0.4, label="address")
    axis.set_xticks(x, labels, rotation=30, ha="right")
    axis.set_ylabel("Missing (%)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "missingness_by_source.png", dpi=160)
    plt.close(fig)

    cardinality = truth["cardinality"]
    keys = ["0", "1", "2", "3", "4+"]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(keys, [cardinality.get(key, 0) for key in keys])
    axis.set_xlabel("True matches per S1")
    axis.set_ylabel("S1 entities")
    fig.tight_layout()
    fig.savefig(figures / "truth_cardinality.png", dpi=160)
    plt.close(fig)


def write_handoff(path: Path, payload: dict[str, Any]) -> None:
    profiles = {item["key"]: item for item in payload["sources"]}
    truth = payload["truth"]
    blocking = payload["blocking"]
    best = max(blocking.items(), key=lambda item: item[1]["positive_edge_recall"])
    total_rows = sum(item["rows"] for item in payload["sources"])
    text = f"""# Forensic-analysis handoff

## 1. Dataset scale

The six source files contain {total_rows:,} rows ({sum(item['file_bytes'] for item in payload['sources'])/2**30:.2f} GiB on disk): {profiles['train_source1']['rows']:,} training and {profiles['test_source1']['rows']:,} test S1 entities. The truth file contains {truth['rows']:,} S1 rows and {truth['positive_edges']:,} positive edges.

## 2. Data-quality findings

Full scans found {truth['unknown_source1_references']} unknown S1 truth references, {truth['unknown_target_references']} unknown targets, {truth['duplicate_ids_within_lists']} duplicate IDs within truth lists, and {truth['source1_ids_as_targets']} S1 IDs used as targets. Conflicting target ownership is {truth['conflicting_target_ownership']}. Country is treated as open-set; France is a measured test-only label, not an error. Field-level empty, whitespace-only, null-like, Unicode, and duplicate measurements are in `data_profile.json`.

## 3. Ground-truth structure

There are {truth['singletons']:,} singleton S1 entities ({_fmt_pct(truth['singleton_rate'])}). Cardinality counts are {truth['cardinality']}; source edge counts are {truth['per_source_edges']}; {truth['matched_in_both_sources']:,} S1 entities have matches in both S2 and S3. Country and missingness breakdowns are {truth['by_country']} and {truth['by_missingness']}.

## 4. Five main matching difficulties

1. Singleton decisions affect {truth['singletons']:,} S1 rows ({_fmt_pct(truth['singleton_rate'])}) and false positives score zero for those rows.
2. Multi-match output is material: {truth['cardinality'].get('4+', 0):,} S1 rows have 4+ targets.
3. Test-only France contributes {sum(profiles[f'test_source{s}']['countries'].get('France',0) for s in (1,2,3)):,} records, creating open-set country and language shift.
4. Missing names/addresses reach maxima of {_fmt_pct(max(item['missing_rates']['name'] for item in payload['sources']))} and {_fmt_pct(max(item['missing_rates']['address'] for item in payload['sources']))} across source files.
5. Exact channels alone do not recover every edge: the strongest measured channel is `{best[0]}` at {_fmt_pct(best[1]['positive_edge_recall'])} sampled recall (measurement scope documented in `blocking_feasibility.md`).

## 5. Five strongest opportunities

1. Exact normalized name and address produce high-precision, low-cost candidates with measured candidate distributions in the blocking report.
2. Address digits are preserved in every comparison view and digit conflicts are measured on all {payload['positive_pairs']['rows']:,} positive edges.
3. Character TF-IDF tolerates punctuation, spacing, and transcription variation; sampled full-target recall is measured separately for S2 and S3.
4. Source-specific retrieval is justified by {truth['per_source_edges'].get('S1-S2',0):,} S1-S2 versus {truth['per_source_edges'].get('S1-S3',0):,} S1-S3 edges.
5. The exact per-S1 singleton-aware objective supports calibrated source/channel-specific thresholds rather than forcing a match.

## 6. Train/test shift

""" + "\n".join(f"- {finding}" for finding in payload["shift_findings"]) + f"""

These are measurements. A hypothesis for the next milestone is that France requires country-aware calibration; test labels do not exist, so no France accuracy claim is made.

## 7. Blocking results

The best sampled recall was `{best[0]}` at {_fmt_pct(best[1]['positive_edge_recall'])}, with {best[1]['average_candidates']:.2f} average candidates and {_fmt_pct(best[1]['all_true_matches_per_s1_coverage'])} all-match coverage. Exact/digit channels were evaluated against every target; TF-IDF used sparse full-target scans with sampled queries, a 40,000-feature vocabulary, cosine >=0.35, and top-20 cap. Values are measured samples and full-volume values are extrapolations.

## 8. Measured compute recommendation

This run peaked at {payload['resources']['peak_rss_mb']:.1f} MiB process RSS on an 8 GiB host with {payload['resources']['available_memory_start_mb']:.1f} MiB available at start. Use a 16-vCPU, 64-GiB RAM, 100-GiB SSD tier for the next milestone so a disk-backed/ANN candidate index, full feature tables, and cross-validation folds can coexist safely. This is a recommendation derived from measured peak plus the avoided 40.9-TiB dense-matrix requirement.

## 9. Ranked modelling recommendation

1. Gradient-boosted pair classifier on normalized similarities, token/digit evidence, missingness, country and source indicators; exclude numeric entity-ID components.
2. Calibrated source-specific decision thresholds optimized with exact macro per-S1 F0.5 and singleton handling.
3. Character/word sparse retrieval union plus exact/digit blocks, followed by hard-negative mining.
4. Country-aware calibration with France treated as unseen/open-set, never inferred from external data.
5. A compact text encoder only if sparse candidates plateau; any model must be MIT/Apache-2.0 and <=8B parameters.

## 10. Next three experiments

1. Implement leakage-safe grouped validation and the exact per-S1 macro F0.5 scorer, then baseline deterministic rules.
2. Build a persistent sparse/ANN index and measure full-S1 recall/candidate volume for the union without the query sample.
3. Train a first pair ranker with deterministic random and hard negatives, then tune singleton-aware thresholds by source and country.

## 11. Critical unknowns

- Test truth is unavailable; France generalization is a hypothesis only.
- Sampled TF-IDF recall may differ from full-query recall; sample sizes and caps are explicit in the blocking report.
- Hash-equivalent duplicate counts carry a negligible but non-zero 64-bit collision risk.
- The precision impact of candidate unions is unknown until leakage-safe validation is implemented.

## Reproduction

From the repository root: `.venv/bin/python -m entity_resolution.forensic_analysis --output-dir reports --exact-sample 500 --tfidf-sample 100`.
"""
    path.write_text(text, encoding="utf-8")


_PROFILE_CACHE: list[dict[str, Any]] = []


def run_analysis(output_dir: Path, *, sample_limit: int = 2_000, tfidf_sample: int = 250) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / ".forensic_work"
    work_dir.mkdir(exist_ok=True)
    figures = output_dir / "figures"
    paths = load_data_paths()
    available_start = psutil.virtual_memory().available / (1024 * 1024)
    monitor = ResourceMonitor()
    validate_all(paths)
    print("stage=structural_validation status=pass", flush=True)
    issues: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    id_hash_paths: dict[str, Path] = {}
    for key in SOURCE_KEYS:
        print(f"stage=profile dataset={key} status=started", flush=True)
        profile_checkpoint = work_dir / f"{key}_profile.json"
        id_hash_path = work_dir / f"{key}_id_hashes.npy"
        if profile_checkpoint.is_file() and id_hash_path.is_file():
            profile = json.loads(profile_checkpoint.read_text(encoding="utf-8"))
        else:
            profile, id_hash_path = profile_source(key, paths[key], work_dir, monitor)
            profile_checkpoint.write_text(
                json.dumps(profile, default=_json_default), encoding="utf-8"
            )
        profiles.append(profile)
        id_hash_paths[key] = id_hash_path
        print(
            f"stage=profile dataset={key} status=complete rows={profile['rows']} "
            f"peak_rss_mb={monitor.peak_mb:.1f}",
            flush=True,
        )
    global _PROFILE_CACHE
    _PROFILE_CACHE = profiles
    truth = inspect_truth(
        paths["train_ground_truth"], id_hash_paths, issues, work_dir, monitor
    )
    print(
        f"stage=truth status=complete edges={truth['positive_edges']} "
        f"peak_rss_mb={monitor.peak_mb:.1f}",
        flush=True,
    )
    for item in profiles:
        expected_prefix = f"S{item['key'][-1]}-"
        _issue(
            issues,
            severity="pass",
            dataset=item["key"],
            category="required_columns_unique_ids_source_prefix",
            count=0,
            detail=f"Full structural validation passed; IDs unique with prefix {expected_prefix}.",
        )
        for source_field in ("name", "address"):
            count = item["empty_whitespace_null_like"].get(
                f"{source_field}_null_like", 0
            )
            _issue(
                issues,
                severity="warning" if count else "pass",
                dataset=item["key"],
                category=f"{source_field}_null_like_literal",
                count=count,
                detail="Literal null-like tokens; organizer values were not rewritten.",
            )

    database = work_dir / "positive_join.sqlite"
    print("stage=join_database status=started", flush=True)
    connection = build_join_database(paths, database, monitor)
    print(
        f"stage=join_database status=complete peak_rss_mb={monitor.peak_mb:.1f}",
        flush=True,
    )
    truth.update(truth_structure_from_database(connection))
    _issue(
        issues,
        severity="error" if truth["conflicting_target_ownership"] else "pass",
        dataset="train_ground_truth",
        category="target_linked_to_multiple_source1_entities",
        count=truth["conflicting_target_ownership"],
        detail="Exact SQL count of targets with more than one distinct S1 owner.",
    )
    positive = write_positive_analysis(
        connection, output_dir / "positive_pair_analysis.parquet", monitor
    )
    positive["feature_summary_by_source"] = summarize_pair_parquet(
        output_dir / "positive_pair_analysis.parquet", "target_source"
    )
    positive["feature_summary_by_country"] = summarize_pair_parquet(
        output_dir / "positive_pair_analysis.parquet", "country"
    )
    print(
        f"stage=positive_pairs status=complete rows={positive['rows']} "
        f"peak_rss_mb={monitor.peak_mb:.1f}",
        flush=True,
    )
    exact, _candidate_sets, _true_sets = exact_blocking_experiment(
        paths, connection, sample_limit, monitor
    )
    print("stage=exact_blocking status=complete", flush=True)
    tfidf = sparse_tfidf_experiment(
        paths, connection, monitor, sample_limit=tfidf_sample
    )
    print("stage=tfidf_blocking status=complete", flush=True)
    from entity_resolution.negative_samples import create_negative_samples

    negative_counts = create_negative_samples(
        output_dir / "negative_pair_samples.parquet", per_type=500
    )
    negative_summary = summarize_pair_parquet(
        output_dir / "negative_pair_samples.parquet", "sample_type"
    )
    print(f"stage=negative_samples status=complete counts={negative_counts}", flush=True)
    blocking = exact | tfidf
    shift = _shift_findings(profiles)
    shift_details = detailed_shift(paths)
    country_profiles = detailed_country_profiles(paths)
    for source, details in shift_details.items():
        france = details["france_sample"]
        shift.append(
            f"{source} sampled test-token OOV rate is {_fmt_pct(details['test_oov_token_rate'])}; "
            f"top-100 train/test token overlap is {_fmt_pct(details['top_100_token_overlap'])}. "
            f"France sample n={france['rows']:,}, address-missing "
            f"{_fmt_pct(france['address_missing_rate'])}, name/address non-ASCII "
            f"{_fmt_pct(france['name_non_ascii_rate'])}/{_fmt_pct(france['address_non_ascii_rate'])}, "
            f"and address-digit rate {_fmt_pct(france['address_has_digit_rate'])}."
        )
    for key, countries in country_profiles.items():
        replacement_count = sum(
            item["unicode_replacement_count"] for item in countries.values()
        )
        control_count = sum(item["control_character_count"] for item in countries.values())
        _issue(
            issues,
            severity="warning" if replacement_count or control_count else "pass",
            dataset=key,
            category="encoding_and_unicode_controls",
            count=replacement_count + control_count,
            detail=(
                f"UTF-8 parse passed; replacement characters={replacement_count}, "
                f"unexpected C0 controls={control_count}."
            ),
        )
        _issue(
            issues,
            severity="pass",
            dataset=key,
            category="country_labels_open_set",
            count=0,
            detail=f"Observed labels={sorted(countries)}; unseen labels are not intrinsically invalid.",
        )
    resources = {
        "available_memory_start_mb": available_start,
        "peak_rss_mb": monitor.peak_mb,
        "runtime_seconds": monitor.runtime_seconds,
        "logical_memory_mb": psutil.virtual_memory().total / (1024 * 1024),
    }
    payload: dict[str, Any] = {
        "seed": SEED,
        "sources": profiles,
        "truth": truth,
        "positive_pairs": positive,
        "negative_pair_samples": {"counts": negative_counts, "feature_summary": negative_summary},
        "blocking": blocking,
        "shift_findings": shift,
        "train_test_shift": shift_details,
        "country_profiles": country_profiles,
        "resources": resources,
    }
    _write_quality_issues(output_dir / "data_quality_issues.tsv", issues)
    (output_dir / "data_profile.json").write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )
    write_profile_markdown(output_dir / "data_profile.md", payload)
    write_blocking_markdown(output_dir / "blocking_feasibility.md", blocking, resources)
    write_handoff(output_dir / "handoff_summary.md", payload)
    _figures(profiles, truth, figures)
    connection.close()
    database.unlink(missing_ok=True)
    for hash_path in id_hash_paths.values():
        hash_path.unlink(missing_ok=True)
        hash_path.with_name(hash_path.name.replace("_id_hashes.npy", "_profile.json")).unlink(
            missing_ok=True
        )
    work_dir.rmdir()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--exact-sample", type=int, default=2_000)
    parser.add_argument("--tfidf-sample", type=int, default=250)
    args = parser.parse_args()
    payload = run_analysis(
        args.output_dir, sample_limit=args.exact_sample, tfidf_sample=args.tfidf_sample
    )
    print(json.dumps(payload["resources"], indent=2))


if __name__ == "__main__":
    main()
