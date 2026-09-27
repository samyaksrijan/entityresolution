# ruff: noqa: E501
"""Create deterministic, non-Cartesian negative-pair comparison samples."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz.fuzz import ratio

from entity_resolution.config import load_data_paths
from entity_resolution.forensic_analysis import (
    SEED,
    _pair_features,
    _read_chunks,
    _stable_u64,
)
from entity_resolution.views import alphanumeric_compact, digit_sequence, punctuation_normalized

Record = tuple[str, str, str, str]
Pair = tuple[Record, Record]


def _sample_queries(path: Path, limit: int) -> list[Record]:
    selected: list[Record] = []
    for chunk in _read_chunks(path):
        for row in chunk.itertuples(index=False, name=None):
            if _stable_u64(row[0]) % 10_000 < 10:
                selected.append(row)
    selected.sort(key=lambda row: _stable_u64(row[0]))
    return selected[:limit]


def _truth_for_queries(path: Path, query_ids: set[str]) -> dict[str, set[str]]:
    truth = {query_id: set() for query_id in query_ids}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["source1_entity_id"] in truth:
                raw = row["matched_entity_ids"]
                truth[row["source1_entity_id"]] = set() if raw == "" else set(raw.split(","))
    return truth


def _add(
    buckets: dict[str, list[Pair]],
    seen: dict[str, set[tuple[str, str]]],
    category: str,
    query: Record,
    target: Record,
    truth: dict[str, set[str]],
    limit: int,
) -> None:
    key = (query[0], target[0])
    if target[0] in truth[query[0]] or key in seen[category] or len(buckets[category]) >= limit:
        return
    seen[category].add(key)
    buckets[category].append((query, target))


def create_negative_samples(output: Path, *, per_type: int = 500) -> dict[str, int]:
    paths = load_data_paths()
    queries = _sample_queries(paths["train_source1"], 1_000)
    query_by_id = {row[0]: row for row in queries}
    truth = _truth_for_queries(paths["train_ground_truth"], set(query_by_id))
    exact_name: dict[str, list[str]] = defaultdict(list)
    name_prefix: dict[str, list[str]] = defaultdict(list)
    address_prefix: dict[str, list[str]] = defaultdict(list)
    for entity_id, name, address, _country in queries:
        normalized_name = punctuation_normalized(name)
        if normalized_name:
            exact_name[normalized_name].append(entity_id)
        compact_name = alphanumeric_compact(name)
        if len(compact_name) >= 6:
            name_prefix[compact_name[:6]].append(entity_id)
        compact_address = alphanumeric_compact(address)
        if len(compact_address) >= 8:
            address_prefix[compact_address[:8]].append(entity_id)

    categories = (
        "random_negative",
        "similar_name_negative",
        "similar_address_negative",
        "same_normalized_name_negative",
        "digit_conflict_negative",
    )
    buckets: dict[str, list[Pair]] = {category: [] for category in categories}
    seen: dict[str, set[tuple[str, str]]] = {category: set() for category in categories}
    random_targets: list[Record] = []
    for source_number in (2, 3):
        for chunk in _read_chunks(paths[f"train_source{source_number}"]):
            for target in chunk.itertuples(index=False, name=None):
                target_id, target_name, target_address, _country = target
                if _stable_u64(target_id) % 10_000 < 5 and len(random_targets) < 10_000:
                    random_targets.append(target)
                normalized_name = punctuation_normalized(target_name)
                for query_id in exact_name.get(normalized_name, ()):
                    _add(
                        buckets,
                        seen,
                        "same_normalized_name_negative",
                        query_by_id[query_id],
                        target,
                        truth,
                        per_type,
                    )
                compact_name = alphanumeric_compact(target_name)
                if len(compact_name) >= 6:
                    for query_id in name_prefix.get(compact_name[:6], ())[:8]:
                        query = query_by_id[query_id]
                        score = ratio(
                            punctuation_normalized(query[1]), normalized_name
                        ) / 100.0
                        if 0.65 <= score < 1.0:
                            _add(
                                buckets,
                                seen,
                                "similar_name_negative",
                                query,
                                target,
                                truth,
                                per_type,
                            )
                        query_digits = set(digit_sequence(query[2]))
                        target_digits = set(digit_sequence(target_address))
                        if score >= 0.60 and query_digits and target_digits and query_digits.isdisjoint(target_digits):
                            _add(
                                buckets,
                                seen,
                                "digit_conflict_negative",
                                query,
                                target,
                                truth,
                                per_type,
                            )
                compact_address = alphanumeric_compact(target_address)
                if len(compact_address) >= 8:
                    for query_id in address_prefix.get(compact_address[:8], ())[:8]:
                        query = query_by_id[query_id]
                        score = ratio(
                            punctuation_normalized(query[2]),
                            punctuation_normalized(target_address),
                        ) / 100.0
                        if 0.65 <= score < 1.0:
                            _add(
                                buckets,
                                seen,
                                "similar_address_negative",
                                query,
                                target,
                                truth,
                                per_type,
                            )

    generator = random.Random(SEED)
    query_order = list(queries)
    target_order = list(random_targets)
    generator.shuffle(query_order)
    generator.shuffle(target_order)
    for index in range(max(len(query_order), len(target_order)) * 3):
        query = query_order[index % len(query_order)]
        target = target_order[(index * 7919) % len(target_order)]
        _add(
            buckets,
            seen,
            "random_negative",
            query,
            target,
            truth,
            per_type,
        )
        if len(buckets["random_negative"]) >= per_type:
            break

    rows: list[dict[str, Any]] = []
    for category, pairs in buckets.items():
        for query, target in pairs:
            feature = _pair_features(
                (
                    query[0],
                    target[0],
                    target[0][:2],
                    query[1],
                    query[2],
                    query[3],
                    target[1],
                    target[2],
                    target[3],
                )
            )
            feature["sample_type"] = category
            rows.append(feature)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output, compression="zstd")
    return {category: len(pairs) for category, pairs in buckets.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/negative_pair_samples.parquet"))
    parser.add_argument("--per-type", type=int, default=500)
    args = parser.parse_args()
    print(create_negative_samples(args.output, per_type=args.per_type))


if __name__ == "__main__":
    main()
