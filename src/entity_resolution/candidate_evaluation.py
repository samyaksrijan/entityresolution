"""Candidate recall, coverage, oracle F0.5, and country-agreement measurements."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def per_s1_fbeta(
    predicted: set[str], truth: set[str], *, beta: float = 0.5
) -> float:
    """Exact competition-style per-S1 F-beta, including singleton semantics."""
    if not truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    true_positive = len(predicted & truth)
    if not true_positive:
        return 0.0
    precision = true_positive / len(predicted)
    recall = true_positive / len(truth)
    beta_squared = beta * beta
    return (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)


def oracle_macro_f05(
    candidates: dict[str, set[str]], truth: dict[str, set[str]]
) -> float:
    """Maximum macro F0.5 when only true edges present in candidates are predicted."""
    scores = []
    for s1_id, true_ids in truth.items():
        oracle_prediction = candidates.get(s1_id, set()) & true_ids
        scores.append(per_s1_fbeta(oracle_prediction, true_ids, beta=0.5))
    return float(np.mean(scores)) if scores else 0.0


def _base_metrics(
    candidates: dict[str, set[str]], truth: dict[str, set[str]]
) -> dict[str, float | int]:
    counts = np.asarray([len(candidates.get(s1_id, set())) for s1_id in truth], dtype=np.int64)
    positive_edges = sum(len(values) for values in truth.values())
    recalled = sum(len(candidates.get(s1_id, set()) & values) for s1_id, values in truth.items())
    covered = sum(values <= candidates.get(s1_id, set()) for s1_id, values in truth.items())
    return {
        "s1_entities": len(truth),
        "positive_edges": positive_edges,
        "recalled_edges": recalled,
        "positive_edge_recall": recalled / max(positive_edges, 1),
        "all_truth_recovered_rate": covered / max(len(truth), 1),
        "at_least_one_missed_truth_rate": 1 - covered / max(len(truth), 1),
        "average_candidates": float(counts.mean()) if len(counts) else 0.0,
        "median_candidates": float(np.median(counts)) if len(counts) else 0.0,
        "p90_candidates": float(np.quantile(counts, 0.90)) if len(counts) else 0.0,
        "p95_candidates": float(np.quantile(counts, 0.95)) if len(counts) else 0.0,
        "p99_candidates": float(np.quantile(counts, 0.99)) if len(counts) else 0.0,
        "maximum_candidates": int(counts.max()) if len(counts) else 0,
        "zero_candidate_rate": float((counts == 0).mean()) if len(counts) else 0.0,
        "oracle_macro_f0_5": oracle_macro_f05(candidates, truth),
    }


def evaluate_candidates(
    candidate_frame: pd.DataFrame,
    truth: dict[str, set[str]],
    metadata: pd.DataFrame,
    *,
    runtime_seconds: float = 0.0,
    peak_rss_mb: float = 0.0,
) -> dict[str, Any]:
    candidates = {
        s1_id: set(group["target_id"].tolist())
        for s1_id, group in candidate_frame.groupby("s1_id", sort=False)
    }
    overall = _base_metrics(candidates, truth)
    overall["runtime_seconds"] = runtime_seconds
    overall["peak_rss_mb"] = peak_rss_mb
    groupings: dict[str, dict[str, Any]] = {}
    requested_groupings = (
        "country",
        "cardinality_bucket",
        "truth_composition",
        "address_state",
        "target_address_state",
        "exact_name_state",
        "exact_address_state",
    )
    for grouping in (column for column in requested_groupings if column in metadata.columns):
        groups: dict[str, Any] = {}
        for value, subset in metadata.groupby(grouping, sort=True):
            ids = subset["s1_id"].tolist()
            subset_truth = {s1_id: truth[s1_id] for s1_id in ids}
            groups[str(value)] = _base_metrics(candidates, subset_truth)
        groupings[grouping] = groups
    overall["groups"] = groupings
    return overall


def filter_channel_topk(frame: pd.DataFrame, channel: str, top_k: int) -> pd.DataFrame:
    selected = frame[(frame["channel"] == channel) & (frame["channel_rank"] <= top_k)]
    return selected.drop_duplicates(["s1_id", "target_id", "target_source"])


def union_topk(
    frames: list[pd.DataFrame], channels: set[str], top_k: int
) -> pd.DataFrame:
    selected = []
    for frame in frames:
        part = frame[frame["channel"].isin(channels)]
        part = part[part["channel_rank"] <= top_k]
        selected.append(part)
    if not selected:
        return pd.DataFrame(columns=["s1_id", "target_id", "target_source"])
    return pd.concat(selected, ignore_index=True).drop_duplicates(
        ["s1_id", "target_id", "target_source"]
    )


def measure_country_edge_agreement(
    positive_parquet: Path, output_tsv: Path
) -> pd.DataFrame:
    """Measure country agreement across every materialized positive edge."""
    parquet = pq.ParquetFile(positive_parquet)
    totals: Counter[str] = Counter()
    agreements: Counter[str] = Counter()
    for batch in parquet.iter_batches(
        columns=["target_source", "country_agreement"], batch_size=250_000
    ):
        frame = batch.to_pandas()
        for source, group in frame.groupby("target_source", sort=True):
            totals[str(source)] += len(group)
            agreements[str(source)] += int(group["country_agreement"].sum())
    rows = []
    for source in ("S2", "S3"):
        total = totals[source]
        agree = agreements[source]
        rows.append(
            {
                "scope": source,
                "positive_edges": total,
                "country_agreements": agree,
                "country_disagreements": total - agree,
                "agreement_rate": agree / total,
                "disagreement_rate": (total - agree) / total,
            }
        )
    total = sum(totals.values())
    agree = sum(agreements.values())
    rows.append(
        {
            "scope": "overall",
            "positive_edges": total,
            "country_agreements": agree,
            "country_disagreements": total - agree,
            "agreement_rate": agree / total,
            "disagreement_rate": (total - agree) / total,
        }
    )
    result = pd.DataFrame(rows)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_tsv, sep="\t", index=False)
    return result


def estimate_output_bytes(
    measured_rows: int, measured_bytes: int, sample_s1: int, full_s1: int
) -> int:
    if measured_rows <= 0 or sample_s1 <= 0:
        return 0
    bytes_per_s1 = measured_bytes / sample_s1
    estimate = bytes_per_s1 * full_s1
    return int(math.ceil(estimate))
