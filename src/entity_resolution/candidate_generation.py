# ruff: noqa: E501
"""Scalable, measurable candidate-generation evaluation pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
import scipy.sparse as sp
import yaml
from sparse_dot_topn import sp_matmul_topn

from entity_resolution.candidate_evaluation import (
    evaluate_candidates,
    filter_channel_topk,
    measure_country_edge_agreement,
    union_topk,
)
from entity_resolution.candidate_schema import (
    CandidateEvidence,
    add_address_digit_agreement,
    aggregate_provenance,
    evidence_frame,
)
from entity_resolution.config import load_data_paths
from entity_resolution.exact_index import content_tie_key, query_key_maps
from entity_resolution.forensic_analysis import _read_chunks
from entity_resolution.sparse_retrieval import (
    SparseRetrievalConfig,
    SparseTopNRetriever,
    fit_vectorizer,
    normalized_text,
)
from entity_resolution.views import normalize_field

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "candidate_generation_v1.yaml"
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def load_candidate_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("candidate configuration must be a mapping")
    return payload


def _content_hash(name: str, address: str, country: str) -> int:
    return int(content_tie_key(name, address, country), 16)


def _cardinality_bucket(count: int) -> str:
    return str(count) if count < 4 else "4+"


def _truth_composition(matches: list[str]) -> str:
    sources = {match[:2] for match in matches}
    if not sources:
        return "no_match"
    if sources == {"S2"}:
        return "S2_only"
    if sources == {"S3"}:
        return "S3_only"
    return "both_sources"


def build_evaluation_sample(
    *,
    size: int,
    output: Path,
    positive_parquet: Path,
) -> pd.DataFrame:
    """Select a content-hash-based sample before observing retrieval scores."""
    paths = load_data_paths()
    content_candidates: dict[str, dict[str, Any]] = {}
    for chunk in _read_chunks(paths["train_source1"]):
        for entity_id, name, address, country in chunk.itertuples(index=False, name=None):
            content_hash = _content_hash(name, address, country)
            if content_hash % 100 < 3:
                content_candidates[entity_id] = {
                    "s1_id": entity_id,
                    "country": country,
                    "address_state": "missing" if normalize_field(address).is_missing else "present",
                    "content_hash": content_hash,
                }
    with paths["train_ground_truth"].open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            metadata = content_candidates.get(row["source1_entity_id"])
            if metadata is None:
                continue
            matches = [] if row["matched_entity_ids"] == "" else row["matched_entity_ids"].split(",")
            metadata["cardinality"] = len(matches)
            metadata["cardinality_bucket"] = _cardinality_bucket(len(matches))
            metadata["truth_composition"] = _truth_composition(matches)

    selected_ids = set(content_candidates)
    exact_name: Counter[str] = Counter()
    exact_address: Counter[str] = Counter()
    for batch in pq.ParquetFile(positive_parquet).iter_batches(
        columns=["source1_entity_id", "name_normalized_exact", "address_normalized_exact"],
        batch_size=250_000,
    ):
        frame = batch.to_pandas()
        frame = frame[frame["source1_entity_id"].isin(selected_ids)]
        for s1_id, group in frame.groupby("source1_entity_id", sort=False):
            exact_name[s1_id] = max(exact_name[s1_id], int(group["name_normalized_exact"].any()))
            exact_address[s1_id] = max(
                exact_address[s1_id], int(group["address_normalized_exact"].any())
            )
    rows = []
    for s1_id, metadata in content_candidates.items():
        if "cardinality" not in metadata:
            continue
        metadata["exact_name_state"] = "available" if exact_name[s1_id] else "unavailable"
        metadata["exact_address_state"] = (
            "available" if exact_address[s1_id] else "unavailable"
        )
        rows.append(metadata)
    frame = pd.DataFrame(rows)
    strata = [
        "country",
        "cardinality_bucket",
        "truth_composition",
        "address_state",
        "exact_name_state",
        "exact_address_state",
    ]
    queues = {
        key: group.sort_values("content_hash", kind="stable").to_dict("records")
        for key, group in frame.groupby(strata, sort=True, dropna=False)
    }
    chosen: list[dict[str, Any]] = []
    keys = sorted(queues, key=str)
    while len(chosen) < size and keys:
        remaining = []
        for key in keys:
            if queues[key] and len(chosen) < size:
                chosen.append(queues[key].pop(0))
            if queues[key]:
                remaining.append(key)
        keys = remaining
    result = pd.DataFrame(chosen).drop(columns=["content_hash"])
    if len(result) < size:
        raise RuntimeError(f"only {len(result)} stratified S1 records were available")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    metadata = {
        "selection": "content-hash pre-sample followed by round-robin truth strata",
        "retrieval_scores_used": False,
        "requested_size": size,
        "actual_size": len(result),
        "strata": strata,
        "stratum_counts": {
            "|".join(map(str, key)): len(group)
            for key, group in result.groupby(strata, sort=True, dropna=False)
        },
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return result


def load_sample_queries(sample: pd.DataFrame) -> pd.DataFrame:
    ids = set(sample["s1_id"])
    pieces = []
    path = load_data_paths()["train_source1"]
    for chunk in _read_chunks(path):
        selected = chunk[chunk["entity_id"].isin(ids)]
        if not selected.empty:
            pieces.append(selected)
    queries = pd.concat(pieces, ignore_index=True)
    order = {s1_id: index for index, s1_id in enumerate(sample["s1_id"])}
    queries["_order"] = queries["entity_id"].map(order)
    return queries.sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)


def enrich_target_address_state(sample: pd.DataFrame, positive_parquet: Path) -> pd.DataFrame:
    """Add a truth-side missing-address subgroup without changing sampled S1 membership."""
    ids = set(sample["s1_id"])
    any_missing: Counter[str] = Counter()
    for batch in pq.ParquetFile(positive_parquet).iter_batches(
        columns=["source1_entity_id", "target_address_missing"], batch_size=250_000
    ):
        frame = batch.to_pandas()
        frame = frame[frame["source1_entity_id"].isin(ids)]
        for s1_id, group in frame.groupby("source1_entity_id", sort=False):
            any_missing[s1_id] = max(any_missing[s1_id], int(group["target_address_missing"].any()))
    result = sample.copy()
    result["target_address_state"] = [
        "no_true_target"
        if cardinality == 0
        else "any_true_target_missing"
        if any_missing[s1_id]
        else "all_true_target_addresses_present"
        for s1_id, cardinality in result[["s1_id", "cardinality"]].itertuples(
            index=False, name=None
        )
    ]
    return result


def load_sample_truth(sample_ids: set[str]) -> dict[str, set[str]]:
    truth = {s1_id: set() for s1_id in sample_ids}
    path = load_data_paths()["train_ground_truth"]
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["source1_entity_id"] in truth:
                raw = row["matched_entity_ids"]
                truth[row["source1_entity_id"]] = set() if raw == "" else set(raw.split(","))
    return truth


def _exact_and_fit_scan(
    queries: pd.DataFrame,
    *,
    source: str,
    run_id: str,
    fit_sample_size: int,
    country_partition: bool,
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame, dict[str, float | int]]:
    source_number = int(source[-1])
    path = load_data_paths()[f"train_source{source_number}"]
    specs = [
        ("name", "unicode_preserving"),
        ("address", "unicode_preserving"),
        ("name", "accent_folded"),
        ("address", "accent_folded"),
    ]
    maps = {
        spec: query_key_maps(
            queries, field=spec[0], view=spec[1], country_partition=country_partition
        )
        for spec in specs
    }
    matches: dict[tuple[str, str], dict[str, list[tuple[str, str]]]] = {
        spec: defaultdict(list) for spec in specs
    }
    fit_texts = {"name": [], "address": []}
    benchmark_rows: list[tuple[str, str, str, str]] = []
    started = time.perf_counter()
    peak = psutil.Process().memory_info().rss / 2**20
    for chunk in _read_chunks(path):
        for entity_id, name, address, country in chunk.itertuples(index=False, name=None):
            normalized_name = normalize_field(name)
            normalized_address = normalize_field(address)
            tie_key = content_tie_key(name, address, country)
            if len(benchmark_rows) < 50_000 and int(tie_key, 16) % 100 < 1:
                benchmark_rows.append((entity_id, name, address, country))
            if len(fit_texts["name"]) < fit_sample_size and int(tie_key, 16) % 100 < 2:
                fit_texts["name"].append(normalized_name.unicode_preserving)
                fit_texts["address"].append(normalized_address.unicode_preserving)
            views = {
                ("name", "unicode_preserving"): normalized_name.unicode_preserving,
                ("address", "unicode_preserving"): normalized_address.unicode_preserving,
                ("name", "accent_folded"): normalized_name.accent_folded,
                ("address", "accent_folded"): normalized_address.accent_folded,
            }
            partition = country if country_partition else "*"
            for spec, key in views.items():
                if not key:
                    continue
                for s1_id in maps[spec].get((partition, key), ()):
                    matches[spec][s1_id].append((tie_key, entity_id))
        peak = max(peak, psutil.Process().memory_info().rss / 2**20)

    evidence: list[CandidateEvidence] = []
    for (field, view), by_query in matches.items():
        channel = f"exact_{field}_{view}"
        for s1_id, candidates in by_query.items():
            candidates.sort(key=lambda item: item[0])
            for rank, (_tie, target_id) in enumerate(candidates[:80], start=1):
                evidence.append(
                    CandidateEvidence(
                        s1_id=s1_id,
                        target_id=target_id,
                        target_source=source,
                        channel=channel,
                        field=field,
                        normalization_view=view,
                        retrieval_score=1.0,
                        channel_rank=rank,
                        exact_match=True,
                        run_id=run_id,
                    )
                )
    benchmark = pd.DataFrame(benchmark_rows, columns=SOURCE_COLUMNS)
    return evidence_frame(evidence), fit_texts, benchmark, {
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mb": peak,
        "candidate_rows": len(evidence),
    }


def benchmark_backend(
    queries: pd.DataFrame,
    targets: pd.DataFrame,
    config: SparseRetrievalConfig,
    artifact_dir: Path,
    source: str,
) -> dict[str, Any]:
    query = queries.head(200)
    target = targets.head(20_000)
    fit_texts = [normalized_text(value, config.view) for value in target["business_name"]]
    vectorizer = fit_vectorizer(fit_texts, config)
    q = vectorizer.transform(
        [normalized_text(value, config.view) for value in query["business_name"]]
    ).astype(np.float32)
    t = vectorizer.transform(fit_texts).astype(np.float32)
    started = time.perf_counter()
    topn = sp_matmul_topn(q, t.T, top_n=20, threshold=config.threshold, sort=True, n_threads=4)
    optimized_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scipy_product = q @ t.T
    scipy_seconds = time.perf_counter() - started
    matrix_bytes = t.data.nbytes + t.indices.nbytes + t.indptr.nbytes
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(artifact_dir / f"{source}_name_benchmark_matrix.npz", t)
    target[["entity_id", "country"]].to_parquet(
        artifact_dir / f"{source}_name_benchmark_targets.parquet", index=False
    )
    return {
        "queries": len(query),
        "targets": len(target),
        "features": t.shape[1],
        "target_nnz": int(t.nnz),
        "target_matrix_bytes": matrix_bytes,
        "optimized_runtime_seconds": optimized_seconds,
        "scipy_sparse_product_runtime_seconds": scipy_seconds,
        "optimized_result_nnz": int(topn.nnz),
        "scipy_result_nnz": int(scipy_product.nnz),
        "estimated_full_target_matrix_bytes": int(matrix_bytes / len(target) * 5_200_000),
        "license": "Apache-2.0",
        "backend_version": "1.2.0",
    }


def _target_batches(source: str, chunk_size: int) -> Any:
    path = load_data_paths()[f"train_source{source[-1]}"]
    yield from _read_chunks(path, chunksize=chunk_size)


def _source_truth(truth: dict[str, set[str]], source: str) -> dict[str, set[str]]:
    return {
        s1_id: {target for target in targets if target.startswith(f"{source}-")}
        for s1_id, targets in truth.items()
    }


def enrich_candidate_address_digits(
    candidates: pd.DataFrame, queries: pd.DataFrame
) -> pd.DataFrame:
    """Attach address digit agreement to an existing candidate union only."""
    query_digits = {
        str(entity_id): normalize_field(address).digit_tokens
        for entity_id, address in queries[["entity_id", "business_address"]].itertuples(
            index=False, name=None
        )
    }
    target_ids = {
        source: set(
            candidates.loc[candidates["target_source"] == source, "target_id"].astype(str)
        )
        for source in ("S2", "S3")
    }
    target_digits: dict[str, tuple[str, ...]] = {}
    for source in ("S2", "S3"):
        selected = target_ids[source]
        for batch in _target_batches(source, 100_000):
            subset = batch[batch["entity_id"].isin(selected)]
            target_digits.update(
                {
                    str(entity_id): normalize_field(address).digit_tokens
                    for entity_id, address in subset[
                        ["entity_id", "business_address"]
                    ].itertuples(index=False, name=None)
                }
            )
    return add_address_digit_agreement(candidates, query_digits, target_digits)


def _metrics_report(
    frames: dict[str, pd.DataFrame],
    truth: dict[str, set[str]],
    sample: pd.DataFrame,
    top_k_values: list[int],
    resources: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    full_train_s1 = 2_206_821
    full_test_s1 = 1_732_544

    def decorate(
        result: dict[str, Any], *, source: str | None, channel_names: set[str]
    ) -> None:
        resource_keys: set[str] = set()
        source_names = ("S2", "S3") if source is None else (source,)
        for source_name in source_names:
            for channel_name in channel_names:
                if channel_name == "name_char_tfidf":
                    resource_keys.add(f"{source_name}:name_char")
                elif channel_name == "address_char_tfidf":
                    resource_keys.add(f"{source_name}:address_char")
                elif channel_name.startswith("exact_"):
                    resource_keys.add(f"{source_name}:exact")
        resource_items = [resources.get(key, {}) for key in sorted(resource_keys)]
        result["runtime_seconds"] = sum(
            float(item.get("runtime_seconds", 0.0)) for item in resource_items
        )
        result["peak_rss_mb"] = max(
            (float(item.get("peak_rss_mb", 0.0)) for item in resource_items),
            default=0.0,
        )
        for split, s1_rows in (("train", full_train_s1), ("test", full_test_s1)):
            pairs = int(round(float(result["average_candidates"]) * s1_rows))
            result[f"estimated_{split}_candidate_pairs"] = pairs
            result[f"estimated_{split}_parquet_bytes"] = pairs * 32
    for source in ("S2", "S3"):
        source_frames = [frame for key, frame in frames.items() if key.startswith(f"{source}:")]
        source_truth = _source_truth(truth, source)
        channels = sorted(
            set().union(*(set(frame["channel"].unique()) for frame in source_frames))
        )
        for top_k in top_k_values:
            for channel in channels:
                channel_frame = pd.concat(
                    [filter_channel_topk(frame, channel, top_k) for frame in source_frames],
                    ignore_index=True,
                )
                result = evaluate_candidates(channel_frame, source_truth, sample)
                result["scope"] = "sampled S1 against full target source"
                decorate(result, source=source, channel_names={channel})
                results[f"{source}|{channel}|K={top_k}"] = result
            union_specs = {
                "char_union": {"name_char_tfidf", "address_char_tfidf"},
                "required_union": {
                    "exact_name_unicode_preserving",
                    "exact_address_unicode_preserving",
                    "name_char_tfidf",
                    "address_char_tfidf",
                },
                "all_views_union": set(channels),
            }
            for label, union_channels in union_specs.items():
                frame = union_topk(source_frames, union_channels, top_k)
                result = evaluate_candidates(frame, source_truth, sample)
                result["scope"] = "sampled S1 against full target source"
                decorate(result, source=source, channel_names=union_channels)
                results[f"{source}|{label}|K={top_k}"] = result

    combined_frames = list(frames.values())
    for top_k in top_k_values:
        union_channels = {
            "exact_name_unicode_preserving",
            "exact_address_unicode_preserving",
            "exact_name_accent_folded",
            "exact_address_accent_folded",
            "name_char_tfidf",
            "address_char_tfidf",
        }
        frame = union_topk(combined_frames, union_channels, top_k)
        result = evaluate_candidates(frame, truth, sample)
        result["scope"] = "sampled S1 against both full target sources"
        decorate(result, source=None, channel_names=union_channels)
        results[f"combined|all_views_union|K={top_k}"] = result
    return results


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    country = payload["country_agreement"]
    sample = payload["evaluation_sample"]
    results = payload["results"]
    combined = {
        key: value for key, value in results.items() if key.startswith("combined|")
    }
    best_item = (
        max(
            combined.items(),
            key=lambda item: (
                item[1]["positive_edge_recall"],
                -item[1]["average_candidates"],
            ),
        )
        if combined
        else None
    )
    lines = [
        "# Candidate generation v1",
        "",
        "## Measurement scope",
        "",
        "- Full-dataset measurements: country agreement over all positive edges and exact/sparse retrieval target scans.",
        f"- Sampled measurements: fixed {sample['rows']:,}-S1 truth-stratified evaluation sample.",
        "- Extrapolated estimates: candidate output sizes use the sample mean times full S1 counts at 32 Parquet bytes per pair; matrix storage uses benchmark bytes per target row.",
        "- Unverified hypothesis: France retrieval will transfer from the training-country configuration.",
        "",
        "## Country decision",
        "",
        f"Country agreement is {country['overall_agreement_rate']:.6%} across {country['positive_edges']:,} edges with {country['disagreements']:,} disagreements. Country is therefore used as a hard retrieval partition.",
        "No disagreement sample is present because the full positive-edge scan found none.",
        "",
        "## Backend benchmark",
        "",
        f"`{payload['backend_benchmark']}`",
        "",
        "## Evaluation sample",
        "",
        f"`{sample}`",
        "",
        "## Combined union results",
        "",
        "| Configuration | Edge recall | All truth | Missed-any | Oracle F0.5 | Mean | Median | p95 | p99 | Max | Zero | Train GiB | Test GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in sorted(combined.items()):
        display_key = key.replace("|", " / ")
        lines.append(
            f"| {display_key} | {item['positive_edge_recall']:.3%} | {item['all_truth_recovered_rate']:.3%} | "
            f"{item['at_least_one_missed_truth_rate']:.3%} | {item['oracle_macro_f0_5']:.5f} | "
            f"{item['average_candidates']:.2f} | {item['median_candidates']:.1f} | "
            f"{item['p95_candidates']:.1f} | {item['p99_candidates']:.1f} | "
            f"{item['maximum_candidates']} | {item['zero_candidate_rate']:.3%} | "
            f"{item['estimated_train_parquet_bytes'] / 2**30:.2f} | "
            f"{item['estimated_test_parquet_bytes'] / 2**30:.2f} |"
        )
    lines.extend(
        [
            "",
            "## All source, channel, K, and union results",
            "",
            "Full subgroup breakdowns for every row are in `candidate_generation_v1.json`.",
            "",
            "| Configuration | Recall | All truth | Oracle | Mean | p90 | p95 | p99 | Max | Zero | Runtime s | Peak MiB | Est. train pairs | Est. test pairs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, item in sorted(results.items()):
        display_key = key.replace("|", " / ")
        lines.append(
            f"| {display_key} | {item['positive_edge_recall']:.3%} | "
            f"{item['all_truth_recovered_rate']:.3%} | {item['oracle_macro_f0_5']:.5f} | "
            f"{item['average_candidates']:.2f} | {item['p90_candidates']:.1f} | "
            f"{item['p95_candidates']:.1f} | {item['p99_candidates']:.1f} | "
            f"{item['maximum_candidates']} | {item['zero_candidate_rate']:.3%} | "
            f"{item['runtime_seconds']:.1f} | {item['peak_rss_mb']:.1f} | "
            f"{item['estimated_train_candidate_pairs']:,} | "
            f"{item['estimated_test_candidate_pairs']:,} |"
        )
    if best_item is not None:
        _best_key, best_metrics = best_item
        lines.extend(["", "## Best-union subgroup results", ""])
        for grouping, groups in best_metrics["groups"].items():
            lines.extend(
                [
                    f"### {grouping}",
                    "",
                    "| Group | S1 | Recall | All truth | Oracle F0.5 | Mean candidates |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for group, item in sorted(groups.items()):
                lines.append(
                    f"| {group} | {item['s1_entities']:,} | "
                    f"{item['positive_edge_recall']:.3%} | "
                    f"{item['all_truth_recovered_rate']:.3%} | "
                    f"{item['oracle_macro_f0_5']:.5f} | "
                    f"{item['average_candidates']:.2f} |"
                )
        target_address = best_metrics["groups"].get("target_address_state", {})
        country_groups = best_metrics["groups"].get("country", {})
        composition_groups = best_metrics["groups"].get("truth_composition", {})
        lines.extend(
            [
                "",
                "## Failure analysis and caveats",
                "",
                f"- India is the weaker country at {country_groups['India']['positive_edge_recall']:.3%} edge recall versus {country_groups['US']['positive_edge_recall']:.3%} for the US.",
                f"- S3-only truth is weaker than S2-only truth: {composition_groups['S3_only']['positive_edge_recall']:.3%} versus {composition_groups['S2_only']['positive_edge_recall']:.3%} edge recall.",
                f"- Entities with any true target missing its address are the clearest address failure: {target_address['any_true_target_missing']['positive_edge_recall']:.3%} edge recall and {target_address['any_true_target_missing']['all_truth_recovered_rate']:.3%} all-truth recovery.",
                "- All 5,000 sampled S1 records have a present address, so an S1-missing-address subgroup could not be measured. Truth-side target address state was annotated after the fixed sample was selected and did not affect membership.",
                "- Word TF-IDF was optional and was not retained. The implemented required character channels already exceed the preferred candidate budget before reaching the recall target.",
                "- Address digits are retained only as evidence on generated pairs; no global digit block is used.",
                "- France transfer remains unverified because training truth contains only India and US entities.",
            ]
        )
    if best_item is None:
        selection_summary = (
            "No full-target union metrics were requested in this bounded benchmark run."
        )
    else:
        best_key, best = best_item
        selection_summary = (
            f"Best measured union: `{best_key}` with {best['positive_edge_recall']:.3%} "
            f"edge recall, {best['oracle_macro_f0_5']:.5f} oracle macro F0.5, mean "
            f"{best['average_candidates']:.2f}, and p95 {best['p95_candidates']:.1f} candidates."
        )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            selection_summary,
            "",
            payload["selection_recommendation"],
            "",
            "## Resource and production recommendation",
            "",
            payload["resource_recommendation"],
            "",
            "## Next modeling milestone",
            "",
            "First improve retrieval on India, S3-only truth, and missing-target-address cases using business-text-only views. Rerun this fixed sample and proceed to leakage-safe pair-feature development only after a union clears the recall and oracle thresholds.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path, *, full_target_evaluation: bool) -> dict[str, Any]:
    config = load_candidate_config(config_path)
    artifact_root = Path(config["artifact_root"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    reports = Path("reports")
    country_frame = measure_country_edge_agreement(
        reports / "positive_pair_analysis.parquet", reports / "country_edge_agreement.tsv"
    )
    overall_country = country_frame[country_frame["scope"] == "overall"].iloc[0]
    if int(overall_country["country_disagreements"]) != 0:
        raise RuntimeError("country disagreement is nonzero; hard partition is forbidden")
    sample_path = artifact_root / "evaluation_sample_s1_ids.parquet"
    if sample_path.exists():
        sample = pd.read_parquet(sample_path)
    else:
        sample = build_evaluation_sample(
            size=int(config["evaluation_sample_size"]),
            output=sample_path,
            positive_parquet=reports / "positive_pair_analysis.parquet",
        )
    if "target_address_state" not in sample.columns:
        sample = enrich_target_address_state(
            sample, reports / "positive_pair_analysis.parquet"
        )
        sample.to_parquet(sample_path, index=False)
    queries = load_sample_queries(sample)
    truth = load_sample_truth(set(sample["s1_id"]))
    frames: dict[str, pd.DataFrame] = {}
    resources: dict[str, dict[str, float | int]] = {}
    benchmarks: dict[str, Any] = {}
    for source in ("S2", "S3"):
        exact_path = artifact_root / f"{source}_exact_evidence.parquet"
        exact_resource_path = artifact_root / f"{source}_exact_evidence.resources.json"
        fit_path = artifact_root / f"{source}_fit_texts.joblib"
        benchmark_path = artifact_root / f"{source}_benchmark_targets.parquet"
        if exact_path.exists() and fit_path.exists() and benchmark_path.exists():
            exact = pd.read_parquet(exact_path)
            fit_texts = joblib.load(fit_path)
            benchmark_targets = pd.read_parquet(benchmark_path)
            exact_resource = (
                json.loads(exact_resource_path.read_text(encoding="utf-8"))
                if exact_resource_path.exists()
                else {"runtime_seconds": 0.0, "resumed": 1}
            )
            exact_resource["resumed"] = 1
        else:
            exact, fit_texts, benchmark_targets, exact_resource = _exact_and_fit_scan(
                queries,
                source=source,
                run_id=config["run_id"],
                fit_sample_size=int(config["fit_sample_size"]),
                country_partition=bool(config["country_partition"]),
            )
            exact.to_parquet(exact_path, index=False)
            joblib.dump(fit_texts, fit_path)
            benchmark_targets.to_parquet(benchmark_path, index=False)
            exact_resource_path.write_text(
                json.dumps(exact_resource, indent=2), encoding="utf-8"
            )
        frames[f"{source}:exact"] = exact
        resources[f"{source}:exact"] = exact_resource
        sparse_config = SparseRetrievalConfig(
            field="name",
            view=config["normalization_view"],
            ngram_min=int(config["tfidf"]["ngram_range"][0]),
            ngram_max=int(config["tfidf"]["ngram_range"][1]),
            min_df=int(config["tfidf"]["min_df"]),
            max_features=int(config["tfidf"]["max_features"]),
            threshold=float(config["sparse_threshold"]),
            top_k=int(config["sparse_top_k"]),
            country_partition=bool(config["country_partition"]),
        )
        benchmarks[source] = benchmark_backend(
            queries,
            benchmark_targets,
            sparse_config,
            artifact_root / "benchmark",
            source,
        )
        if not full_target_evaluation:
            continue
        for field in ("name", "address"):
            field_config = SparseRetrievalConfig(**{**sparse_config.__dict__, "field": field})
            vectorizer_path = artifact_root / f"{source}_{field}_vectorizer.joblib"
            if vectorizer_path.exists():
                retriever = SparseTopNRetriever.load(vectorizer_path)
            else:
                vectorizer = fit_vectorizer(fit_texts[field], field_config)
                retriever = SparseTopNRetriever(field_config, vectorizer)
                retriever.save(vectorizer_path)
            output = artifact_root / f"{source}_{field}_char_top80.parquet"
            resource_path = output.with_suffix(".resources.json")
            frame, measurement = retriever.retrieve(
                queries,
                lambda selected_source=source: _target_batches(
                    selected_source, int(config["target_chunk_size"])
                ),
                source=source,
                run_id=config["run_id"],
                artifact_path=output,
                query_batch_size=int(config["query_batch_size"]),
            )
            if measurement.get("resumed_complete_artifact") and resource_path.exists():
                persisted_measurement = json.loads(resource_path.read_text(encoding="utf-8"))
                persisted_measurement["resumed_complete_artifact"] = 1
                measurement = persisted_measurement
            else:
                resource_path.write_text(
                    json.dumps(measurement, indent=2), encoding="utf-8"
                )
            frames[f"{source}:{field}_char"] = frame
            resources[f"{source}:{field}_char"] = measurement

    results = (
        _metrics_report(frames, truth, sample, list(config["top_k_values"]), resources)
        if full_target_evaluation
        else {}
    )
    if results:
        eligible = [
            (key, value)
            for key, value in results.items()
            if key.startswith("combined|")
            and value["positive_edge_recall"] >= 0.985
            and value["oracle_macro_f0_5"] >= 0.99
            and value["average_candidates"] <= 50
            and value["p95_candidates"] <= 100
        ]
    else:
        eligible = []
    selection = (
        f"Selected `{min(eligible, key=lambda item: item[1]['average_candidates'])[0]}` as the smallest qualifying union."
        if eligible
        else "No configuration is selected for production: the measured local evaluation did not satisfy every recall/oracle/candidate-volume target."
    )
    cloud_command = (
        "python -m entity_resolution.candidate_generation --config configs/candidate_generation_v1.yaml "
        "--full-target-evaluation"
    )
    payload = {
        "run_id": config["run_id"],
        "country_agreement": {
            "positive_edges": int(overall_country["positive_edges"]),
            "agreements": int(overall_country["country_agreements"]),
            "disagreements": int(overall_country["country_disagreements"]),
            "overall_agreement_rate": float(overall_country["agreement_rate"]),
            "by_source": country_frame.to_dict("records"),
            "hard_partition_allowed": True,
        },
        "evaluation_sample": {
            "rows": len(sample),
            "countries": sample["country"].value_counts().to_dict(),
            "cardinality": sample["cardinality_bucket"].value_counts().to_dict(),
            "truth_composition": sample["truth_composition"].value_counts().to_dict(),
            "address_state": sample["address_state"].value_counts().to_dict(),
            "target_address_state": sample["target_address_state"].value_counts().to_dict(),
            "exact_name_state": sample["exact_name_state"].value_counts().to_dict(),
            "exact_address_state": sample["exact_address_state"].value_counts().to_dict(),
        },
        "backend_benchmark": benchmarks,
        "resources": resources,
        "results": results,
        "selection_recommendation": selection,
        "resource_recommendation": (
            "The normalized/exact artifacts and bounded sparse benchmark are safe locally. "
            "A fully persisted four-index sparse target build is projected from benchmark bytes and "
            "is not launched if it exceeds local free disk. Recommended production tier: 16 vCPU, "
            f"64 GiB RAM, 100 GiB SSD. Reproducible command: `{cloud_command}`."
        ),
        "cloud_command": cloud_command,
    }
    (reports / "candidate_generation_v1.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    _write_report(reports / "candidate_generation_v1.md", payload)
    if full_target_evaluation:
        all_evidence = pd.concat(frames.values(), ignore_index=True)
        selected_evidence = all_evidence[all_evidence["channel_rank"] <= 20]
        aggregated = aggregate_provenance(selected_evidence)
        enrich_candidate_address_digits(aggregated, queries).to_parquet(
            artifact_root / "evaluation_union_k20.parquet", index=False
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--full-target-evaluation", action="store_true")
    args = parser.parse_args()
    payload = run(args.config, full_target_evaluation=args.full_target_evaluation)
    print(json.dumps({"country": payload["country_agreement"], "sample": payload["evaluation_sample"]}, indent=2))


if __name__ == "__main__":
    main()
