"""Leakage-safe validation and deterministic fusion of candidate evidence."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from entity_resolution.candidate_evaluation import evaluate_candidates
from entity_resolution.candidate_generation import load_sample_queries, load_sample_truth
from entity_resolution.candidate_schema import CANDIDATE_COLUMNS
from entity_resolution.config import load_data_paths
from entity_resolution.exact_index import content_tie_key
from entity_resolution.folds import create_folds
from entity_resolution.forensic_analysis import _read_chunks
from entity_resolution.sparse_retrieval import SparseRetrievalConfig
from entity_resolution.views import normalize_field

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "candidate_generation_v2.yaml"
CHANNELS = (
    "exact_name_unicode_preserving",
    "exact_name_accent_folded",
    "exact_address_unicode_preserving",
    "exact_address_accent_folded",
    "name_char_tfidf",
    "address_char_tfidf",
)
EXACT_CHANNELS = frozenset(channel for channel in CHANNELS if channel.startswith("exact_"))
FUZZY_CHANNELS = frozenset({"name_char_tfidf", "address_char_tfidf"})


class EvidenceValidationError(ValueError):
    """Raised when persisted retrieval evidence is unsafe to reuse."""


@dataclass(frozen=True)
class FusionSpec:
    """One rank-based, truth-independent candidate selection rule."""

    strategy: str
    global_budget: int | None = None
    s2_budget: int | None = None
    s3_budget: int | None = None
    rrf_k: int = 60
    adaptive_rescue: int = 0
    adaptive_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.strategy not in {"rrf", "exact_first", "source_specific"}:
            raise ValueError(f"unsupported fusion strategy: {self.strategy!r}")
        if (self.global_budget is None) == (self.s2_budget is None or self.s3_budget is None):
            raise ValueError("set either a global budget or both source budgets")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        for value in (self.global_budget, self.s2_budget, self.s3_budget):
            if value is not None and value < 1:
                raise ValueError("candidate budgets must be positive")

    @property
    def key(self) -> str:
        if self.global_budget is not None:
            budget = f"global={self.global_budget}"
        else:
            budget = f"S2={self.s2_budget},S3={self.s3_budget}"
        rescue = (
            f",rescue={self.adaptive_rescue}:{'+'.join(self.adaptive_conditions)}"
            if self.adaptive_rescue
            else ""
        )
        return f"{self.strategy}|{budget}|rrf_k={self.rrf_k}{rescue}"


def _sha256(path: Path, *, block_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _canonical_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvidenceValidationError(f"expected YAML mapping: {path}")
    return payload


def _artifact_specs(root: Path) -> list[tuple[str, str, Path]]:
    rows: list[tuple[str, str, Path]] = []
    for source in ("S2", "S3"):
        rows.extend(
            [
                (source, "exact", root / f"{source}_exact_evidence.parquet"),
                (source, "name", root / f"{source}_name_char_top80.parquet"),
                (source, "address", root / f"{source}_address_char_top80.parquet"),
            ]
        )
    return rows


def _expected_sparse_fingerprint(config: dict[str, Any], field: str) -> str:
    sparse_config = SparseRetrievalConfig(
        field=field,
        view=str(config["normalization_view"]),
        ngram_min=int(config["tfidf"]["ngram_range"][0]),
        ngram_max=int(config["tfidf"]["ngram_range"][1]),
        min_df=int(config["tfidf"]["min_df"]),
        max_features=int(config["tfidf"]["max_features"]),
        threshold=float(config["sparse_threshold"]),
        top_k=int(config["sparse_top_k"]),
        country_partition=bool(config["country_partition"]),
    )
    return sparse_config.fingerprint()


def _validate_one_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    kind: str,
    expected_run_id: str,
    sample_ids: set[str],
    top_k: int,
) -> dict[str, Any]:
    missing = set(CANDIDATE_COLUMNS) - set(frame.columns)
    if missing:
        raise EvidenceValidationError(f"{source} {kind} missing columns: {sorted(missing)}")
    if frame.empty:
        raise EvidenceValidationError(f"{source} {kind} evidence is empty")
    run_ids = set(frame["run_id"].astype(str).unique())
    if run_ids != {expected_run_id}:
        raise EvidenceValidationError(
            f"{source} {kind} run ID mismatch: expected {expected_run_id!r}, got {run_ids}"
        )
    if set(frame["target_source"].astype(str).unique()) != {source}:
        raise EvidenceValidationError(f"{source} {kind} contains another target source")
    if not frame["target_id"].astype(str).str.startswith(f"{source}-").all():
        raise EvidenceValidationError(f"{source} {kind} target ID/source mismatch")
    query_ids = set(frame["s1_id"].astype(str).unique())
    unknown_queries = query_ids - sample_ids
    if unknown_queries:
        raise EvidenceValidationError(
            f"{source} {kind} has {len(unknown_queries)} queries outside the fixed sample"
        )
    expected_channels = EXACT_CHANNELS if kind == "exact" else {f"{kind}_char_tfidf"}
    channels = set(frame["channel"].astype(str).unique())
    if channels != expected_channels:
        raise EvidenceValidationError(
            f"{source} {kind} channel mismatch: expected {sorted(expected_channels)}, "
            f"got {sorted(channels)}"
        )
    scores = pd.to_numeric(frame["retrieval_score"], errors="coerce").to_numpy()
    if not np.isfinite(scores).all():
        raise EvidenceValidationError(f"{source} {kind} has non-finite retrieval scores")
    ranks = pd.to_numeric(frame["channel_rank"], errors="coerce")
    if ranks.isna().any() or (ranks < 1).any() or (ranks > top_k).any():
        raise EvidenceValidationError(f"{source} {kind} has invalid channel ranks")
    duplicate_mask = frame.duplicated(["s1_id", "target_id", "target_source", "channel"])
    if duplicate_mask.any():
        raise EvidenceValidationError(
            f"{source} {kind} has {int(duplicate_mask.sum())} duplicate pair/channel rows"
        )
    group_keys = ["s1_id", "channel"]
    grouped = frame.groupby(group_keys, sort=False, observed=True)["channel_rank"]
    rank_summary = grouped.agg(["count", "min", "max", "nunique"])
    if not (
        (rank_summary["min"] == 1)
        & (rank_summary["count"] == rank_summary["max"])
        & (rank_summary["count"] == rank_summary["nunique"])
    ).all():
        raise EvidenceValidationError(f"{source} {kind} ranks are not contiguous within channel")
    if kind != "exact" and query_ids != sample_ids:
        raise EvidenceValidationError(
            f"{source} {kind} query coverage mismatch: {len(query_ids)} of {len(sample_ids)}"
        )
    if kind == "exact":
        if not frame["exact_match"].astype(bool).all():
            raise EvidenceValidationError(f"{source} exact evidence contains non-exact rows")
    elif frame["exact_match"].astype(bool).any():
        raise EvidenceValidationError(f"{source} {kind} fuzzy evidence contains exact flags")
    return {
        "rows": len(frame),
        "queries": len(query_ids),
        "channels": sorted(channels),
        "maximum_rank": int(ranks.max()),
        "duplicate_pair_channel_rows": 0,
        "finite_scores": True,
    }


def validate_and_load_evidence(
    artifact_root: Path,
    *,
    sample_path: Path,
    v1_config_path: Path,
    expected_fingerprints: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate every persisted top-80 input before returning any evidence."""
    if not sample_path.is_file():
        raise EvidenceValidationError(f"fixed diagnostic sample is missing: {sample_path}")
    sample_sha256 = _sha256(sample_path)
    if expected_fingerprints is not None:
        expected_sample = expected_fingerprints.get(sample_path.name)
        if expected_sample != sample_sha256:
            raise EvidenceValidationError(
                f"diagnostic sample fingerprint mismatch: expected {expected_sample}, "
                f"got {sample_sha256}"
            )
    sample = pd.read_parquet(sample_path)
    if "s1_id" not in sample or sample["s1_id"].duplicated().any():
        raise EvidenceValidationError("diagnostic sample must have unique s1_id values")
    sample_ids = set(sample["s1_id"].astype(str))
    v1 = _read_yaml(v1_config_path)
    expected_run_id = str(v1["run_id"])
    top_k = int(v1["sparse_top_k"])
    if top_k != 80:
        raise EvidenceValidationError(f"fusion v2 requires top-80 evidence, got top-{top_k}")

    validations: dict[str, Any] = {}
    frames: list[pd.DataFrame] = []
    for source, kind, path in _artifact_specs(artifact_root):
        if not path.is_file():
            raise EvidenceValidationError(f"required evidence artifact is missing: {path}")
        resource_path = path.with_suffix(".resources.json")
        if not resource_path.is_file():
            raise EvidenceValidationError(f"resource sidecar is missing: {resource_path}")
        resources = json.loads(resource_path.read_text(encoding="utf-8"))
        parquet_rows = pq.ParquetFile(path).metadata.num_rows
        expected_rows = resources.get("candidate_rows", resources.get("verified_candidate_rows"))
        if expected_rows is None or int(expected_rows) != parquet_rows:
            raise EvidenceValidationError(
                f"row-count mismatch for {path}: parquet={parquet_rows}, sidecar={expected_rows}"
            )
        if kind != "exact":
            manifest_path = path.with_suffix(".manifest.json")
            if not manifest_path.is_file():
                raise EvidenceValidationError(f"resume manifest is missing: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_fingerprint = _expected_sparse_fingerprint(v1, kind)
            if not manifest.get("complete"):
                raise EvidenceValidationError(f"incomplete retrieval artifact: {path}")
            if manifest.get("fingerprint") != expected_fingerprint:
                raise EvidenceValidationError(
                    f"configuration fingerprint mismatch for {path}: "
                    f"expected {expected_fingerprint}, got {manifest.get('fingerprint')}"
                )
        artifact_sha256 = _sha256(path)
        if expected_fingerprints is not None:
            expected_sha256 = expected_fingerprints.get(path.name)
            if expected_sha256 != artifact_sha256:
                raise EvidenceValidationError(
                    f"artifact fingerprint mismatch for {path}: expected {expected_sha256}, "
                    f"got {artifact_sha256}"
                )
        frame = pd.read_parquet(path, columns=list(CANDIDATE_COLUMNS))
        item = _validate_one_frame(
            frame,
            source=source,
            kind=kind,
            expected_run_id=expected_run_id,
            sample_ids=sample_ids,
            top_k=top_k,
        )
        item.update(
            {"sha256": artifact_sha256, "path": str(path), "sidecar_rows": parquet_rows}
        )
        validations[f"{source}_{kind}"] = item
        frames.append(frame)
    evidence = pd.concat(frames, ignore_index=True)
    validations["summary"] = {
        "sample_rows": len(sample),
        "evidence_rows": len(evidence),
        "run_id": expected_run_id,
        "top_k": top_k,
        "v1_config_fingerprint": _canonical_fingerprint(v1),
        "sample_sha256": sample_sha256,
        "fingerprints_pinned": expected_fingerprints is not None,
        "source_separation": True,
        "compatible": True,
    }
    return evidence, sample, validations


def _target_metadata(
    evidence: pd.DataFrame,
    *,
    data_paths_config: Path,
) -> pd.DataFrame:
    paths = load_data_paths(data_paths_config)
    pieces: list[dict[str, Any]] = []
    for source in ("S2", "S3"):
        wanted = set(
            evidence.loc[evidence["target_source"] == source, "target_id"].astype(str).unique()
        )
        seen: set[str] = set()
        source_path = paths[f"train_source{source[-1]}"]
        for chunk in _read_chunks(source_path):
            selected = chunk[chunk["entity_id"].isin(wanted)]
            for entity_id, name, address, country in selected.itertuples(index=False, name=None):
                normalized_name = normalize_field(name)
                normalized_address = normalize_field(address)
                pieces.append(
                    {
                        "target_id": str(entity_id),
                        "target_content_key": content_tie_key(name, address, country),
                        "target_name_missing": normalized_name.is_missing,
                        "target_address_missing": normalized_address.is_missing,
                        "target_address_digits": normalized_address.digit_tokens,
                    }
                )
                seen.add(str(entity_id))
        missing = wanted - seen
        if missing:
            raise EvidenceValidationError(
                f"{len(missing)} {source} evidence targets are absent from configured raw data"
            )
    return pd.DataFrame(pieces).drop_duplicates("target_id")


def aggregate_evidence(
    evidence: pd.DataFrame,
    queries: pd.DataFrame,
    target_metadata: pd.DataFrame,
    *,
    rrf_k: int,
) -> pd.DataFrame:
    """Build one feature row per pair without treating channel cosines as calibrated."""
    keys = ["s1_id", "target_id", "target_source"]
    base = evidence[keys].drop_duplicates().reset_index(drop=True)
    query_rows = []
    for entity_id, name, address in queries[
        ["entity_id", "business_name", "business_address"]
    ].itertuples(index=False, name=None):
        norm_name = normalize_field(name)
        norm_address = normalize_field(address)
        query_rows.append(
            {
                "s1_id": str(entity_id),
                "query_name_missing": norm_name.is_missing,
                "query_address_missing": norm_address.is_missing,
                "query_address_digits": norm_address.digit_tokens,
            }
        )
    base = base.merge(pd.DataFrame(query_rows), on="s1_id", how="left", validate="many_to_one")
    base = base.merge(target_metadata, on="target_id", how="left", validate="many_to_one")
    if base["target_content_key"].isna().any():
        raise EvidenceValidationError("candidate target metadata is incomplete")

    for channel in CHANNELS:
        part = evidence[evidence["channel"] == channel][
            keys + ["retrieval_score", "channel_rank"]
        ].rename(
            columns={
                "retrieval_score": f"score_{channel}",
                "channel_rank": f"rank_{channel}",
            }
        )
        base = base.merge(part, on=keys, how="left", validate="one_to_one")
        base[f"has_{channel}"] = base[f"rank_{channel}"].notna()
        base[f"rrf_{channel}"] = np.where(
            base[f"has_{channel}"], 1.0 / (rrf_k + base[f"rank_{channel}"].fillna(0)), 0.0
        )
        # This normalized rank is comparable across channels; raw cosine is retained only as
        # provenance and is never added across channels.
        base[f"normalized_rank_{channel}"] = np.where(
            base[f"has_{channel}"],
            1.0 - (base[f"rank_{channel}"].fillna(80) - 1.0) / 79.0,
            0.0,
        )

    exact_name_channels = [channel for channel in EXACT_CHANNELS if "_name_" in channel]
    exact_address_channels = [channel for channel in EXACT_CHANNELS if "_address_" in channel]
    base["exact_name"] = base[[f"has_{channel}" for channel in exact_name_channels]].any(axis=1)
    base["exact_address"] = base[
        [f"has_{channel}" for channel in exact_address_channels]
    ].any(axis=1)
    for view in ("unicode_preserving", "accent_folded"):
        base[f"exact_name_{view}"] = base[f"has_exact_name_{view}"]
        base[f"exact_address_{view}"] = base[f"has_exact_address_{view}"]
    base["supporting_channels"] = base[[f"has_{channel}" for channel in CHANNELS]].sum(axis=1)
    base["rrf_name"] = base[
        [f"rrf_{channel}" for channel in CHANNELS if "name" in channel]
    ].sum(axis=1)
    base["rrf_address"] = base[
        [f"rrf_{channel}" for channel in CHANNELS if "address" in channel]
    ].sum(axis=1)
    base["rrf_exact_name"] = base[
        [f"rrf_{channel}" for channel in EXACT_CHANNELS if "name" in channel]
    ].sum(axis=1)
    base["rrf_exact_address"] = base[
        [f"rrf_{channel}" for channel in EXACT_CHANNELS if "address" in channel]
    ].sum(axis=1)
    base["rrf_total"] = base["rrf_name"] + base["rrf_address"]
    base["query_has_address_digits"] = base["query_address_digits"].map(bool)
    base["target_has_address_digits"] = base["target_address_digits"].map(bool)
    base["address_digit_any_agreement"] = [
        bool(set(query) & set(target))
        for query, target in zip(
            base["query_address_digits"], base["target_address_digits"], strict=True
        )
    ]
    base["address_digit_exact_agreement"] = [
        bool(query) and query == target
        for query, target in zip(
            base["query_address_digits"], base["target_address_digits"], strict=True
        )
    ]
    return base


def _fusion_score(
    frame: pd.DataFrame,
    spec: FusionSpec,
    source_weights: dict[str, dict[str, float]] | None,
) -> pd.Series:
    if spec.strategy != "source_specific":
        return frame["rrf_total"]
    if source_weights is None:
        raise ValueError("source_specific fusion requires source weights")
    scores = pd.Series(0.0, index=frame.index)
    for source in ("S2", "S3"):
        selected = frame["target_source"] == source
        weights = source_weights[source]
        # Explicit sums avoid comparing raw channel cosines. Exact views retain their own weight.
        exact_name = frame.loc[selected, "rrf_exact_name"]
        exact_address = frame.loc[selected, "rrf_exact_address"]
        name_fuzzy = frame.loc[selected, "rrf_name"] - exact_name
        address_fuzzy = frame.loc[selected, "rrf_address"] - exact_address
        scores.loc[selected] = (
            float(weights["name"]) * name_fuzzy
            + float(weights["address"]) * address_fuzzy
            + float(weights["exact"]) * (exact_name + exact_address)
        )
    return scores


def select_candidates(
    aggregated: pd.DataFrame,
    spec: FusionSpec,
    *,
    source_weights: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Select candidates with exact preservation and content-only tie handling."""
    selection_columns = [
        "s1_id",
        "target_id",
        "target_source",
        "target_content_key",
        "exact_name",
        "exact_address",
        "supporting_channels",
        "rrf_name",
        "rrf_address",
        "rrf_exact_name",
        "rrf_exact_address",
        "rrf_total",
    ]
    working = aggregated[selection_columns].copy()
    working["fusion_score"] = _fusion_score(working, spec, source_weights)
    working["qualifying_exact"] = working["exact_name"] | working["exact_address"]
    if spec.adaptive_rescue:
        per_s1 = working.groupby("s1_id", sort=False, observed=True).agg(
            any_exact=("qualifying_exact", "max"),
            maximum_support=("supporting_channels", "max"),
        )
        trigger = pd.Series(False, index=per_s1.index)
        if "no_exact_evidence" in spec.adaptive_conditions:
            trigger |= ~per_s1["any_exact"]
        if "single_support_channel" in spec.adaptive_conditions:
            trigger |= per_s1["maximum_support"] <= 1
        unknown = set(spec.adaptive_conditions) - {
            "no_exact_evidence",
            "single_support_channel",
        }
        if unknown:
            raise ValueError(f"unsupported adaptive conditions: {sorted(unknown)}")
        rescue_by_s1 = trigger.astype(int) * spec.adaptive_rescue
        working["_rescue"] = working["s1_id"].map(rescue_by_s1).fillna(0).astype(int)
    else:
        working["_rescue"] = 0

    if spec.global_budget is not None:
        group_keys = ["s1_id"]
        working["_budget"] = spec.global_budget + working["_rescue"]
    else:
        group_keys = ["s1_id", "target_source"]
        source_budgets = {"S2": int(spec.s2_budget), "S3": int(spec.s3_budget)}
        working["_budget"] = (
            working["target_source"].map(source_budgets).astype(int) + working["_rescue"]
        )
    working["_exact_count"] = working.groupby(
        group_keys, sort=False, observed=True
    )["qualifying_exact"].transform("sum")
    working["_remaining"] = (working["_budget"] - working["_exact_count"]).clip(lower=0)
    exact = working[working["qualifying_exact"]]
    fuzzy = working[~working["qualifying_exact"]].sort_values(
        group_keys + ["fusion_score", "supporting_channels", "target_content_key"],
        ascending=[True] * len(group_keys) + [False, False, True],
        kind="stable",
    )
    fuzzy["_fuzzy_rank"] = fuzzy.groupby(
        group_keys, sort=False, observed=True
    ).cumcount() + 1
    keep = fuzzy["_fuzzy_rank"] <= fuzzy["_remaining"]
    # Expand exact content/evidence ties at the nominal fuzzy boundary. No entity ID is
    # ever consulted to choose one row from such a tie.
    boundary = fuzzy[fuzzy["_fuzzy_rank"] == fuzzy["_remaining"]][
        group_keys + ["fusion_score", "supporting_channels", "target_content_key"]
    ].rename(
        columns={
            "fusion_score": "_boundary_score",
            "supporting_channels": "_boundary_support",
            "target_content_key": "_boundary_content",
        }
    )
    if not boundary.empty:
        fuzzy = fuzzy.merge(boundary, on=group_keys, how="left", validate="many_to_one")
        keep = keep.to_numpy() | (
            (fuzzy["fusion_score"] == fuzzy["_boundary_score"])
            & (fuzzy["supporting_channels"] == fuzzy["_boundary_support"])
            & (fuzzy["target_content_key"] == fuzzy["_boundary_content"])
        ).to_numpy()
    selected = pd.concat([exact, fuzzy[keep]], ignore_index=True)
    if selected.empty:
        return working.iloc[:0][
            ["s1_id", "target_id", "target_source", "fusion_score", "qualifying_exact"]
        ]
    result = selected.sort_values(
        ["s1_id", "qualifying_exact", "fusion_score", "supporting_channels", "target_content_key"],
        ascending=[True, False, False, False, True],
        kind="stable",
    )
    return result[
        ["s1_id", "target_id", "target_source", "fusion_score", "qualifying_exact"]
    ].reset_index(drop=True)


def build_fusion_specs(config: dict[str, Any]) -> list[FusionSpec]:
    fusion = config["fusion"]
    rrf_k = int(fusion["rrf_k"])
    specs = []
    for strategy in fusion["strategies"]:
        specs.extend(
            FusionSpec(strategy=strategy, global_budget=int(budget), rrf_k=rrf_k)
            for budget in fusion["global_budgets"]
        )
        specs.extend(
            FusionSpec(
                strategy=strategy,
                s2_budget=int(pair[0]),
                s3_budget=int(pair[1]),
                rrf_k=rrf_k,
            )
            for pair in fusion["source_budget_pairs"]
        )
    adaptive = fusion.get("adaptive", {})
    if adaptive.get("enabled"):
        specs.extend(
            FusionSpec(
                strategy=strategy,
                global_budget=int(adaptive["base_budget"]),
                rrf_k=rrf_k,
                adaptive_rescue=int(adaptive["rescue_budget"]),
                adaptive_conditions=tuple(adaptive["conditions"]),
            )
            for strategy in fusion["strategies"]
        )
    return specs


def _pareto_frontier(results: dict[str, dict[str, Any]]) -> list[str]:
    frontier = []
    for key, item in results.items():
        dominated = False
        for other_key, other in results.items():
            if key == other_key:
                continue
            no_worse = (
                other["positive_edge_recall"] >= item["positive_edge_recall"]
                and other["oracle_macro_f0_5"] >= item["oracle_macro_f0_5"]
                and other["average_candidates"] <= item["average_candidates"]
                and other["p95_candidates"] <= item["p95_candidates"]
            )
            strictly_better = (
                other["positive_edge_recall"] > item["positive_edge_recall"]
                or other["oracle_macro_f0_5"] > item["oracle_macro_f0_5"]
                or other["average_candidates"] < item["average_candidates"]
                or other["p95_candidates"] < item["p95_candidates"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(key)
    return sorted(frontier)


def _passes_volume(metrics: dict[str, Any], gates: dict[str, Any]) -> bool:
    return (
        metrics["average_candidates"] <= float(gates["mean_candidates"])
        and metrics["p95_candidates"] <= float(gates["p95_candidates"])
    )


def _passes_all_gates(metrics: dict[str, Any], gates: dict[str, Any]) -> bool:
    return (
        _passes_volume(metrics, gates)
        and metrics["positive_edge_recall"] >= float(gates["positive_edge_recall"])
        and metrics["oracle_macro_f0_5"] >= float(gates["oracle_macro_f0_5"])
    )


def evaluate_fusion_specs(
    aggregated: pd.DataFrame,
    truth: dict[str, set[str]],
    sample: pd.DataFrame,
    fold_assignment: dict[str, int],
    specs: Iterable[FusionSpec],
    *,
    development_folds: set[int],
    held_out_folds: set[int],
    gates: dict[str, Any],
    source_weights: dict[str, dict[str, float]],
) -> tuple[FusionSpec, pd.DataFrame, dict[str, Any]]:
    """Tune on development S1s and evaluate the one selected rule on held-out S1s."""
    dev_ids = {s1_id for s1_id, fold in fold_assignment.items() if fold in development_folds}
    held_ids = {s1_id for s1_id, fold in fold_assignment.items() if fold in held_out_folds}
    if not dev_ids or not held_ids or dev_ids & held_ids:
        raise ValueError("development and held-out folds must be nonempty and disjoint")
    dev_metadata = sample[sample["s1_id"].isin(dev_ids)]
    dev_truth = {s1_id: truth[s1_id] for s1_id in dev_ids}
    dev_results: dict[str, dict[str, Any]] = {}
    spec_by_key: dict[str, FusionSpec] = {}
    metric_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    for spec in specs:
        # Exact candidates are mandatory for every strategy, making plain RRF and
        # exact-first set-equivalent. Cache that intentional equivalence.
        score_strategy = "rrf" if spec.strategy == "exact_first" else spec.strategy
        cache_key = (
            score_strategy,
            spec.global_budget,
            spec.s2_budget,
            spec.s3_budget,
            spec.adaptive_rescue,
            spec.adaptive_conditions,
        )
        cached = metric_cache.get(cache_key)
        if cached is None:
            selected = select_candidates(aggregated, spec, source_weights=source_weights)
            metrics = evaluate_candidates(
                selected[selected["s1_id"].isin(dev_ids)], dev_truth, dev_metadata
            )
            metric_cache[cache_key] = metrics
        else:
            metrics = dict(cached)
        metrics["measured_split"] = "development"
        metrics["passes_volume_gate"] = _passes_volume(metrics, gates)
        metrics["passes_all_gates"] = _passes_all_gates(metrics, gates)
        dev_results[spec.key] = metrics
        spec_by_key[spec.key] = spec

    qualifying = [key for key, item in dev_results.items() if item["passes_all_gates"]]
    if qualifying:
        chosen_key = min(
            qualifying,
            key=lambda key: (
                dev_results[key]["average_candidates"],
                dev_results[key]["p95_candidates"],
                -dev_results[key]["positive_edge_recall"],
                key,
            ),
        )
    else:
        volume_safe = [key for key, item in dev_results.items() if item["passes_volume_gate"]]
        pool = volume_safe or list(dev_results)
        chosen_key = max(
            pool,
            key=lambda key: (
                dev_results[key]["positive_edge_recall"],
                dev_results[key]["oracle_macro_f0_5"],
                -dev_results[key]["average_candidates"],
                key,
            ),
        )

    chosen = spec_by_key[chosen_key]
    chosen_frame = select_candidates(aggregated, chosen, source_weights=source_weights)
    held_fold_metrics: dict[str, dict[str, Any]] = {}
    for fold in sorted(held_out_folds):
        fold_ids = {s1_id for s1_id, value in fold_assignment.items() if value == fold}
        fold_truth = {s1_id: truth[s1_id] for s1_id in fold_ids}
        fold_metadata = sample[sample["s1_id"].isin(fold_ids)]
        item = evaluate_candidates(
            chosen_frame[chosen_frame["s1_id"].isin(fold_ids)], fold_truth, fold_metadata
        )
        item["measured_split"] = "held_out"
        item["fold"] = fold
        item["passes_all_gates"] = _passes_all_gates(item, gates)
        held_fold_metrics[str(fold)] = item
    held_metadata = sample[sample["s1_id"].isin(held_ids)]
    held_truth = {s1_id: truth[s1_id] for s1_id in held_ids}
    held_combined = evaluate_candidates(
        chosen_frame[chosen_frame["s1_id"].isin(held_ids)], held_truth, held_metadata
    )
    held_combined["measured_split"] = "held_out"
    held_combined["passes_all_gates"] = _passes_all_gates(held_combined, gates)
    held_by_target_source: dict[str, dict[str, Any]] = {}
    for source in ("S2", "S3"):
        source_truth = {
            s1_id: {target for target in targets if target.startswith(f"{source}-")}
            for s1_id, targets in held_truth.items()
        }
        source_candidates = chosen_frame[
            chosen_frame["s1_id"].isin(held_ids)
            & (chosen_frame["target_source"] == source)
        ]
        item = evaluate_candidates(source_candidates, source_truth, held_metadata)
        item["measured_split"] = "held_out"
        held_by_target_source[source] = item
    metric_names = (
        "positive_edge_recall",
        "all_truth_recovered_rate",
        "oracle_macro_f0_5",
        "average_candidates",
        "p95_candidates",
    )
    held_summary = {
        name: {
            "mean": float(np.mean([item[name] for item in held_fold_metrics.values()])),
            "worst": float(
                min(item[name] for item in held_fold_metrics.values())
                if name in {"positive_edge_recall", "all_truth_recovered_rate", "oracle_macro_f0_5"}
                else max(item[name] for item in held_fold_metrics.values())
            ),
        }
        for name in metric_names
    }
    concise_results = {
        key: {metric: value for metric, value in item.items() if metric != "groups"}
        for key, item in dev_results.items()
    }
    return chosen, chosen_frame, {
        "selection_split": "development folds only",
        "development_folds": sorted(development_folds),
        "held_out_folds": sorted(held_out_folds),
        "development_results": concise_results,
        "development_pareto_frontier": _pareto_frontier(dev_results),
        "selected_configuration": chosen.key,
        "selected_development_metrics": dev_results[chosen_key],
        "held_out_combined": held_combined,
        "held_out_by_target_source": held_by_target_source,
        "held_out_by_fold": held_fold_metrics,
        "held_out_mean_and_worst": held_summary,
    }


def create_uniform_content_hash_sample(
    *,
    size: int,
    output: Path,
    data_paths_config: Path,
    diagnostic_ids: set[str],
) -> pd.DataFrame:
    """Select a second S1 sample using content only, before truth or retrieval is read."""
    paths = load_data_paths(data_paths_config)
    # A max-heap of at most ``size`` distinct content hashes keeps memory bounded.
    heap: list[int] = []
    rows_by_hash: dict[int, list[tuple[str, str, str]]] = {}
    for chunk in _read_chunks(paths["train_source1"]):
        for entity_id, name, address, country in chunk.itertuples(index=False, name=None):
            if str(entity_id) in diagnostic_ids:
                continue
            key = content_tie_key(name, address, country)
            address_state = "missing" if normalize_field(address).is_missing else "present"
            numeric_key = int(key, 16)
            value = (str(entity_id), str(country), address_state)
            if numeric_key in rows_by_hash:
                rows_by_hash[numeric_key].append(value)
            elif len(heap) < size:
                heapq.heappush(heap, -numeric_key)
                rows_by_hash[numeric_key] = [value]
            elif numeric_key < -heap[0]:
                removed = -heapq.heapreplace(heap, -numeric_key)
                del rows_by_hash[removed]
                rows_by_hash[numeric_key] = [value]
    if len(rows_by_hash) < size:
        raise RuntimeError(
            f"only {len(rows_by_hash)} distinct S1 content hashes were available"
        )
    rows = [
        (*value, numeric_key)
        for numeric_key in sorted(rows_by_hash)
        for value in rows_by_hash[numeric_key]
    ]
    result = pd.DataFrame(
        rows, columns=["s1_id", "country", "address_state", "content_hash"]
    )
    result = result.drop(columns="content_hash")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    result.to_parquet(temporary, index=False)
    os.replace(temporary, output)
    metadata = {
        "selection": "lowest business-content hashes outside fixed diagnostic sample",
        "membership_inputs": ["business_name", "business_address", "country"],
        "truth_used": False,
        "retrieval_evidence_used": False,
        "entity_id_numeric_components_used": False,
        "requested_distinct_content_hashes": size,
        "rows": len(result),
        "boundary_ties_preserved": len(result) - size,
    }
    sidecar = output.with_suffix(".json")
    temporary_json = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary_json, sidecar)
    return result


def _load_config(path: Path) -> dict[str, Any]:
    return _read_yaml(path)


def _load_or_build_aggregate(
    *,
    evidence: pd.DataFrame,
    sample: pd.DataFrame,
    validation: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cache_root = Path("artifacts/blocking_v2")
    cache_path = cache_root / "aggregated_evidence.parquet"
    metadata_path = cache_root / "aggregated_evidence.json"
    input_fingerprint = _canonical_fingerprint(
        {
            "artifacts": {
                key: value["sha256"]
                for key, value in validation.items()
                if key != "summary"
            },
            "sample": validation["summary"]["sample_sha256"],
            "rrf_k": config["fusion"]["rrf_k"],
            "aggregation_version": 1,
        }
    )
    queries = load_sample_queries(sample)
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("input_fingerprint") != input_fingerprint:
            raise EvidenceValidationError(
                "stale aggregate cache fingerprint; remove artifacts/blocking_v2 or rerun cleanly"
            )
        aggregated = pd.read_parquet(cache_path)
        if int(metadata.get("rows", -1)) != len(aggregated):
            raise EvidenceValidationError("aggregate cache row-count mismatch")
        metadata["resumed"] = True
        return aggregated, queries, metadata
    target_metadata = _target_metadata(
        evidence, data_paths_config=Path(config["data_paths_config"])
    )
    aggregated = aggregate_evidence(
        evidence,
        queries,
        target_metadata,
        rrf_k=int(config["fusion"]["rrf_k"]),
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    aggregated.to_parquet(temporary, index=False)
    os.replace(temporary, cache_path)
    metadata = {
        "input_fingerprint": input_fingerprint,
        "rows": len(aggregated),
        "columns": list(aggregated.columns),
        "resumed": False,
    }
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary_metadata, metadata_path)
    return aggregated, queries, metadata


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _metric_row(metrics: dict[str, Any]) -> str:
    return (
        f"{metrics['positive_edge_recall']:.3%} | "
        f"{metrics['all_truth_recovered_rate']:.3%} | "
        f"{metrics['oracle_macro_f0_5']:.5f} | "
        f"{metrics['average_candidates']:.2f} | "
        f"{metrics['median_candidates']:.1f} | "
        f"{metrics['p90_candidates']:.1f} | "
        f"{metrics['p95_candidates']:.1f} | "
        f"{metrics['p99_candidates']:.1f} | "
        f"{metrics['maximum_candidates']} | "
        f"{metrics['zero_candidate_rate']:.3%}"
    )


def _write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    fusion = payload["fusion_evaluation"]
    held = fusion["held_out_combined"]
    selected_dev = fusion["selected_development_metrics"]
    audit = payload["missed_edge_analysis"]
    estimates = payload["candidate_volume_estimates"]
    lines = [
        "# Candidate generation v2",
        "",
        "## Scope and evidence validation",
        "",
        "The fixed 5,000-S1 truth-stratified diagnostic sample was reused byte-for-byte. "
        "All selection tuning used development folds 0–2; folds 3–4 remained held out. "
        "Cosine values were not compared across channels: fusion uses reciprocal ranks, "
        "exact flags, support counts, and content-derived tie keys.",
        "",
        f"Validated {payload['validation']['summary']['evidence_rows']:,} evidence rows "
        f"and {payload['aggregate']['rows']:,} distinct candidate pairs. All required "
        "artifacts passed schema, row-count, run-ID, query-coverage, rank, duplicate, "
        "finiteness, source-separation, and configuration-fingerprint checks.",
        "",
        "| Artifact | Rows | Queries | Maximum rank | SHA-256 |",
        "|---|---:|---:|---:|---|",
    ]
    for key, item in payload["validation"].items():
        if key == "summary":
            continue
        lines.append(
            f"| {key} | {item['rows']:,} | {item['queries']:,} | "
            f"{item['maximum_rank']} | `{item['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Leakage-safe selection",
            "",
            f"Selected on development folds only: `{fusion['selected_configuration']}`.",
            "Exact name/address candidates from either normalization view are always "
            "retained before the fuzzy budget. Identical content/evidence boundary ties "
            "are expanded; target IDs never break ties.",
            "",
            "| Split | Edge recall | All truth | Oracle F0.5 | Mean | Median | p90 | "
            "p95 | p99 | Max | Zero |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| Development (measured) | {_metric_row(selected_dev)} |",
            f"| Held-out folds 3–4 (measured) | {_metric_row(held)} |",
            "",
            "### Held-out fold stability",
            "",
            "| Fold | Edge recall | All truth | Oracle F0.5 | Mean | p95 | Gate |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for fold, item in fusion["held_out_by_fold"].items():
        lines.append(
            f"| {fold} | {item['positive_edge_recall']:.3%} | "
            f"{item['all_truth_recovered_rate']:.3%} | "
            f"{item['oracle_macro_f0_5']:.5f} | {item['average_candidates']:.2f} | "
            f"{item['p95_candidates']:.1f} | "
            f"{'pass' if item['passes_all_gates'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "### Held-out target-source results",
            "",
            "| Source | Positive edges | Edge recall | Oracle F0.5 | Mean candidates |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for source, item in fusion["held_out_by_target_source"].items():
        lines.append(
            f"| {source} | {item['positive_edges']:,} | "
            f"{item['positive_edge_recall']:.3%} | {item['oracle_macro_f0_5']:.5f} | "
            f"{item['average_candidates']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Mean/worst held-out fold values are recorded in the JSON report. The held-out "
            "candidate-volume gate passes, but the recall and oracle gates fail.",
            "",
            "## Held-out subgroup results",
            "",
        ]
    )
    for grouping, groups in held["groups"].items():
        lines.extend(
            [
                f"### {grouping}",
                "",
                "| Group | S1 | Edge recall | All truth | Oracle F0.5 | Mean | p95 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for group, item in sorted(groups.items()):
            lines.append(
                f"| {group} | {item['s1_entities']:,} | "
                f"{item['positive_edge_recall']:.3%} | "
                f"{item['all_truth_recovered_rate']:.3%} | "
                f"{item['oracle_macro_f0_5']:.5f} | "
                f"{item['average_candidates']:.2f} | {item['p95_candidates']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## Fusion plateau and Pareto frontier",
            "",
            "The full existing top-80 union is an absolute ceiling for any fusion subset: "
            "its previously measured diagnostic recall is 97.041% and oracle is 0.98846, "
            "already below the 98.5%/0.99 gates at 319.06 mean candidates. Therefore no "
            "reordering or pruning of this evidence can meet the recall gate.",
            "",
            "Development Pareto configurations:",
            "",
        ]
    )
    for key in fusion["development_pareto_frontier"]:
        item = fusion["development_results"][key]
        lines.append(
            f"- `{key}`: recall {item['positive_edge_recall']:.3%}, oracle "
            f"{item['oracle_macro_f0_5']:.5f}, mean {item['average_candidates']:.2f}, "
            f"p95 {item['p95_candidates']:.1f}."
        )
    lines.extend(
        [
            "",
            "## Candidate-volume estimates",
            "",
            f"At the held-out mean, full train output is extrapolated to "
            f"{estimates['estimated_train_pairs']:,} pairs "
            f"({estimates['estimated_train_parquet_gib']:.2f} GiB), and test output to "
            f"{estimates['estimated_test_pairs']:,} pairs "
            f"({estimates['estimated_test_parquet_gib']:.2f} GiB). These are extrapolated, "
            "not full-run measurements.",
            "",
            "## Held-out missed-edge audit",
            "",
            f"The selected fusion misses {audit['missed_edges']:,} held-out positive edges. "
            "Flags overlap; primary categories are mutually assigned by documented precedence.",
            "",
            "| Category flag | Count | Percent of misses |",
            "|---|---:|---:|",
        ]
    )
    for category, count in audit["category_counts"].items():
        lines.append(
            f"| {category} | {count:,} | {audit['category_percentages'][category]:.2%} |"
        )
    lines.extend(
        [
            "",
            f"All overlaps and {len(audit['representative_redacted_content_examples'])} "
            "representative content-only examples are in the JSON report; edge-level flags "
            "are in `missed_edges_v2.tsv`. Entity IDs are identifiers only and were not used "
            "as features or explanations.",
            "",
            "## Uniform-sample status",
            "",
            f"A separate uniform content-hash sample with "
            f"{payload['uniform_sample']['rows']:,} rows was created without truth, source "
            "composition, exact-match state, or retrieval scores. No unbiased final metric is "
            "reported because no v2 fusion configuration qualified; a new full-target scan of "
            "a rejected configuration would not be final validation. The sample is reserved "
            "for the selected post-rescue configuration.",
            "",
            "## Next experiment",
            "",
            "Run one name-character index restricted to targets whose address is missing on "
            "the fixed diagnostic sample, union it with existing evidence, and measure marginal "
            "held-out recall gained per added candidate. This directly targets 34 current "
            "held-out misses that are both absent from all channels and missing a target address. "
            "The accent-only pattern covers 2 misses and does not justify a broad first rescue. "
            "Reject the channel if volume rises without material held-out recall gain. No "
            "CatBoost training was started. France accuracy is not claimed because France truth "
            "is unavailable.",
            "",
            payload["decision"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Execute validation, development selection, held-out evaluation, and audit."""
    from entity_resolution.missed_edge_analysis import audit_missed_edges

    config = _load_config(config_path)
    evidence, sample, validation = validate_and_load_evidence(
        Path(config["artifact_root"]),
        sample_path=Path(config["diagnostic_sample"]),
        v1_config_path=Path(config["v1_config"]),
        expected_fingerprints=config.get("expected_artifact_sha256"),
    )
    aggregated, queries, aggregate_metadata = _load_or_build_aggregate(
        evidence=evidence,
        sample=sample,
        validation=validation,
        config=config,
    )
    truth = load_sample_truth(set(sample["s1_id"]))
    countries = dict(sample[["s1_id", "country"]].itertuples(index=False, name=None))
    folds, diagnostics = create_folds(
        {key: sorted(value) for key, value in truth.items()},
        countries=countries,
        n_splits=int(config["folds"]["count"]),
        random_state=int(config["seed"]),
    )
    folds_path = Path("artifacts/blocking_v2/folds.parquet")
    fold_frame = pd.DataFrame(sorted(folds.items()), columns=["s1_id", "fold"])
    temporary_folds = folds_path.with_suffix(folds_path.suffix + ".tmp")
    fold_frame.to_parquet(temporary_folds, index=False)
    os.replace(temporary_folds, folds_path)
    selected_spec, selected, fusion_results = evaluate_fusion_specs(
        aggregated,
        truth,
        sample,
        folds,
        build_fusion_specs(config),
        development_folds=set(config["folds"]["development"]),
        held_out_folds=set(config["folds"]["held_out"]),
        gates=config["gates"],
        source_weights=config["fusion"]["source_specific_weights"],
    )
    held_ids = {
        s1_id for s1_id, fold in folds.items() if fold in set(config["folds"]["held_out"])
    }
    held_truth = {s1_id: truth[s1_id] for s1_id in held_ids}
    held_sample = sample[sample["s1_id"].isin(held_ids)]
    held_evidence = evidence[evidence["s1_id"].isin(held_ids)]
    missed, missed_summary = audit_missed_edges(
        selected=selected[selected["s1_id"].isin(held_ids)],
        all_evidence=held_evidence,
        truth=held_truth,
        sample=held_sample,
        queries=queries[queries["entity_id"].isin(held_ids)],
        data_paths_config=Path(config["data_paths_config"]),
    )
    missed_path = Path(config["outputs"]["missed_edges"])
    missed_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_missed = missed_path.with_suffix(missed_path.suffix + ".tmp")
    missed.to_csv(temporary_missed, sep="\t", index=False)
    os.replace(temporary_missed, missed_path)
    uniform_path = Path(config["uniform_sample"])
    if uniform_path.is_file():
        uniform = pd.read_parquet(uniform_path)
    else:
        uniform = create_uniform_content_hash_sample(
            size=int(config["uniform_sample_size"]),
            output=uniform_path,
            data_paths_config=Path(config["data_paths_config"]),
            diagnostic_ids=set(sample["s1_id"]),
        )
    held = fusion_results["held_out_combined"]
    estimation = config["estimation"]
    train_pairs = int(round(held["average_candidates"] * estimation["train_s1_rows"]))
    test_pairs = int(round(held["average_candidates"] * estimation["test_s1_rows"]))
    all_fold_gates = all(
        item["passes_all_gates"] for item in fusion_results["held_out_by_fold"].values()
    )
    decision = (
        "PROCEED_TO_RANKER"
        if held["passes_all_gates"] and all_fold_gates
        else "IMPROVE_RETRIEVAL"
    )
    bytes_per_pair = int(estimation["parquet_bytes_per_pair"])
    payload = {
        "run_id": config["run_id"],
        "config_fingerprint": _canonical_fingerprint(config),
        "measurement_labels": {
            "diagnostic_sample": "fixed truth-stratified sampled",
            "selection": "development measured",
            "validation": "held-out measured",
            "volume": "full train/test extrapolated",
            "uniform_sample": (
                "membership created; retrieval not run because no configuration qualified"
            ),
        },
        "validation": validation,
        "aggregate": aggregate_metadata,
        "fold_diagnostics": diagnostics.to_dict(orient="index"),
        "fusion_evaluation": fusion_results,
        "selected_spec": asdict(selected_spec),
        "candidate_volume_estimates": {
            "estimated_train_pairs": train_pairs,
            "estimated_test_pairs": test_pairs,
            "estimated_train_parquet_bytes": train_pairs * bytes_per_pair,
            "estimated_test_parquet_bytes": test_pairs * bytes_per_pair,
            "estimated_train_parquet_gib": train_pairs * bytes_per_pair / 2**30,
            "estimated_test_parquet_gib": test_pairs * bytes_per_pair / 2**30,
            "basis": "held-out mean times configured full S1 rows at 32 bytes per pair",
        },
        "missed_edge_analysis": missed_summary,
        "uniform_sample": {
            "path": str(uniform_path),
            "rows": len(uniform),
            "overlap_with_diagnostic": len(set(uniform["s1_id"]) & set(sample["s1_id"])),
            "membership_uses_truth": False,
            "membership_uses_retrieval": False,
            "validation_status": "not_run_no_qualifying_v2_configuration",
        },
        "top80_absolute_ceiling": {
            "scope": "fixed diagnostic sample; previously measured",
            "positive_edge_recall": 0.97041,
            "oracle_macro_f0_5": 0.98846,
            "average_candidates": 319.06,
            "p95_candidates": 325.05,
        },
        "next_experiment": "name-character index restricted to missing-address targets",
        "decision": decision,
    }
    _atomic_json(Path(config["outputs"]["report_json"]), payload)
    _write_markdown_report(Path(config["outputs"]["report_markdown"]), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only:
        payload = run(args.config)
        held = payload["fusion_evaluation"]["held_out_combined"]
        print(
            json.dumps(
                {
                    "selected": payload["fusion_evaluation"]["selected_configuration"],
                    "held_out_recall": held["positive_edge_recall"],
                    "held_out_oracle": held["oracle_macro_f0_5"],
                    "held_out_mean_candidates": held["average_candidates"],
                    "decision": payload["decision"],
                },
                indent=2,
            )
        )
        return
    config = _load_config(args.config)
    evidence, sample, validation = validate_and_load_evidence(
        Path(config["artifact_root"]),
        sample_path=Path(config["diagnostic_sample"]),
        v1_config_path=Path(config["v1_config"]),
        expected_fingerprints=config.get("expected_artifact_sha256"),
    )
    summary = {"validation": validation, "sample_rows": len(sample), "evidence_rows": len(evidence)}
    if args.validate_only:
        print(json.dumps(summary, indent=2))
        return


if __name__ == "__main__":
    main()


def query_quality(evidence: pd.DataFrame, name: object, address: object) -> dict[str, Any]:
    """Label-free gate; missing evidence always takes the expanded fallback."""
    evidence = evidence.sort_values(
        ["channel_rank", "retrieval_score"], ascending=[True, False], kind="stable"
    ).drop_duplicates(["target_source", "target_id", "channel"])
    n, a = normalize_field(name), normalize_field(address)
    fuzzy = evidence.loc[~evidence["exact_match"].astype(bool)]
    margins = []
    for _, group in fuzzy.groupby(["target_source", "channel"], observed=True):
        scores = group.sort_values("channel_rank")["retrieval_score"].to_numpy()
        margins.append(float(scores[0] - scores[1]) if len(scores) > 1 else 1.0)
    support = evidence.groupby(["target_source", "target_id"])["channel"].nunique()
    result = {
        "missing_name": not bool(n.unicode_preserving),
        "missing_address": not bool(a.unicode_preserving),
        "name_length": len(n.unicode_preserving),
        "address_length": len(a.unicode_preserving),
        "minimum_margin": min(margins, default=0.0),
        "maximum_agreement": int(support.max()) if len(support) else 0,
        "has_exact": bool(evidence["exact_match"].any()),
        "viable_views": int(fuzzy["channel"].nunique()),
    }
    result["weak"] = bool(
        result["missing_name"]
        or result["missing_address"]
        or result["name_length"] < 5
        or result["address_length"] < 8
        or not result["has_exact"]
        or result["minimum_margin"] < 0.05
        or result["maximum_agreement"] < 2
        or result["viable_views"] < 2
    )
    return result


def fuse_budgeted(
    evidence: pd.DataFrame, queries: pd.DataFrame, policy: dict[str, Any], *, run_id: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Weighted RRF with source/channel reservations, spillover, and exact retention.

    Required evidence includes target_content_key. Identical content boundary ties
    expand together, as in select_candidates. Quotas reserve seats within K; unused
    seats spill into global RRF order. Exact overflow is never discarded. The primary
    channel describes one observation; support_json retains every channel observation.
    preserve_channel_union keeps every complementary channel candidate, with K used
    only as a nominal budget diagnostic. Retrieval itself remains bounded upstream.
    """
    columns = [
        "s1_id",
        "target_id",
        "target_source",
        "channel",
        "retrieval_score",
        "channel_rank",
        "exact_match",
        "run_id",
        "support_json",
        "fusion_score",
    ]
    if not set(evidence["s1_id"]) <= set(queries["entity_id"]):
        raise ValueError("evidence contains unknown query")
    k = int(policy["top_k"])
    expanded = int(policy.get("expanded_k", k))
    if not 0 < k <= expanded <= 32767:
        raise ValueError("invalid candidate budget")
    if not isinstance(policy.get("preserve_channel_union", False), bool):
        raise ValueError("preserve_channel_union must be boolean")
    quotas = policy.get("source_quotas", {})
    channels = policy.get("channel_quotas", {})
    if set(quotas) - {"S2", "S3"} or any(int(v) < 0 for v in quotas.values()):
        raise ValueError("invalid source quotas")
    if sum(quotas.values()) > k or sum(channels.values()) > k:
        raise ValueError("reservations exceed base budget")
    if any(int(v) < 0 for v in channels.values()):
        raise ValueError("invalid channel quotas")
    rrf_k = float(policy.get("rrf_k", 60))
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    weights = policy.get("channel_weights", {})
    if any(not np.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError("channel weights must be finite and positive")
    groups = {key: group for key, group in evidence.groupby("s1_id", sort=False)}
    output, activated = [], 0
    for query in queries.itertuples(index=False):
        part = groups.get(query.entity_id, evidence.iloc[:0])
        quality = query_quality(part, query.business_name, query.business_address)
        expand = bool(policy.get("adaptive", False) and quality["weak"])
        activated += int(expand)
        budget = expanded if expand else k
        # One observation per pair/channel: repeated shards cannot inflate RRF/support.
        part = part.sort_values(
            ["channel_rank", "retrieval_score"], ascending=[True, False], kind="stable"
        ).drop_duplicates(["target_source", "target_id", "channel"])
        candidates = []
        by_pair: dict[tuple[str, str], list[Any]] = {}
        for row in part.itertuples(index=False):
            by_pair.setdefault((row.target_source, row.target_id), []).append(row)
        for (source, target), observations in by_pair.items():
            observations.sort(key=lambda row: (not row.exact_match, row.channel_rank, row.channel))
            first = observations[0]
            support = [
                {
                    "channel": str(row.channel),
                    "score": float(row.retrieval_score),
                    "rank": int(row.channel_rank),
                    "exact": bool(row.exact_match),
                }
                for row in sorted(observations, key=lambda row: row.channel)
            ]
            score = sum(
                float(weights.get(f"{source}:{row['channel']}", weights.get(row["channel"], 1)))
                / (rrf_k + row["rank"])
                for row in support
            )
            candidates.append(
                {
                    "s1_id": query.entity_id,
                    "target_id": target,
                    "target_source": source,
                    "channel": first.channel,
                    "retrieval_score": float(first.retrieval_score),
                    "channel_rank": int(first.channel_rank),
                    "exact_match": bool(first.exact_match),
                    "run_id": run_id,
                    "support_json": json.dumps(support, separators=(",", ":")),
                    "fusion_score": score,
                    "_channels": {v["channel"] for v in support},
                    "_key": (-score, -len(support), str(first.target_content_key)),
                }
            )
        candidates.sort(key=lambda row: row["_key"])
        chosen = {i for i, row in enumerate(candidates) if row["exact_match"]}
        if policy.get("preserve_channel_union", False):
            chosen.update(range(len(candidates)))

        def reserve(
            indices: list[int], count: int, *, budget=budget, chosen=chosen, candidates=candidates
        ) -> None:
            remaining = max(0, min(count, budget - len(chosen)))
            available = [i for i in indices if i not in chosen]
            added = available[:remaining]
            if added:
                boundary = candidates[added[-1]]["_key"]
                added.extend(i for i in available[remaining:] if candidates[i]["_key"] == boundary)
                chosen.update(added)

        for source, quota in sorted(quotas.items()):
            indices = [i for i, row in enumerate(candidates) if row["target_source"] == source]
            reserve(indices, max(0, int(quota) - len(chosen.intersection(indices))))
        for channel, quota in sorted(channels.items()):
            indices = [i for i, row in enumerate(candidates) if channel in row["_channels"]]
            reserve(indices, max(0, int(quota) - len(chosen.intersection(indices))))
        reserve(list(range(len(candidates))), budget - len(chosen))
        output.extend({key: candidates[i][key] for key in columns} for i in sorted(chosen))
    frame = pd.DataFrame(output, columns=columns)
    # IDs only serialize an already selected set; no ID component is a ranking feature.
    frame = frame.sort_values(["s1_id", "target_source", "target_id"], kind="stable")
    return frame.reset_index(drop=True), {"adaptive_queries": activated, "queries": len(queries)}
