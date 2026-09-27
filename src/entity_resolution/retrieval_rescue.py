"""Bounded, artifact-first retrieval evaluation and one missing-address name rescue.

Run fusion/budgets first:
  python -m entity_resolution.retrieval_rescue --data-paths configs/data_paths.yaml
Then explicitly evaluate the one rescue (cloud recommended): add --run-rescue.
No test data or test ownership is read by this evaluation command.
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from entity_resolution.candidate_evaluation import evaluate_candidates
from entity_resolution.candidate_fusion import (
    FusionSpec,
    _atomic_json,
    _canonical_fingerprint,
    _sha256,
    fuse_budgeted,
    query_quality,
    select_candidates,
    validate_and_load_evidence,
)
from entity_resolution.config import load_data_paths
from entity_resolution.exact_index import content_tie_key
from entity_resolution.folds import create_folds
from entity_resolution.ground_truth import load_ground_truth
from entity_resolution.metrics import evaluate_macro_f0_5_reference
from entity_resolution.missed_edge_analysis import TAXONOMY, classify_variations, hashlib_key
from entity_resolution.production_candidates import (
    EVIDENCE_COLUMNS,
    CachedRetriever,
    _protect_outputs,
    configuration_hash,
    read_batches,
)
from entity_resolution.sparse_retrieval import SparseRetrievalConfig, SparseTopNRetriever
from entity_resolution.views import normalize_field

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/retrieval_rescue.yaml"


def rescue_queries(queries: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    groups = {s1: group for s1, group in evidence.groupby("s1_id", sort=False)}
    selected = []
    for row in queries.itertuples(index=False):
        quality = query_quality(
            groups.get(row.entity_id, evidence.iloc[:0]), row.business_name, row.business_address
        )
        selected.append(quality["weak"] and not quality["missing_name"])
    return queries.loc[selected].reset_index(drop=True)


def retrieve_missing_address(
    retriever: CachedRetriever,
    queries: pd.DataFrame,
    evidence: pd.DataFrame,
    run_id: str,
    config: dict[str, Any],
):
    """Production rescue consumes only label-free gates and existing sparse blocks."""
    if not config.get("enabled", False):
        return pd.DataFrame(columns=EVIDENCE_COLUMNS), {"activated_queries": 0}
    if config.get("gate", "weak_query") != "weak_query":
        raise ValueError("unsupported rescue gate")
    k = int(config["top_k"])
    if not 0 < k <= 32767:
        raise ValueError("rescue top_k must fit int16")
    gated = rescue_queries(queries, evidence)
    rows = retriever.sparse_rows(
        gated, run_id, missing_address_only=True, top_k=k, threshold=float(config["threshold"])
    )
    return pd.DataFrame(rows, columns=EVIDENCE_COLUMNS), {
        "activated_queries": len(gated),
        "candidate_rows": len(rows),
    }


def _training_context(paths, sample, missed, cache: Path, batch_size: int):
    """One projected scan per source; retain only sampled queries/audit targets.

    Missing-address targets are streamed directly to Parquet for the single optional
    rescue. All paths originate in data_paths.yaml. Cache identities cover source bytes.
    """
    cache.mkdir(parents=True, exist_ok=True)
    inputs = {
        k: _sha256(paths[k])
        for k in ("train_source1", "train_source2", "train_source3", "train_ground_truth")
    }
    identity = {
        "inputs": inputs,
        "sample": configuration_hash(sorted(sample.s1_id)),
        "misses": configuration_hash(sorted(missed.target_id)),
        "version": 1,
    }
    manifest = cache / "context.json"
    if manifest.exists():
        state = json.loads(manifest.read_text())
        if state["identity"] != identity:
            raise ValueError("stale training context; use a fresh cache directory")
        for name, digest in state["files"].items():
            if _sha256(cache / name) != digest:
                raise ValueError("corrupt training context")
        return (
            pd.read_parquet(cache / "queries.parquet"),
            pd.read_parquet(cache / "audit_targets.parquet"),
            load_ground_truth(cache / "sample_truth.tsv"),
            state,
        )
    wanted = set(sample.s1_id)
    # Filter then invoke authoritative strict parser rather than retaining all labels in RAM.
    truth_path = cache / "sample_truth.tsv"
    with (
        paths["train_ground_truth"].open(newline="") as source,
        truth_path.open("w", newline="") as dest,
    ):
        reader, writer = csv.reader(source, delimiter="\t"), csv.writer(dest, delimiter="\t")
        writer.writerow(next(reader))
        for row in reader:
            if row[0] in wanted:
                writer.writerow(row)
    truth = load_ground_truth(truth_path)
    if set(truth) != wanted:
        raise ValueError("sample truth coverage mismatch")
    queries, targets = [], []
    audit_ids = set(missed.target_id)
    schema = pa.schema(
        [(c, pa.string()) for c in ("entity_id", "business_name", "business_address", "country")]
    )
    for number in (1, 2, 3):
        file = cache / f"S{number}_missing_address.parquet"
        writer = pq.ParquetWriter(file, schema) if number > 1 else None
        try:
            for frame in read_batches(paths[f"train_source{number}"], batch_size):
                if number == 1:
                    queries.append(frame[frame.entity_id.isin(wanted)])
                else:
                    targets.append(frame[frame.entity_id.isin(audit_ids)])
                    # Normalize only address to identify the label-free target subindex.
                    mask = [
                        not normalize_field(v).unicode_preserving for v in frame.business_address
                    ]
                    subset = frame.loc[mask]
                    if len(subset):
                        writer.write_table(
                            pa.Table.from_pandas(subset, schema=schema, preserve_index=False)
                        )
        finally:
            if writer:
                writer.close()
        print(f"cached training source {number}", flush=True)
    q = pd.concat(queries, ignore_index=True).sort_values("entity_id").reset_index(drop=True)
    t = pd.concat(targets, ignore_index=True).sort_values("entity_id").reset_index(drop=True)
    if set(q.entity_id) != wanted or set(t.entity_id) != audit_ids:
        raise ValueError("query/audit target coverage mismatch")
    q.to_parquet(cache / "queries.parquet", index=False)
    t.to_parquet(cache / "audit_targets.parquet", index=False)
    files = [
        "sample_truth.tsv",
        "queries.parquet",
        "audit_targets.parquet",
        "S2_missing_address.parquet",
        "S3_missing_address.parquet",
    ]
    state = {"identity": identity, "files": {name: _sha256(cache / name) for name in files}}
    _atomic_json(manifest, state)
    return q, t, truth, state


def _metrics(frame, truth, baseline, elapsed):
    truth = {s1: truth[s1] for s1 in sorted(truth)}
    result = evaluate_candidates(
        frame, truth, pd.DataFrame({"s1_id": list(truth)}), runtime_seconds=elapsed
    )
    pairs = set(frame[["s1_id", "target_id"]].itertuples(index=False, name=None))
    by_query = {s1: set(g.target_id) for s1, g in frame.groupby("s1_id", sort=False)}
    oracle = {s1: sorted(values & by_query.get(s1, set())) for s1, values in truth.items()}
    official = evaluate_macro_f0_5_reference({s1: sorted(v) for s1, v in truth.items()}, oracle)
    if not np.isclose(result["oracle_macro_f0_5"], official, rtol=0, atol=1e-12):
        raise ValueError("oracle disagrees with authoritative competition metric")
    actual = {(s1, target) for s1, targets in truth.items() for target in targets}
    gained = len((pairs - baseline) & actual)
    added = len(pairs - baseline)
    result.update(
        full_owner_queries=sum(v <= by_query.get(s1, set()) for s1, v in truth.items()),
        partial_owner_queries=sum(
            bool(v & by_query.get(s1, set())) and not v <= by_query.get(s1, set())
            for s1, v in truth.items()
        ),
        zero_owner_queries=sum(
            bool(v) and not v & by_query.get(s1, set()) for s1, v in truth.items()
        ),
        total_candidate_rows=len(frame),
        incremental_edges_recovered=gained,
        edges_lost_vs_baseline=len((baseline - pairs) & actual),
        incremental_candidates_added=added,
        net_candidate_change=len(pairs) - len(baseline),
        candidates_added_per_recovered_edge=added / gained if gained else None,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (2**20 if sys.platform == "darwin" else 1024),
        peak_memory_scope="whole evaluation process high-water mark",
    )
    result["source_recall"] = {}
    for source in ("S2", "S3"):
        true = {pair for pair in actual if pair[1].startswith(source + "-")}
        result["source_recall"][source] = {
            "covered": len(true & pairs),
            "total": len(true),
            "recall": len(true & pairs) / max(len(true), 1),
        }
    result.pop("groups", None)
    return result


def taxonomy(missed, queries, targets, recovered):
    q = queries.set_index("entity_id").to_dict("index")
    t = targets.set_index("entity_id").to_dict("index")
    rows = []
    for row in missed.itertuples(index=False):
        query = {"name": q[row.s1_id]["business_name"], "address": q[row.s1_id]["business_address"]}
        target = {
            "name": t[row.target_id]["business_name"],
            "address": t[row.target_id]["business_address"],
        }
        flags = classify_variations(
            query, target, present=not row.absent_from_every_existing_top80_channel
        )
        layer = (
            "insufficient_input"
            if flags["both_fields_sparse"] or flags["missing_or_null_name"]
            else "retrieval_channel"
            if flags["absent_every_channel"]
            else "fusion_truncation"
        )
        rows.append(
            {
                "s1_id": row.s1_id,
                "target_id": row.target_id,
                "layer": layer,
                "example": {
                    "content_key": hashlib_key(*query.values(), *target.values()),
                    "query_name_tokens": len(normalize_field(query["name"]).tokens),
                    "target_name_tokens": len(normalize_field(target["name"]).tokens),
                    "missing_address": flags["missing_address"],
                },
                **flags,
            }
        )
    result = {}
    for category in TAXONOMY:
        group = [r for r in rows if r[category]]
        pairs = {(r["s1_id"], r["target_id"]) for r in group}
        result[category] = {
            "count": len(group),
            "percentage": 100 * len(group) / max(len(rows), 1),
            "examples": [
                r["example"] for r in sorted(group, key=lambda r: r["example"]["content_key"])[:3]
            ],
            "existing_channel_present": sum(r["fusion_truncation"] for r in group),
            "recovered_by_configuration": {
                key: len(pairs & values) for key, values in recovered.items()
            },
        }
    return {
        "categories": result,
        "layers": pd.Series([r["layer"] for r in rows]).value_counts().to_dict(),
        "overlapping_flags": True,
        "missed_edges": len(rows),
    }


def run(config_path: Path, *, data_paths_config: Path | None = None, run_rescue: bool = False):
    config = yaml.safe_load(config_path.read_text())
    v2 = yaml.safe_load(Path(config["v2_config"]).read_text())
    paths = load_data_paths(data_paths_config or Path(config["data_paths_config"]))
    cache = Path(config["cache"])
    _protect_outputs(cache, paths)
    _protect_outputs(Path(config["output"]), paths)
    evidence, sample, validation = validate_and_load_evidence(
        Path(v2["artifact_root"]),
        sample_path=Path(v2["diagnostic_sample"]),
        v1_config_path=Path(v2["v1_config"]),
        expected_fingerprints=v2["expected_artifact_sha256"],
    )
    missed = pd.read_csv(v2["outputs"]["missed_edges"], sep="\t", keep_default_na=False)
    queries, targets, truth_list, context = _training_context(paths, sample, missed, cache, 50000)
    truth = {s1: set(v) for s1, v in truth_list.items()}
    folds, _ = create_folds(
        truth_list,
        countries=dict(zip(sample.s1_id, sample.country, strict=True)),
        n_splits=v2["folds"]["count"],
        random_state=v2["seed"],
    )
    stored = pd.read_parquet(config["folds"])
    if folds != dict(zip(stored.s1_id, stored.fold, strict=True)):
        raise ValueError("authoritative grouped folds disagree with persisted folds")
    held_ids = {s1 for s1, fold in folds.items() if fold in v2["folds"]["held_out"]}
    dev_ids = set(truth) - held_ids
    # Project cached aggregate; no target re-normalization or re-retrieval for C1/C2.
    cols = [
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
    aggregate_metadata = json.loads(Path(config["aggregate"]).with_suffix(".json").read_text())
    expected_aggregate = _canonical_fingerprint(
        {
            "artifacts": {
                key: value["sha256"] for key, value in validation.items() if key != "summary"
            },
            "sample": validation["summary"]["sample_sha256"],
            "rrf_k": v2["fusion"]["rrf_k"],
            "aggregation_version": 1,
        }
    )
    if aggregate_metadata["input_fingerprint"] != expected_aggregate:
        raise ValueError("stale aggregate configuration or source fingerprints")
    aggregate = pd.read_parquet(config["aggregate"], columns=cols)
    if len(aggregate) != validation["summary"].get(
        "distinct_pairs", len(evidence.drop_duplicates(["s1_id", "target_id"]))
    ):
        raise ValueError("cached aggregate coverage mismatch")
    evidence = evidence.merge(
        aggregate[["s1_id", "target_id", "target_content_key"]],
        on=["s1_id", "target_id"],
        validate="many_to_one",
    )
    if evidence.target_content_key.isna().any():
        raise ValueError("aggregate metadata incomplete")
    baseline_started = time.perf_counter()
    baseline = select_candidates(aggregate, FusionSpec("rrf", global_budget=40))
    baseline_elapsed = time.perf_counter() - baseline_started
    del aggregate
    baseline_pairs = {
        split: set(
            baseline.loc[baseline.s1_id.isin(ids), ["s1_id", "target_id"]].itertuples(
                index=False, name=None
            )
        )
        for split, ids in (("held_out", held_ids), ("development", dev_ids))
    }
    results, recovered, gates = {}, {}, {}
    audit_pairs = set(missed[["s1_id", "target_id"]].itertuples(index=False, name=None))

    def measure(name, selected, elapsed):
        results[name] = {}
        for split, ids in (("development", dev_ids), ("held_out", held_ids)):
            part = selected[selected.s1_id.isin(ids)]
            results[name][split] = _metrics(
                part, {s1: truth[s1] for s1 in ids}, baseline_pairs[split], elapsed
            )
        results[name]["held_out_by_fold"] = {}
        for fold in v2["folds"]["held_out"]:
            ids = {s1 for s1, value in folds.items() if value == fold}
            part = selected[selected.s1_id.isin(ids)]
            fold_baseline = {pair for pair in baseline_pairs["held_out"] if pair[0] in ids}
            results[name]["held_out_by_fold"][str(fold)] = _metrics(
                part, {s1: truth[s1] for s1 in ids}, fold_baseline, elapsed
            )
        recovered[name] = audit_pairs.intersection(
            selected[["s1_id", "target_id"]].itertuples(index=False, name=None)
        )
        print(
            name,
            json.dumps(
                {
                    k: results[name]["held_out"][k]
                    for k in ("oracle_macro_f0_5", "recalled_edges", "average_candidates")
                }
            ),
            flush=True,
        )

    measure("baseline_v2_k40", baseline, baseline_elapsed)
    del baseline
    for name, policy in config["policies"].items():
        started = time.perf_counter()
        pieces, activated = [], 0
        for start in range(0, len(queries), config["query_batch_size"]):
            query = queries.iloc[start : start + config["query_batch_size"]]
            frame, gating = fuse_budgeted(
                evidence[evidence.s1_id.isin(query.entity_id)],
                query,
                policy,
                run_id=config["run_id"],
            )
            pieces.append(frame[["s1_id", "target_id", "target_source"]])
            activated += gating["adaptive_queries"]
        selected = pd.concat(pieces, ignore_index=True)
        gates[name] = activated
        measure(name, selected, time.perf_counter() - started)
        del selected, pieces
    started = time.perf_counter()
    union = evidence[["s1_id", "target_id", "target_source"]].drop_duplicates()
    measure("top80_union_ceiling", union, time.perf_counter() - started)
    del union
    # Freeze choice using development metrics only, before observing held-out comparisons.
    selectable = list(config["policies"])

    def choice_key(name):
        m = results[name]["development"]
        return (
            -m["oracle_macro_f0_5"],
            -m["full_owner_queries"],
            -m["positive_edge_recall"],
            m["p95_candidates"],
            m["average_candidates"],
            m["runtime_seconds"],
            name,
        )

    selected_stretch = min(selectable, key=choice_key)
    rescue_status = "not run; synthetic tests only"
    rescue_input_hashes = {}
    rescue_cost = {}
    if run_rescue:
        # Gate 2 is now complete. One mechanism, one prespecified K/threshold.
        started = time.perf_counter()
        gated = rescue_queries(queries, evidence)
        extra_pieces = []
        for source in ("S2", "S3"):
            saved = joblib.load(Path(v2["artifact_root"]) / f"{source}_name_vectorizer.joblib")
            vectorizer = saved["vectorizer"] if isinstance(saved, dict) else saved
            rescue_input_hashes[source] = _sha256(
                Path(v2["artifact_root"]) / f"{source}_name_vectorizer.joblib"
            )
            v1 = yaml.safe_load(Path(v2["v1_config"]).read_text())
            if (
                tuple(vectorizer.ngram_range) != tuple(v1["tfidf"]["ngram_range"])
                or vectorizer.min_df != v1["tfidf"]["min_df"]
                or vectorizer.analyzer != "char_wb"
            ):
                raise ValueError("incompatible rescue vocabulary")
            spec = SparseRetrievalConfig(
                field="name",
                top_k=config["rescue"]["top_k"],
                threshold=config["rescue"]["threshold"],
                n_threads=int(config["threads"]),
            )
            retriever = SparseTopNRetriever(spec, vectorizer)
            subset = cache / f"{source}_missing_address.parquet"

            def factory(subset=subset):
                for batch in pq.ParquetFile(subset).iter_batches(batch_size=20000):
                    yield batch.to_pandas()

            extra, _ = retriever.retrieve(
                gated,
                factory,
                source=source,
                run_id=config["run_id"],
                query_batch_size=config["query_batch_size"],
            )
            extra["channel"] = "missing_address_name_char"
            ties = {}
            wanted = set(extra.target_id)
            for batch in factory():
                for row in batch[batch.entity_id.isin(wanted)].itertuples(index=False):
                    ties[row.entity_id] = content_tie_key(
                        row.business_name, row.business_address, row.country
                    )
            extra["target_content_key"] = extra.target_id.map(ties)
            extra_pieces.append(extra)
        extra = pd.concat(extra_pieces, ignore_index=True)
        extra.to_parquet(cache / "rescue_evidence.parquet", index=False)
        extended = pd.concat([evidence, extra], ignore_index=True)
        # Add rescue alongside the existing union to measure truly new retrieval coverage.
        measure(
            "rescue_union_ceiling",
            extended[["s1_id", "target_id", "target_source"]].drop_duplicates(),
            time.perf_counter() - started,
        )
        selected, gating = fuse_budgeted(
            extended, queries, config["policies"][selected_stretch], run_id=config["run_id"]
        )
        measure("rescue_stretch", selected, time.perf_counter() - started)
        gates["missing_address_name_char"] = len(gated)
        base_dev = results[selected_stretch]["development"]
        rescued_dev = results["rescue_stretch"]["development"]
        gained = rescued_dev["recalled_edges"] - base_dev["recalled_edges"]
        base_pairs = set(
            evidence.loc[evidence.s1_id.isin(dev_ids), ["s1_id", "target_id"]].itertuples(
                index=False, name=None
            )
        )
        added_pairs = (
            set(
                extra.loc[extra.s1_id.isin(dev_ids), ["s1_id", "target_id"]].itertuples(
                    index=False, name=None
                )
            )
            - base_pairs
        )
        added_truth = sum(target in truth[s1] for s1, target in added_pairs)
        cost = len(added_pairs) / added_truth if added_truth else None
        rescue_cost = {
            "development_new_channel_pairs": len(added_pairs),
            "development_new_positive_edges": added_truth,
            "candidates_per_new_channel_positive": cost,
            "maximum_accepted_cost": config["maximum_rescue_cost"],
        }
        acceptable = cost is not None and cost <= config["maximum_rescue_cost"]
        rescue_status = (
            "retain"
            if gained > 0
            and rescued_dev["oracle_macro_f0_5"] >= base_dev["oracle_macro_f0_5"]
            and acceptable
            else "stop: no development gain or excessive candidate cost"
        )
    audit = taxonomy(missed, queries, targets, recovered)
    payload = {
        "results": results,
        "gating_counts": gates,
        "selected_stretch": selected_stretch,
        "selection_split": (
            "development only; held-out is diagnostic, not untouched final validation"
        ),
        "rescue_status": rescue_status,
        "rescue_vectorizer_sha256": rescue_input_hashes,
        "rescue_cost": rescue_cost,
        "rescue_evidence_sha256": _sha256(cache / "rescue_evidence.parquet")
        if run_rescue
        else None,
        "taxonomy": audit,
        "context": context,
        "validation": validation,
        "configuration": config,
        "aggregate_sha256": _sha256(Path(config["aggregate"])),
        "folds_sha256": _sha256(Path(config["folds"])),
    }
    _atomic_json(Path(config["output"]), payload)
    return payload


def rebudget_cached(
    config_path: Path, budget: int, *, data_paths_config: Path | None = None
) -> dict[str, Any]:
    """One cheap budget repair using validated, already generated training evidence."""
    config = yaml.safe_load(config_path.read_text())
    paths = load_data_paths(data_paths_config or Path(config["data_paths_config"]))
    _protect_outputs(Path(config["output"]), paths)
    payload = json.loads(Path(config["output"]).read_text())
    cache = Path(config["cache"])
    context = payload["context"]
    for name, digest in context["files"].items():
        if _sha256(cache / name) != digest:
            raise ValueError("corrupt cached training context")
    if _sha256(cache / "rescue_evidence.parquet") != payload["rescue_evidence_sha256"]:
        raise ValueError("corrupt cached rescue evidence")
    v2 = yaml.safe_load(Path(config["v2_config"]).read_text())
    evidence, sample, validation = validate_and_load_evidence(
        Path(v2["artifact_root"]),
        sample_path=Path(v2["diagnostic_sample"]),
        v1_config_path=Path(v2["v1_config"]),
        expected_fingerprints=v2["expected_artifact_sha256"],
    )
    if validation != payload["validation"]:
        raise ValueError("changed diagnostic evidence")
    if _sha256(Path(config["aggregate"])) != payload["aggregate_sha256"]:
        raise ValueError("corrupt aggregate")
    meta = pd.read_parquet(
        config["aggregate"], columns=["s1_id", "target_id", "target_content_key"]
    )
    evidence = evidence.merge(meta, on=["s1_id", "target_id"], validate="many_to_one")
    extra = pd.read_parquet(cache / "rescue_evidence.parquet")
    extended = pd.concat([evidence, extra], ignore_index=True)
    queries = pd.read_parquet(cache / "queries.parquet")
    truth_list = load_ground_truth(cache / "sample_truth.tsv")
    truth = {s1: set(values) for s1, values in truth_list.items()}
    folds, _ = create_folds(
        truth_list,
        countries=dict(zip(sample.s1_id, sample.country, strict=True)),
        n_splits=v2["folds"]["count"],
        random_state=v2["seed"],
    )
    if _sha256(Path(config["folds"])) != payload["folds_sha256"]:
        raise ValueError("changed grouped folds")
    held = {s1 for s1, fold in folds.items() if fold in v2["folds"]["held_out"]}
    policy = {**config["policies"]["stretch_adaptive"], "expanded_k": budget}
    # Recreate the reliable reference on the same evidence; no corpus scan.
    base_pieces, pieces = [], []
    activated = 0
    started = time.perf_counter()
    for start in range(0, len(queries), config["query_batch_size"]):
        query = queries.iloc[start : start + config["query_batch_size"]]
        baseline, _ = fuse_budgeted(
            evidence[evidence.s1_id.isin(query.entity_id)],
            query,
            config["policies"]["fixed_k40"],
            run_id=config["run_id"],
        )
        selected, gating = fuse_budgeted(
            extended[extended.s1_id.isin(query.entity_id)], query, policy, run_id=config["run_id"]
        )
        base_pieces.append(baseline[["s1_id", "target_id", "target_source"]])
        pieces.append(selected[["s1_id", "target_id", "target_source"]])
        activated += gating["adaptive_queries"]
    elapsed = time.perf_counter() - started
    baseline = pd.concat(base_pieces, ignore_index=True)
    selected = pd.concat(pieces, ignore_index=True)
    name = f"rescue_adaptive_k{budget}"
    results = {}
    for split, ids in (("development", set(truth) - held), ("held_out", held)):
        base_pairs = set(
            baseline.loc[baseline.s1_id.isin(ids), ["s1_id", "target_id"]].itertuples(
                index=False, name=None
            )
        )
        results[split] = _metrics(
            selected[selected.s1_id.isin(ids)], {s1: truth[s1] for s1 in ids}, base_pairs, elapsed
        )
    results["held_out_by_fold"] = {}
    for fold in v2["folds"]["held_out"]:
        ids = {s1 for s1, value in folds.items() if value == fold}
        base_pairs = set(
            baseline.loc[baseline.s1_id.isin(ids), ["s1_id", "target_id"]].itertuples(
                index=False, name=None
            )
        )
        results["held_out_by_fold"][str(fold)] = _metrics(
            selected[selected.s1_id.isin(ids)], {s1: truth[s1] for s1 in ids}, base_pairs, elapsed
        )
    missed = pd.read_csv(v2["outputs"]["missed_edges"], sep="\t", keep_default_na=False)
    audit_pairs = set(missed[["s1_id", "target_id"]].itertuples(index=False, name=None))
    recovered = audit_pairs.intersection(
        selected[["s1_id", "target_id"]].itertuples(index=False, name=None)
    )
    audit = taxonomy(
        missed, queries, pd.read_parquet(cache / "audit_targets.parquet"), {name: recovered}
    )
    for category, row in audit["categories"].items():
        payload["taxonomy"]["categories"][category]["recovered_by_configuration"].update(
            row["recovered_by_configuration"]
        )
    payload["results"][name] = results
    payload["gating_counts"][name] = activated
    payload["rebudget"] = {
        "policy": policy,
        "no_new_retrieval": True,
        "runtime_includes_reference_refusion": True,
    }

    def key(candidate):
        m = payload["results"][candidate]["development"]
        return (
            -m["oracle_macro_f0_5"],
            -m["full_owner_queries"],
            -m["positive_edge_recall"],
            m["p95_candidates"],
            m["average_candidates"],
            m["runtime_seconds"],
        )

    payload["selected_stretch"] = min(("rescue_stretch", name), key=key)
    _atomic_json(Path(config["output"]), payload)
    print(name, json.dumps(results["held_out"]), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-paths", type=Path)
    parser.add_argument("--run-rescue", action="store_true")
    parser.add_argument(
        "--rebudget", type=int, help="Re-fuse saved rescue evidence; no new retrieval"
    )
    args = parser.parse_args()
    if args.rebudget is not None:
        rebudget_cached(args.config, args.rebudget, data_paths_config=args.data_paths)
    else:
        run(args.config, data_paths_config=args.data_paths, run_rescue=args.run_rescue)


if __name__ == "__main__":
    main()
