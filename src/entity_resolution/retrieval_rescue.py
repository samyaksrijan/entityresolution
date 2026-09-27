"""Artifact-first budget expansion and a bounded missing-address name rescue.

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
    # Retain only positive predictions after the authoritative evaluator returns.
    # A second Python set containing millions of negative pairs is unnecessary.
    by_query: dict[str, set[str]] = {}
    retained_baseline = 0
    for s1, target in frame[["s1_id", "target_id"]].itertuples(index=False, name=None):
        retained_baseline += (s1, target) in baseline
        true_targets = truth.get(s1)
        if true_targets is not None and target in true_targets:
            by_query.setdefault(s1, set()).add(target)
    pairs = {(s1, target) for s1, values in by_query.items() for target in values}
    oracle = {s1: sorted(values & by_query.get(s1, set())) for s1, values in truth.items()}
    official = evaluate_macro_f0_5_reference({s1: sorted(v) for s1, v in truth.items()}, oracle)
    if not np.isclose(result["oracle_macro_f0_5"], official, rtol=0, atol=1e-12):
        raise ValueError("oracle disagrees with authoritative competition metric")
    actual = {(s1, target) for s1, targets in truth.items() for target in targets}
    gained = len((pairs - baseline) & actual)
    added = len(frame) - retained_baseline
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
        net_candidate_change=len(frame) - len(baseline),
        candidates_added_per_recovered_edge=added / gained if gained else None,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (2**20 if sys.platform == "darwin" else 1024),
        peak_memory_scope="whole evaluation process high-water mark",
    )
    result["source_recall"] = {}
    for source in ("S2", "S3"):
        true = {pair for pair in actual if pair[1].startswith(source + "-")}
        source_truth = {
            s1: sorted(t for t in targets if t.startswith(source + "-"))
            for s1, targets in truth.items()
        }
        source_oracle = {
            s1: sorted(set(targets) & by_query.get(s1, set()))
            for s1, targets in source_truth.items()
        }
        result["source_recall"][source] = {
            "covered": len(true & pairs),
            "total": len(true),
            "recall": len(true & pairs) / max(len(true), 1),
            "oracle_macro_f0_5": evaluate_macro_f0_5_reference(source_truth, source_oracle),
            "full_owner_queries": sum(
                set(v) <= by_query.get(s1, set()) for s1, v in source_truth.items()
            ),
            "candidate_rows": int(frame.target_source.eq(source).sum()),
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
            "selection_rule": (
                "report candidate cost; enforce feasibility through bounded generation"
            ),
        }
        rescue_status = (
            "retain"
            if gained > 0
            and rescued_dev["oracle_macro_f0_5"] >= base_dev["oracle_macro_f0_5"]
            else "stop: no development oracle/coverage gain"
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


def _subset_index(cache: Path, asset_root: Path, *, threads: int) -> CachedRetriever:
    """Cache the label-free missing-address corpus in the production sparse format.

    The same sparse_rows method runs in diagnostics and full production, including
    content-based cutoff ties. No audit IDs or labels influence this index.
    """
    from scipy import sparse

    root = cache / "missing_address_index"
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "subsets": {s: _sha256(cache / f"{s}_missing_address.parquet") for s in ("S2", "S3")},
        "vocabulary": {
            s: _sha256(asset_root / f"{s}_name_vectorizer.joblib") for s in ("S2", "S3")
        },
        "version": 1,
    }
    config = {
        "threads": threads,
        "max_evidence_rows": 2000000,
        "sparse": {"top_k": 80, "threshold": 0.2},
    }
    retriever = CachedRetriever(root, config)
    if retriever.manifest.exists():
        state = json.loads(retriever.manifest.read_text())
        if state["identity"] != identity:
            raise ValueError("incompatible missing-address index")
        for name, digest in state["files"].items():
            if _sha256(root / name) != digest:
                raise ValueError("corrupt missing-address index")
        return retriever
    blocks, files = [], {}
    for source in ("S2", "S3"):
        saved = joblib.load(asset_root / f"{source}_name_vectorizer.joblib")
        vectorizer = saved["vectorizer"]
        name = f"{source}_name.joblib"
        joblib.dump(vectorizer, root / name)
        files[name] = _sha256(root / name)
        targets = pd.read_parquet(cache / f"{source}_missing_address.parquet")
        targets["name"] = [normalize_field(v).unicode_preserving for v in targets.business_name]
        targets["tie"] = [
            content_tie_key(r.business_name, r.business_address, r.country)
            for r in targets.itertuples(index=False)
        ]
        for country, group in targets.groupby("country", sort=True):
            group = group.sort_values("tie", kind="stable")
            for start in range(0, len(group), 20000):
                part = group.iloc[start : start + 20000]
                number = len(blocks)
                meta_name, matrix_name = (
                    f"block-{number:06d}.parquet",
                    f"block-{number:06d}-name.npz",
                )
                metadata = part[["entity_id", "tie"]].rename(columns={"entity_id": "id"})
                metadata.assign(address_missing=True).to_parquet(root / meta_name, index=False)
                sparse.save_npz(
                    root / matrix_name, vectorizer.transform(part.name).astype(np.float32)
                )
                blocks.append(
                    {
                        "source": source,
                        "country": country,
                        "metadata": meta_name,
                        "name": matrix_name,
                        "number": number,
                    }
                )
                for name in (meta_name, matrix_name):
                    files[name] = _sha256(root / name)
        print(f"cached production-format rescue index {source}", flush=True)
    _atomic_json(retriever.manifest, {"identity": identity, "files": files, "blocks": blocks})
    return retriever


def grouped_measurements(selected, truth, baseline, folds, held_folds, elapsed):
    """Official pooled and fold metrics; source metrics keep empty-owner queries."""
    ids_all = set(truth)
    held = {s1 for s1, fold in folds.items() if fold in held_folds}
    groups = {"all": ids_all, "development": ids_all - held, "held_out": held}
    groups.update(
        {
            f"fold_{fold}": {s1 for s1, f in folds.items() if f == fold}
            for fold in sorted(set(folds.values()))
        }
    )
    result = {}
    for name, ids in groups.items():
        result[name] = _metrics(
            selected[selected.s1_id.isin(ids)],
            {s1: truth[s1] for s1 in ids},
            {pair for pair in baseline if pair[0] in ids},
            elapsed,
        )
    scores = [result[f"fold_{fold}"]["oracle_macro_f0_5"] for fold in sorted(set(folds.values()))]
    result["fold_summary"] = {
        "mean_oracle": float(np.mean(scores)),
        "minimum_oracle": min(scores),
        "standard_deviation": float(np.std(scores)),
        "folds": len(scores),
    }
    return result


def maximize_cached(
    config_path: Path,
    *,
    data_paths_config: Path | None = None,
    rescue_budget: int = 512,
    threshold: float = 0.18,
):
    """Prespecified artifact reuse plus one ranked-tail missing-address experiment."""
    config = yaml.safe_load(config_path.read_text())
    paths = load_data_paths(data_paths_config or Path(config["data_paths_config"]))
    cache = Path(config["cache"])
    _protect_outputs(cache, paths)
    previous = json.loads(Path(config["output"]).read_text())
    context = previous["context"]
    for name, digest in context["files"].items():
        if _sha256(cache / name) != digest:
            raise ValueError("corrupt cached training context")
    for key, digest in context["identity"]["inputs"].items():
        if _sha256(paths[key]) != digest:
            raise ValueError("changed raw training input")
    if _sha256(cache / "rescue_evidence.parquet") != previous["rescue_evidence_sha256"]:
        raise ValueError("corrupt cached rescue evidence")
    v2 = yaml.safe_load(Path(config["v2_config"]).read_text())
    evidence, sample, validation = validate_and_load_evidence(
        Path(v2["artifact_root"]),
        sample_path=Path(v2["diagnostic_sample"]),
        v1_config_path=Path(v2["v1_config"]),
        expected_fingerprints=v2["expected_artifact_sha256"],
    )
    if validation != previous["validation"]:
        raise ValueError("changed diagnostic evidence")
    if _sha256(Path(config["aggregate"])) != previous["aggregate_sha256"]:
        raise ValueError("corrupt aggregate")
    meta = pd.read_parquet(
        config["aggregate"], columns=["s1_id", "target_id", "target_content_key"]
    )
    evidence = evidence.merge(meta, on=["s1_id", "target_id"], validate="many_to_one")
    del meta
    queries = pd.read_parquet(cache / "queries.parquet")
    truth_list = load_ground_truth(cache / "sample_truth.tsv")
    truth = {s1: set(values) for s1, values in truth_list.items()}
    folds, _ = create_folds(
        truth_list,
        countries=dict(zip(sample.s1_id, sample.country, strict=True)),
        n_splits=v2["folds"]["count"],
        random_state=v2["seed"],
    )
    stored = pd.read_parquet(config["folds"])
    if _sha256(Path(config["folds"])) != previous["folds_sha256"] or folds != dict(
        zip(stored.s1_id, stored.fold, strict=True)
    ):
        raise ValueError("changed grouped folds")
    baseline, _ = fuse_budgeted(evidence, queries, {"top_k": 40}, run_id="score_reference")
    pair_columns = ["s1_id", "target_id", "target_source"]
    baseline = baseline[pair_columns]
    base_pairs = set(baseline[["s1_id", "target_id"]].itertuples(index=False, name=None))
    result_path = cache / f"score_evaluation_k{rescue_budget}.json"
    report = {
        "results": {},
        "selection": (
            "highest mean official oracle over existing grouped folds; "
            "diagnostic folds reused, not untouched validation"
        ),
        "configuration": {"rescue_budget": rescue_budget, "threshold": threshold},
        "inputs": {
            "context": context,
            "validation": validation,
            "aggregate_sha256": previous["aggregate_sha256"],
            "folds_sha256": previous["folds_sha256"],
        },
    }

    def measure(name, selected, elapsed):
        result = grouped_measurements(
            selected, truth, base_pairs, folds, v2["folds"]["held_out"], elapsed
        )
        report["results"][name] = result
        _atomic_json(result_path, report)
        print(
            name,
            json.dumps(
                {
                    "mean_oracle": result["fold_summary"]["mean_oracle"],
                    "held_oracle": result["held_out"]["oracle_macro_f0_5"],
                    "mean_candidates": result["all"]["average_candidates"],
                    "covered": result["all"]["recalled_edges"],
                }
            ),
            flush=True,
        )

    measure("safe_k40", baseline, 0)
    del baseline
    for name, budget in (("emergency_k20", 20), ("fixed_k80", 80)):
        started = time.perf_counter()
        selected, _ = fuse_budgeted(evidence, queries, {"top_k": budget}, run_id=name)
        measure(name, selected[pair_columns], time.perf_counter() - started)
        del selected
    base_union = evidence[pair_columns].drop_duplicates()
    measure("original_top80_union", base_union, 0)
    old_extra = pd.read_parquet(cache / "rescue_evidence.parquet")
    measure(
        "rescue_k20_union", pd.concat([base_union, old_extra[pair_columns]]).drop_duplicates(), 0
    )
    started = time.perf_counter()
    pieces = []
    extended = pd.concat([evidence, old_extra], ignore_index=True)
    for start in range(0, len(queries), config["query_batch_size"]):
        query = queries.iloc[start : start + config["query_batch_size"]]
        selected, _ = fuse_budgeted(
            extended[extended.s1_id.isin(query.entity_id)],
            query,
            {
                "top_k": 40,
                "expanded_k": 400,
                "adaptive": True,
                "source_quotas": {"S2": 20, "S3": 20},
            },
            run_id="legacy_adaptive_k400",
        )
        pieces.append(selected[pair_columns])
    measure(
        "legacy_adaptive_k400", pd.concat(pieces, ignore_index=True), time.perf_counter() - started
    )
    del pieces, extended, selected, old_extra
    gated = rescue_queries(queries, evidence)
    report["activated_queries"] = len(gated)
    started = time.perf_counter()
    retriever = _subset_index(cache, Path(v2["artifact_root"]), threads=config["threads"])
    report["index_seconds"] = time.perf_counter() - started
    extra_path = cache / f"rescue_k{rescue_budget}_t{threshold}.parquet"
    identity = {
        "index_sha256": _sha256(retriever.manifest),
        "query_sha256": context["files"]["queries.parquet"],
        "gated_ids": configuration_hash(sorted(gated.entity_id)),
        "top_k": rescue_budget,
        "threshold": threshold,
        "implementation_sha256": _sha256(Path(__file__).with_name("production_candidates.py")),
    }
    sidecar = extra_path.with_suffix(".json")
    if sidecar.exists():
        info = json.loads(sidecar.read_text())
        if info["identity"] != identity or _sha256(extra_path) != info["sha256"]:
            raise ValueError("incompatible/corrupt ranked rescue evidence")
        retrieval_seconds = info["retrieval_seconds"]
    else:
        started = time.perf_counter()
        writer = None
        try:
            for start in range(0, len(gated), config["query_batch_size"]):
                query = gated.iloc[start : start + config["query_batch_size"]]
                extra = pd.DataFrame(
                    retriever.sparse_rows(
                        query,
                        "ranked_rescue",
                        missing_address_only=True,
                        top_k=rescue_budget,
                        threshold=threshold,
                    ),
                    columns=EVIDENCE_COLUMNS,
                )
                extra = extra.astype(
                    {"retrieval_score": "float32", "channel_rank": "int16", "exact_match": "bool"}
                )
                table = pa.Table.from_pandas(extra, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        extra_path.with_suffix(".tmp"), table.schema, compression="zstd"
                    )
                writer.write_table(table)
                print(
                    f"ranked rescue queries {min(start + len(query), len(gated))}/{len(gated)}",
                    flush=True,
                )
        finally:
            if writer:
                writer.close()
        extra_path.with_suffix(".tmp").replace(extra_path)
        retrieval_seconds = time.perf_counter() - started
        _atomic_json(
            sidecar,
            {
                "identity": identity,
                "sha256": _sha256(extra_path),
                "retrieval_seconds": retrieval_seconds,
            },
        )
    report["rescue_evidence"] = {
        "path": str(extra_path),
        "sha256": _sha256(extra_path),
        "retrieval_seconds": retrieval_seconds,
        "identity": identity,
    }
    extra = pd.read_parquet(extra_path, columns=pair_columns + ["channel_rank", "retrieval_score"])
    for name, budget, minimum in (
        ("rescue_k80_union", 80, 0.2),
        (f"rescue_k{rescue_budget}_union", rescue_budget, 0.2),
        (f"rescue_k{rescue_budget}_lower_threshold_union", rescue_budget, threshold),
    ):
        part = extra[(extra.channel_rank <= budget) & (extra.retrieval_score >= minimum)]
        selected = pd.concat([base_union, part[pair_columns]], ignore_index=True).drop_duplicates()
        measure(name, selected, retrieval_seconds)
        del selected, part

    def key(name):
        item = report["results"][name]
        return (
            -item["fold_summary"]["mean_oracle"],
            -item["fold_summary"]["minimum_oracle"],
            item["fold_summary"]["standard_deviation"],
            -item["all"]["full_owner_queries"],
            -item["all"]["positive_edge_recall"],
            item["all"]["average_candidates"],
        )

    report["selected_max_score"] = min(report["results"], key=key)
    _atomic_json(result_path, report)
    return report


def expand_name_channel(config_path: Path, *, data_paths_config: Path | None = None, budget=512):
    """One full-corpus name-budget extension using the production sparse backend.

    The bounded cache contains no labels. Unlike the initial artifacts, its ties
    are resolved before truncation with exactly the production implementation.
    """
    from scipy import sparse

    config = yaml.safe_load(config_path.read_text())
    paths = load_data_paths(data_paths_config or Path(config["data_paths_config"]))
    cache = Path(config["cache"])
    _protect_outputs(cache, paths)
    context = json.loads((cache / "context.json").read_text())
    for name, digest in context["files"].items():
        if _sha256(cache / name) != digest:
            raise ValueError("corrupt training context")
    for key, digest in context["identity"]["inputs"].items():
        if _sha256(paths[key]) != digest:
            raise ValueError("changed raw training source")
    v2 = yaml.safe_load(Path(config["v2_config"]).read_text())
    assets = Path(v2["artifact_root"])
    root = cache / "full_name_index"
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "inputs": {s: context["identity"]["inputs"][f"train_source{s[-1]}"] for s in ("S2", "S3")},
        "vocabulary": {s: _sha256(assets / f"{s}_name_vectorizer.joblib") for s in ("S2", "S3")},
        "version": 1,
    }
    retriever = CachedRetriever(
        root,
        {
            "threads": config["threads"],
            "max_evidence_rows": 2000000,
            "sparse": {"top_k": budget, "threshold": 0.2},
        },
    )
    started = time.perf_counter()
    if retriever.manifest.exists():
        state = json.loads(retriever.manifest.read_text())
        if state["identity"] != identity:
            raise ValueError("incompatible full name index")
        for name, digest in state["files"].items():
            if _sha256(root / name) != digest:
                raise ValueError("corrupt full name index")
    else:
        blocks, files = [], {}
        for source in ("S2", "S3"):
            vectorizer = joblib.load(assets / f"{source}_name_vectorizer.joblib")["vectorizer"]
            name = f"{source}_name.joblib"
            joblib.dump(vectorizer, root / name)
            files[name] = _sha256(root / name)
            # No address channel is needed by this name-only diagnostic cache.
            joblib.dump(None, root / f"{source}_address.joblib")
            files[f"{source}_address.joblib"] = _sha256(root / f"{source}_address.joblib")
            count = 0
            for targets in read_batches(paths[f"train_source{source[-1]}"], 50000):
                targets["name"] = [
                    normalize_field(v).unicode_preserving for v in targets.business_name
                ]
                targets["tie"] = [
                    content_tie_key(r.business_name, r.business_address, r.country)
                    for r in targets.itertuples(index=False)
                ]
                for country, group in targets.groupby("country", sort=True):
                    group = group.sort_values("tie", kind="stable")
                    for start in range(0, len(group), 20000):
                        part = group.iloc[start : start + 20000]
                        number = len(blocks)
                        meta_name = f"block-{number:06d}.parquet"
                        matrix_name = f"block-{number:06d}-name.npz"
                        metadata = part[["entity_id", "tie"]].rename(columns={"entity_id": "id"})
                        metadata.assign(address_missing=False).to_parquet(
                            root / meta_name, index=False
                        )
                        sparse.save_npz(
                            root / matrix_name, vectorizer.transform(part.name).astype(np.float32)
                        )
                        blocks.append(
                            {
                                "source": source,
                                "country": country,
                                "metadata": meta_name,
                                "name": matrix_name,
                                "number": number,
                            }
                        )
                        for name in (meta_name, matrix_name):
                            files[name] = _sha256(root / name)
                count += len(targets)
                if count % 500000 == 0:
                    print(f"full name index {source}: {count} targets", flush=True)
            print(f"full name index {source}: complete {count}", flush=True)
        _atomic_json(retriever.manifest, {"identity": identity, "files": files, "blocks": blocks})
    index_seconds = time.perf_counter() - started
    queries = pd.read_parquet(cache / "queries.parquet")
    output = cache / f"full_name_k{budget}.parquet"
    sidecar = output.with_suffix(".json")
    run_identity = {
        "index_sha256": _sha256(retriever.manifest),
        "budget": budget,
        "query_sha256": context["files"]["queries.parquet"],
        "implementation_sha256": _sha256(Path(__file__).with_name("production_candidates.py")),
    }
    if sidecar.exists():
        previous = json.loads(sidecar.read_text())
        if previous["identity"] != run_identity or _sha256(output) != previous["sha256"]:
            raise ValueError("incompatible/corrupt full name evidence")
        return previous
    started = time.perf_counter()
    writer = None
    try:
        for start in range(0, len(queries), config["query_batch_size"]):
            query = queries.iloc[start : start + config["query_batch_size"]]
            # This cache contains only name matrices; the absent address vectorizers
            # are skipped by the same sparse backend used in production.
            rows = retriever.sparse_rows(
                query,
                "full_name_expansion",
                missing_address_only=False,
                top_k=budget,
                threshold=0.2,
            )
            extra = pd.DataFrame(rows, columns=EVIDENCE_COLUMNS)
            extra = extra.astype(
                {"retrieval_score": "float32", "channel_rank": "int16", "exact_match": "bool"}
            )
            table = pa.Table.from_pandas(extra, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    output.with_suffix(".tmp"), table.schema, compression="zstd"
                )
            writer.write_table(table)
            print(f"full name retrieval {start + len(query)}/{len(queries)}", flush=True)
    finally:
        if writer:
            writer.close()
    output.with_suffix(".tmp").replace(output)
    result = {
        "identity": run_identity,
        "sha256": _sha256(output),
        "index_seconds": index_seconds,
        "retrieval_seconds": time.perf_counter() - started,
    }
    _atomic_json(sidecar, result)
    return result


def evaluate_name_expansion(
    config_path: Path,
    *,
    budget=512,
    rescue_budget=512,
    address_budget: int | None = None,
    data_paths_config: Path | None = None,
):
    """Measure a small set of channel-tail budgets with bounded candidate memory."""
    config = yaml.safe_load(config_path.read_text())
    cache = Path(config["cache"])
    _protect_outputs(cache, load_data_paths(data_paths_config or Path(config["data_paths_config"])))
    report = json.loads((cache / f"score_evaluation_k{rescue_budget}.json").read_text())
    context = report["inputs"]["context"]
    for name, digest in context["files"].items():
        if _sha256(cache / name) != digest:
            raise ValueError("corrupt training context")
    rescue_path = Path(report["rescue_evidence"]["path"])
    artifact_paths = {"name": cache / f"full_name_k{budget}.parquet", "rescue": rescue_path}
    if address_budget is not None:
        artifact_paths["address"] = cache / f"full_address_k{address_budget}.parquet"
    infos = {}
    for label, path in artifact_paths.items():
        info = json.loads(path.with_suffix(".json").read_text())
        if _sha256(path) != info["sha256"]:
            raise ValueError(f"corrupt {label} evidence")
        if info["identity"]["query_sha256"] != context["files"]["queries.parquet"]:
            raise ValueError(f"wrong {label} query sample")
        infos[label] = {"path": str(path), **info}
    if _sha256(Path(config["aggregate"])) != report["inputs"]["aggregate_sha256"]:
        raise ValueError("corrupt aggregate")
    if _sha256(Path(config["folds"])) != report["inputs"]["folds_sha256"]:
        raise ValueError("changed grouped folds")
    columns = ["s1_id", "target_id", "target_source"]
    aggregate = pd.read_parquet(
        config["aggregate"],
        columns=columns
        + [
            "target_content_key",
            "exact_name",
            "exact_address",
            "supporting_channels",
            "rrf_name",
            "rrf_address",
            "rrf_exact_name",
            "rrf_exact_address",
            "rrf_total",
        ],
    )
    reference = select_candidates(aggregate, FusionSpec("rrf", global_budget=40))
    baseline = set(reference[["s1_id", "target_id"]].itertuples(index=False, name=None))
    original = aggregate[columns]
    del aggregate, reference
    truth = {s1: set(v) for s1, v in load_ground_truth(cache / "sample_truth.tsv").items()}
    stored = pd.read_parquet(config["folds"])
    folds = dict(zip(stored.s1_id, stored.fold, strict=True))
    v2 = yaml.safe_load(Path(config["v2_config"]).read_text())
    report["expanded_evidence"] = infos
    report["incremental_reference"] = "safe_k40"
    variants = [(min(160, budget), None), (budget, None)]
    if address_budget is not None:
        variants = [(budget, address_budget)]
    for name_k, address_k in dict.fromkeys(variants):
        key = f"full_name_k{name_k}_rescue_k{rescue_budget}"
        if address_k is not None:
            key += f"_address_k{address_k}"
        root = cache / key
        root.mkdir(parents=True, exist_ok=True)
        paths = []
        ids = sorted(truth)
        for start in range(0, len(ids), 250):
            batch_ids = ids[start : start + 250]
            pieces = [original[original.s1_id.isin(batch_ids)]]
            for label, path in artifact_paths.items():
                if label == "address" and address_k is None:
                    continue
                filters = [("s1_id", "in", batch_ids)]
                if label == "name":
                    filters.append(("channel_rank", "<=", name_k))
                if label == "address":
                    filters.append(("channel_rank", "<=", address_k))
                pieces.append(pd.read_parquet(path, columns=columns, filters=filters))
            selected = pd.concat(pieces, ignore_index=True).drop_duplicates()
            path = root / f"part-{start // 250:06d}.parquet"
            selected.to_parquet(path, index=False, compression="zstd")
            paths.append(path)
        elapsed = sum(info["retrieval_seconds"] for info in infos.values())
        item, _ = measure_candidate_shards(
            paths, truth, baseline, folds, v2["folds"]["held_out"], elapsed
        )
        report["results"][key] = item
        ref = report["results"][f"rescue_k{rescue_budget}_lower_threshold_union"]
        item["gain_vs_rescue_union"] = {
            "folds": {
                str(f): item[f"fold_{f}"]["oracle_macro_f0_5"]
                - ref[f"fold_{f}"]["oracle_macro_f0_5"]
                for f in sorted(set(folds.values()))
            },
            "sources": {
                s: item["all"]["source_recall"][s]["oracle_macro_f0_5"]
                - ref["all"]["source_recall"][s]["oracle_macro_f0_5"]
                for s in ("S2", "S3")
            },
        }
        print(
            key,
            json.dumps(
                {
                    "mean_oracle": item["fold_summary"]["mean_oracle"],
                    "held_oracle": item["held_out"]["oracle_macro_f0_5"],
                    "mean_candidates": item["all"]["average_candidates"],
                    "covered": item["all"]["recalled_edges"],
                }
            ),
            flush=True,
        )

    def key(name):
        item = report["results"][name]
        return (
            -item["fold_summary"]["mean_oracle"],
            -item["fold_summary"]["minimum_oracle"],
            item["fold_summary"]["standard_deviation"],
            -item["all"]["full_owner_queries"],
            -item["all"]["positive_edge_recall"],
            item["all"]["average_candidates"],
        )

    report["selected_max_score"] = min(report["results"], key=key)
    suffix = f"_address_k{address_budget}" if address_budget is not None else ""
    _atomic_json(cache / f"name_evaluation_k{budget}_rescue_k{rescue_budget}{suffix}.json", report)
    return report


def measure_candidate_shards(paths, truth, baseline, folds, held_folds, elapsed):
    """Evaluate large candidate sets with only per-query counts and positives in RAM.

    The official metric and recall/coverage still run through _metrics and the
    authoritative evaluator. Negative IDs are needed only while reading one batch;
    they contribute to volume and incremental cost, never to an oracle prediction.
    """
    from collections import Counter

    counts, retained, base_counts = Counter(), Counter(), Counter(s1 for s1, _ in baseline)
    source_counts = {source: Counter() for source in ("S2", "S3")}
    covered = set()
    seen_queries = set()
    for path in paths:
        file_queries = set()
        for batch in pq.ParquetFile(path).iter_batches(
            columns=["s1_id", "target_id", "target_source"], batch_size=65536
        ):
            for s1, target, source in batch.to_pandas().itertuples(index=False, name=None):
                if s1 not in truth or source not in source_counts:
                    raise ValueError("unknown query/source in candidate shard")
                counts[s1] += 1
                source_counts[source][s1] += 1
                retained[s1] += (s1, target) in baseline
                file_queries.add(s1)
                if target in truth[s1]:
                    covered.add((s1, target))
        if file_queries & seen_queries:
            raise ValueError("query appears in multiple candidate shards")
        seen_queries.update(file_queries)
    positives = pd.DataFrame(
        [(s1, target, target[:2]) for s1, target in sorted(covered)],
        columns=["s1_id", "target_id", "target_source"],
    )
    result = grouped_measurements(positives, truth, baseline, folds, held_folds, elapsed)
    held = {s1 for s1, f in folds.items() if f in held_folds}
    groups = {"all": set(truth), "held_out": held, "development": set(truth) - held}
    groups.update(
        {
            f"fold_{f}": {s1 for s1, v in folds.items() if v == f}
            for f in sorted(set(folds.values()))
        }
    )
    for name, ids in groups.items():
        m = result[name]
        values = np.asarray([counts[s1] for s1 in sorted(ids)], dtype=np.int64)
        rows = int(values.sum())
        added = rows - sum(retained[s1] for s1 in ids)
        m.update(
            total_candidate_rows=rows,
            average_candidates=float(values.mean()) if len(values) else 0.0,
            median_candidates=float(np.median(values)) if len(values) else 0.0,
            maximum_candidates=int(values.max()) if len(values) else 0,
            zero_candidate_rate=float((values == 0).mean()) if len(values) else 0.0,
            incremental_candidates_added=added,
            net_candidate_change=rows - sum(base_counts[s1] for s1 in ids),
            candidates_added_per_recovered_edge=added / m["incremental_edges_recovered"]
            if m["incremental_edges_recovered"]
            else None,
        )
        for quantile in (90, 95, 99):
            m[f"p{quantile}_candidates"] = (
                float(np.quantile(values, quantile / 100)) if len(values) else 0.0
            )
        for source in ("S2", "S3"):
            m["source_recall"][source]["candidate_rows"] = sum(
                source_counts[source][s1] for s1 in ids
            )
    return result, covered


def verify_profile_cached(
    config_path: Path,
    profile: str,
    *,
    rescue_path: Path,
    name_path: Path | None = None,
    address_path: Path | None = None,
    data_paths_config: Path | None = None,
):
    """Run the production gate, fusion, Arrow validation and atomic writer on the sample.

    Existing exact/address artifacts remain the unchanged diagnostic reference.
    Expanded name/address and missing-address retrieval use the production sparse backend.
    """
    from entity_resolution.production_candidates import atomic_shard, load_profile

    config = yaml.safe_load(config_path.read_text())
    paths = load_data_paths(data_paths_config or Path(config["data_paths_config"]))
    cache = Path(config["cache"])
    _protect_outputs(cache, paths)
    production = load_profile(DEFAULT_CONFIG.with_name("production_candidates.yaml"), profile)
    name_k = production.get("name_top_k", production["sparse"]["top_k"])
    address_k = production.get("address_top_k", production["sparse"]["top_k"])
    rescue = production["rescue"]
    if name_k > 80 and name_path is None:
        raise ValueError("expanded profile requires matching name evidence")
    if address_k > 80 and address_path is None:
        raise ValueError("expanded profile requires matching address evidence")
    context = json.loads((cache / "context.json").read_text())
    for name, digest in context["files"].items():
        if _sha256(cache / name) != digest:
            raise ValueError("corrupt training context")
    for key, digest in context["identity"]["inputs"].items():
        if _sha256(paths[key]) != digest:
            raise ValueError("changed raw training input")
    inputs = {}
    for label, path in (("rescue", rescue_path), ("name", name_path), ("address", address_path)):
        if path is None:
            continue
        info = json.loads(path.with_suffix(".json").read_text())
        if _sha256(path) != info["sha256"]:
            raise ValueError(f"corrupt {label} evidence")
        evidence_identity = info["identity"]
        if evidence_identity["query_sha256"] != context["files"]["queries.parquet"]:
            raise ValueError(f"wrong {label} query sample")
        index_path = cache / (
            "missing_address_index/index.json"
            if label == "rescue"
            else f"full_{label}_index/index.json"
        )
        if _sha256(index_path) != evidence_identity["index_sha256"]:
            raise ValueError(f"wrong {label} index")
        if label == "rescue" and (
            evidence_identity["top_k"] < rescue["top_k"]
            or evidence_identity["threshold"] > rescue["threshold"]
        ):
            raise ValueError("rescue evidence does not cover profile budget/threshold")
        if label == "name" and evidence_identity["budget"] < name_k:
            raise ValueError("name evidence does not cover profile budget")
        if label == "address" and evidence_identity["budget"] < address_k:
            raise ValueError("address evidence does not cover profile budget")
        inputs[label] = {"path": str(path), **info}
    v2 = yaml.safe_load(Path(config["v2_config"]).read_text())
    evidence, sample, validation = validate_and_load_evidence(
        Path(v2["artifact_root"]),
        sample_path=Path(v2["diagnostic_sample"]),
        v1_config_path=Path(v2["v1_config"]),
        expected_fingerprints=v2["expected_artifact_sha256"],
    )
    previous = json.loads(Path(config["output"]).read_text())
    if (
        validation != previous["validation"]
        or _sha256(Path(config["aggregate"])) != previous["aggregate_sha256"]
    ):
        raise ValueError("changed base evidence")
    aggregate = pd.read_parquet(config["aggregate"])
    baseline = select_candidates(aggregate, FusionSpec("rrf", global_budget=40))
    baseline_pairs = set(baseline[["s1_id", "target_id"]].itertuples(index=False, name=None))
    evidence = evidence.merge(
        aggregate[["s1_id", "target_id", "target_content_key"]],
        on=["s1_id", "target_id"],
        validate="many_to_one",
    )[EVIDENCE_COLUMNS]
    del aggregate, baseline
    queries = pd.read_parquet(cache / "queries.parquet")
    original_gate = set(rescue_queries(queries, evidence).entity_id)
    if configuration_hash(sorted(original_gate)) != inputs["rescue"]["identity"]["gated_ids"]:
        raise ValueError("rescue evidence gate mismatch")
    truth_list = load_ground_truth(cache / "sample_truth.tsv")
    truth = {s1: set(v) for s1, v in truth_list.items()}
    folds, _ = create_folds(
        truth_list,
        countries=dict(zip(sample.s1_id, sample.country, strict=True)),
        n_splits=v2["folds"]["count"],
        random_state=v2["seed"],
    )
    stored = pd.read_parquet(config["folds"])
    if (
        folds != dict(zip(stored.s1_id, stored.fold, strict=True))
        or _sha256(Path(config["folds"])) != previous["folds_sha256"]
    ):
        raise ValueError("changed grouped folds")
    root = cache / f"verified_{profile}"
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "configuration": production,
        "inputs": inputs,
        "validation": validation,
        "queries_sha256": context["files"]["queries.parquet"],
        "production_sha256": _sha256(Path(__file__).with_name("production_candidates.py")),
        "fusion_sha256": _sha256(Path(__file__).with_name("candidate_fusion.py")),
    }
    manifest = root / "manifest.json"
    if manifest.exists():
        existing = json.loads(manifest.read_text())
        if existing["identity"] != identity:
            raise ValueError("incompatible verified profile directory; use a new cache directory")
        for shard in existing["shards"]:
            if _sha256(root / shard["file"]) != shard["sha256"]:
                raise ValueError("corrupt verified profile shard")
        if existing.get("complete"):
            return existing
    state = {
        "identity": identity,
        "shards": [],
        "complete": False,
        "activated_queries": 0,
        "newly_gated_queries": 0,
        "scope": "5,000 training diagnostic queries; reused exact/address evidence",
    }
    started = time.perf_counter()
    for start in range(0, len(queries), 250):
        query = queries.iloc[start : start + 250]
        ids = list(query.entity_id)
        base = evidence[evidence.s1_id.isin(ids)]
        if name_path is not None:
            names = pd.read_parquet(
                name_path, filters=[("s1_id", "in", ids), ("channel_rank", "<=", name_k)]
            )
            base = pd.concat(
                [base[~base.channel.eq("name_char_tfidf")], names[EVIDENCE_COLUMNS]],
                ignore_index=True,
            )
        if address_path is not None:
            addresses = pd.read_parquet(
                address_path, filters=[("s1_id", "in", ids), ("channel_rank", "<=", address_k)]
            )
            base = pd.concat(
                [base[~base.channel.eq("address_char_tfidf")], addresses[EVIDENCE_COLUMNS]],
                ignore_index=True,
            )
        gated = rescue_queries(query, base)
        new_queries = gated[~gated.entity_id.isin(original_gate)]
        state["activated_queries"] += len(gated)
        state["newly_gated_queries"] += len(new_queries)
        extra = pd.read_parquet(
            rescue_path,
            filters=[
                ("s1_id", "in", list(gated.entity_id)),
                ("channel_rank", "<=", rescue["top_k"]),
                ("retrieval_score", ">=", rescue["threshold"]),
            ],
        )
        if len(new_queries):
            retriever = _subset_index(cache, Path(v2["artifact_root"]), threads=config["threads"])
            new = pd.DataFrame(
                retriever.sparse_rows(
                    new_queries,
                    profile,
                    missing_address_only=True,
                    top_k=rescue["top_k"],
                    threshold=rescue["threshold"],
                ),
                columns=EVIDENCE_COLUMNS,
            )
            extra = pd.concat([extra, new], ignore_index=True)
        selected, _ = fuse_budgeted(
            pd.concat([base, extra[EVIDENCE_COLUMNS]], ignore_index=True),
            query,
            production["policy"],
            run_id=profile,
        )
        path = root / f"part-{start // 250:06d}.parquet"
        atomic_shard(selected, path, profile)
        state["shards"].append(
            {
                "file": path.name,
                "sha256": _sha256(path),
                "rows": len(selected),
                "queries": len(query),
                "bytes": path.stat().st_size,
            }
        )
        _atomic_json(manifest, state)
        print(f"verified {profile} queries {start + len(query)}/{len(queries)}", flush=True)
    state["fusion_and_write_seconds"] = time.perf_counter() - started
    state["metrics"], covered_pairs = measure_candidate_shards(
        [root / shard["file"] for shard in state["shards"]],
        truth,
        baseline_pairs,
        folds,
        v2["folds"]["held_out"],
        state["fusion_and_write_seconds"],
    )
    missed = pd.read_csv(v2["outputs"]["missed_edges"], sep="\t", keep_default_na=False)
    audit_pairs = set(missed[["s1_id", "target_id"]].itertuples(index=False, name=None))
    recovered = covered_pairs & audit_pairs
    state["taxonomy"] = taxonomy(
        missed, queries, pd.read_parquet(cache / "audit_targets.parquet"), {profile: recovered}
    )
    state["complete"] = True
    _atomic_json(manifest, state)
    print(profile, json.dumps(state["metrics"]["fold_summary"]), flush=True)
    return state


def expand_address_channel(config_path: Path, *, data_paths_config: Path | None = None, budget=512):
    """Expand the existing address channel with one ephemeral source cache at a time."""
    import shutil

    from scipy import sparse

    config = yaml.safe_load(config_path.read_text())
    paths = load_data_paths(data_paths_config or Path(config["data_paths_config"]))
    cache = Path(config["cache"])
    _protect_outputs(cache, paths)
    context = json.loads((cache / "context.json").read_text())
    for name, digest in context["files"].items():
        if _sha256(cache / name) != digest:
            raise ValueError("corrupt training context")
    for key, digest in context["identity"]["inputs"].items():
        if _sha256(paths[key]) != digest:
            raise ValueError("changed raw training source")
    v2 = yaml.safe_load(Path(config["v2_config"]).read_text())
    assets = Path(v2["artifact_root"])
    queries = pd.read_parquet(cache / "queries.parquet")
    source_runs = {}
    for source in ("S2", "S3"):
        root = cache / f"address_work_{source}"
        output = cache / f"{source}_full_address_k{budget}.parquet"
        sidecar = output.with_suffix(".json")
        identity = {
            "source_sha256": context["identity"]["inputs"][f"train_source{source[-1]}"],
            "vocabulary_sha256": _sha256(assets / f"{source}_address_vectorizer.joblib"),
            "query_sha256": context["files"]["queries.parquet"],
            "budget": budget,
            "implementation_sha256": _sha256(Path(__file__).with_name("production_candidates.py")),
        }
        if sidecar.exists():
            state = json.loads(sidecar.read_text())
            if state["identity"] != identity or _sha256(output) != state["sha256"]:
                raise ValueError("incompatible/corrupt expanded address evidence")
            source_runs[source] = state
            continue
        root.mkdir(parents=True, exist_ok=True)
        retriever = CachedRetriever(
            root,
            {
                "threads": config["threads"],
                "max_evidence_rows": 2000000,
                "sparse": {"top_k": budget, "threshold": 0.2},
            },
        )
        started = time.perf_counter()
        if retriever.manifest.exists():
            index = json.loads(retriever.manifest.read_text())
            if index["identity"] != identity:
                raise ValueError("incompatible address work index")
            for name, digest in index["files"].items():
                if _sha256(root / name) != digest:
                    raise ValueError("corrupt address work index")
        else:
            vectorizer = joblib.load(assets / f"{source}_address_vectorizer.joblib")["vectorizer"]
            files, blocks = {}, []
            for s in ("S2", "S3"):
                for field in ("name", "address"):
                    name = f"{s}_{field}.joblib"
                    joblib.dump(
                        vectorizer if s == source and field == "address" else None, root / name
                    )
                    files[name] = _sha256(root / name)
            count = 0
            for targets in read_batches(paths[f"train_source{source[-1]}"], 50000):
                if shutil.disk_usage(root).free < 1500 * 2**20:
                    raise OSError(
                        "address experiment stopped: preserve 1.5 GiB local disk headroom"
                    )
                targets["address"] = [
                    normalize_field(v).unicode_preserving for v in targets.business_address
                ]
                targets["tie"] = [
                    content_tie_key(r.business_name, r.business_address, r.country)
                    for r in targets.itertuples(index=False)
                ]
                for country, group in targets.groupby("country", sort=True):
                    group = group.sort_values("tie", kind="stable")
                    for start in range(0, len(group), 20000):
                        part = group.iloc[start : start + 20000]
                        number = len(blocks)
                        meta_name, matrix_name = (
                            f"block-{number:06d}.parquet",
                            f"block-{number:06d}-address.npz",
                        )
                        metadata = part[["entity_id", "tie"]].rename(columns={"entity_id": "id"})
                        metadata.assign(address_missing=part.address.eq("")).to_parquet(
                            root / meta_name, index=False
                        )
                        sparse.save_npz(
                            root / matrix_name,
                            vectorizer.transform(part.address).astype(np.float32),
                        )
                        blocks.append(
                            {
                                "source": source,
                                "country": country,
                                "metadata": meta_name,
                                "address": matrix_name,
                                "number": number,
                            }
                        )
                        for name in (meta_name, matrix_name):
                            files[name] = _sha256(root / name)
                count += len(targets)
                if count % 500000 == 0:
                    print(f"address index {source}: {count}", flush=True)
            _atomic_json(
                retriever.manifest, {"identity": identity, "files": files, "blocks": blocks}
            )
        index_seconds = time.perf_counter() - started
        index_sha256 = _sha256(retriever.manifest)
        started = time.perf_counter()
        writer = None
        try:
            for start in range(0, len(queries), config["query_batch_size"]):
                query = queries.iloc[start : start + config["query_batch_size"]]
                rows = retriever.sparse_rows(
                    query, "address_expansion", top_k=budget, threshold=0.2
                )
                extra = pd.DataFrame(rows, columns=EVIDENCE_COLUMNS).astype(
                    {"retrieval_score": "float32", "channel_rank": "int16", "exact_match": "bool"}
                )
                table = pa.Table.from_pandas(extra, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        output.with_suffix(".tmp"), table.schema, compression="zstd"
                    )
                writer.write_table(table)
                print(
                    f"address retrieval {source}: {start + len(query)}/{len(queries)}", flush=True
                )
        finally:
            if writer:
                writer.close()
        output.with_suffix(".tmp").replace(output)
        state = {
            "identity": identity,
            "sha256": _sha256(output),
            "index_sha256": index_sha256,
            "index_seconds": index_seconds,
            "retrieval_seconds": time.perf_counter() - started,
        }
        _atomic_json(sidecar, state)
        source_runs[source] = state
        # These are only this experiment's derived temporary matrices. The validated
        # evidence and its source/vocabulary/index hashes remain reproducible.
        shutil.rmtree(root)
    output = cache / f"full_address_k{budget}.parquet"
    writer = None
    try:
        for source in ("S2", "S3"):
            for batch in pq.ParquetFile(
                cache / f"{source}_full_address_k{budget}.parquet"
            ).iter_batches():
                if writer is None:
                    writer = pq.ParquetWriter(
                        output.with_suffix(".tmp"), batch.schema, compression="zstd"
                    )
                writer.write_batch(batch)
    finally:
        if writer:
            writer.close()
    output.with_suffix(".tmp").replace(output)
    archived = cache / "full_address_index/index.json"
    _atomic_json(archived, {"source_runs": source_runs, "temporary_matrices_released": True})
    state = {
        "identity": {
            "budget": budget,
            "query_sha256": context["files"]["queries.parquet"],
            "index_sha256": _sha256(archived),
        },
        "sha256": _sha256(output),
        "source_runs": source_runs,
        "retrieval_seconds": sum(r["retrieval_seconds"] for r in source_runs.values()),
        "index_seconds": sum(r["index_seconds"] for r in source_runs.values()),
    }
    _atomic_json(output.with_suffix(".json"), state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-paths", type=Path)
    parser.add_argument("--run-rescue", action="store_true")
    parser.add_argument(
        "--rebudget", type=int, help="Re-fuse saved rescue evidence; no new retrieval"
    )
    parser.add_argument("--maximize-cached", action="store_true")
    parser.add_argument("--rescue-budget", type=int, default=512)
    parser.add_argument("--rescue-threshold", type=float, default=0.18)
    parser.add_argument("--expand-name-budget", type=int)
    parser.add_argument("--evaluate-name-budget", type=int)
    parser.add_argument("--evaluate-address-budget", type=int)
    parser.add_argument("--verify-profile", choices=["competitive_adaptive", "max_score"])
    parser.add_argument("--rescue-evidence", type=Path)
    parser.add_argument("--name-evidence", type=Path)
    parser.add_argument("--address-evidence", type=Path)
    parser.add_argument("--expand-address-budget", type=int)
    args = parser.parse_args()
    if args.expand_address_budget is not None:
        expand_address_channel(
            args.config, data_paths_config=args.data_paths, budget=args.expand_address_budget
        )
    elif args.verify_profile:
        if args.rescue_evidence is None:
            parser.error("--verify-profile requires --rescue-evidence")
        verify_profile_cached(
            args.config,
            args.verify_profile,
            rescue_path=args.rescue_evidence,
            name_path=args.name_evidence,
            address_path=args.address_evidence,
            data_paths_config=args.data_paths,
        )
    elif args.evaluate_name_budget is not None:
        evaluate_name_expansion(
            args.config,
            budget=args.evaluate_name_budget,
            rescue_budget=args.rescue_budget,
            address_budget=args.evaluate_address_budget,
            data_paths_config=args.data_paths,
        )
    elif args.expand_name_budget is not None:
        expand_name_channel(
            args.config, data_paths_config=args.data_paths, budget=args.expand_name_budget
        )
    elif args.maximize_cached:
        maximize_cached(
            args.config,
            data_paths_config=args.data_paths,
            rescue_budget=args.rescue_budget,
            threshold=args.rescue_threshold,
        )
    elif args.rebudget is not None:
        rebudget_cached(args.config, args.rebudget, data_paths_config=args.data_paths)
    else:
        run(args.config, data_paths_config=args.data_paths, run_rescue=args.run_rescue)


if __name__ == "__main__":
    main()
