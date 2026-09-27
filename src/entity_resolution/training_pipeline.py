"""Leakage-safe, resumable OOF CatBoost training and decoder selection."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from entity_resolution.catboost_ranker import fit, predict
from entity_resolution.config import DEFAULT_CONFIG, load_data_paths
from entity_resolution.decode import Decoder, search_thresholds
from entity_resolution.folds import create_folds
from entity_resolution.ground_truth import load_ground_truth
from entity_resolution.metrics import evaluate_macro_f0_5_efficient
from entity_resolution.model_artifact import load_artifact, save_model, seal_artifact, versions
from entity_resolution.negative_sampling import sample, sampled_frame, tie_hash
from entity_resolution.training_schema import (
    atomic_json,
    candidate_diagnostics,
    contract,
    feature_inputs,
    frames,
    open_store,
    protect_output,
    sha256,
    source_ids,
    stage_features,
)

LOG = logging.getLogger(__name__)
SCORE_SCHEMA = pa.schema(
    [
        ("s1_id", pa.string()),
        ("target_id", pa.string()),
        ("target_source", pa.string()),
        ("score", pa.float64()),
        ("fold", pa.int32()),
    ]
)


def load_config(path, smoke=False):
    config = yaml.safe_load(Path(path).read_text())
    required = {
        "seed",
        "fold_count",
        "batch_size",
        "smoke_entities",
        "model",
        "sampling",
        "max_decode_rows",
        "decoder_search",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"ranker config requires exactly: {sorted(required)}")
    if min(config["batch_size"], config["max_decode_rows"], config["smoke_entities"]) < 1:
        raise ValueError("batch, decode and smoke limits must be positive")
    if config["fold_count"] < 2 or config["model"]["thread_count"] < 1:
        raise ValueError("need >=2 folds and >=1 thread")
    if smoke:
        config["fold_count"] = 2
        config["model"]["iterations"] = min(12, config["model"]["iterations"])
        config["model"]["depth"] = min(3, config["model"]["depth"])
        config["model"]["thread_count"] = min(2, config["model"]["thread_count"])
    return config


def reduced_scores(db, decoder_config, max_rows):
    """Exactly preserve decoder decisions while dropping threshold/top-k ineligible rows.

    Ownership only needs the best two eligible owners per target (winner and margin
    runner-up). SQL applies the same score/lexical tie order as Decoder.
    """
    global_t = decoder_config.get("global_threshold") or 0.0
    s2 = decoder_config.get("s2_threshold")
    s3 = decoder_config.get("s3_threshold")
    s2 = global_t if s2 is None else s2
    s3 = global_t if s3 is None else s3
    evidence = decoder_config.get("min_evidence_score")
    if evidence is not None:
        s2, s3 = max(s2, evidence), max(s3, evidence)
    query = """WITH eligible AS (
      SELECT s1,target,source,score, ROW_NUMBER() OVER (
        PARTITION BY s1,source ORDER BY score DESC,s1,target) AS rank
      FROM pairs WHERE (source='S2' AND score>=?) OR (source='S3' AND score>=?)
    ), topk AS (SELECT * FROM eligible WHERE rank<=?), owners AS (
      SELECT *,ROW_NUMBER() OVER (PARTITION BY target ORDER BY score DESC,s1,target) AS owner
      FROM topk
    ) SELECT s1,target,source,score FROM owners WHERE owner<=? ORDER BY s1,target"""
    topk = decoder_config.get("top_k_per_source") or 9223372036854775807
    owners = 2 if decoder_config.get("enforce_target_ownership") else 9223372036854775807
    rows = db.execute(query, (s2, s3, topk, owners)).fetchmany(max_rows + 1)
    if len(rows) > max_rows:
        raise ValueError(
            "max_decode_rows exceeded after exact SQL reduction; increase on a "
            "larger machine or use a smaller training smoke subset"
        )
    result = pd.DataFrame(rows, columns=["s1_id", "target_id", "target_source", "score"])
    result["score"] = result["score"].astype(float)
    return result


def tune_decoder(db, truth, config, max_rows):
    # Evaluate each compact option through the authoritative search API. Combine
    # winners on their shared probability subset to preserve its exact tie-breaker.
    all_ids = set(truth)
    coarse = []
    for threshold in config["thresholds"]:
        for k in config["top_k"]:
            for ownership, margin in ((False, None), (True, 0.0), (True, 0.05)):
                coarse.append(
                    dict(
                        global_threshold=threshold,
                        s2_threshold=None,
                        s3_threshold=None,
                        top_k_per_source=k,
                        enforce_target_ownership=ownership,
                        score_margin=margin,
                        min_evidence_score=None,
                    )
                )

    def search(space):
        # A single frame filtered at the lowest searched threshold is sufficient;
        # top-k and ownership remain entirely in the authoritative decoder.
        floor = min(
            min(
                p["s2_threshold"] if p["s2_threshold"] is not None else p["global_threshold"],
                p["s3_threshold"] if p["s3_threshold"] is not None else p["global_threshold"],
            )
            for p in space
        )
        ceiling_k = max(p["top_k_per_source"] for p in space)
        frame = reduced_scores(
            db, {"global_threshold": floor, "top_k_per_source": ceiling_k}, max_rows
        )
        return search_thresholds(frame, truth, space, all_ids)

    best, _, coarse_results = search(coarse)
    center = best["global_threshold"]
    step = config["refinement_step"]
    local = [best]
    for delta2, delta3 in (
        (-step, -step),
        (step, step),
        (-step, 0),
        (0, -step),
        (step, 0),
        (0, step),
    ):
        local.append(
            {
                **best,
                "s2_threshold": min(1.0, max(0.0, center + delta2)),
                "s3_threshold": min(1.0, max(0.0, center + delta3)),
            }
        )
    best, score, local_results = search(local)
    return best, score, pd.concat([coarse_results, local_results], ignore_index=True)


def score_fold(db, model, fold, path, batch_size):
    temporary = Path(str(path) + ".tmp")
    with pq.ParquetWriter(temporary, SCORE_SCHEMA, compression="zstd") as writer:
        for frame in frames(db, "fold=?", (fold,), batch_size):
            output = frame[["s1_id", "target_id", "target_source"]].copy()
            output["score"] = predict(model, frame)
            output["fold"] = fold
            writer.write_table(
                pa.Table.from_pandas(output, schema=SCORE_SCHEMA, preserve_index=False)
            )
    os.replace(temporary, path)


def import_scores(db, path, fold, assignments, batch_size):
    count = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        updates = []
        for row in batch.to_pylist():
            if row["fold"] != fold or assignments[row["s1_id"]] != fold:
                raise ValueError("OOF provenance mismatch")
            updates.append(
                (row["score"], row["s1_id"], row["target_id"], row["target_source"], fold)
            )
        db.executemany(
            "UPDATE pairs SET score=? WHERE s1=? AND target=? AND source=? AND fold=?", updates
        )
        count += len(updates)
    expected = db.execute("SELECT COUNT(*) FROM pairs WHERE fold=?", (fold,)).fetchone()[0]
    missing = db.execute(
        "SELECT COUNT(*) FROM pairs WHERE fold=? AND score IS NULL", (fold,)
    ).fetchone()[0]
    if count != expected or missing:
        raise ValueError("incomplete OOF predictions")
    db.commit()


def train(
    features, output, config_path, data_paths_config=DEFAULT_CONFIG, *, smoke=False, resume=False
):
    features, output = Path(features), Path(output)
    paths = load_data_paths(data_paths_config)
    protect_output(output, paths, [features])
    config = load_config(config_path, smoke)
    files, fingerprints = feature_inputs(features, "train")
    ids = source_ids(paths, "train")
    truth = load_ground_truth(paths["train_ground_truth"])
    if set(truth) != ids["S1"] or any(t not in ids[t[:2]] for ts in truth.values() for t in ts):
        raise ValueError("ground truth does not match authoritative training source IDs")
    identity = {
        "config": config,
        "smoke": smoke,
        "inputs": fingerprints,
        "data": {
            k: sha256(paths[k])
            for k in ("train_source1", "train_source2", "train_source3", "train_ground_truth")
        },
        "contract": contract(),
        "versions": versions(),
        "implementation": {
            p.name: sha256(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("negative_sampling.py"),
                Path(__file__).with_name("catboost_ranker.py"),
                Path(__file__).with_name("training_schema.py"),
                Path(__file__).with_name("model_artifact.py"),
                Path(__file__).with_name("feature_schema.py"),
                Path(__file__).with_name("folds.py"),
                Path(__file__).with_name("ground_truth.py"),
                Path(__file__).with_name("decode.py"),
                Path(__file__).with_name("metrics.py"),
            ]
        },
    }
    if output.exists():
        if resume:
            _, metadata = load_artifact(output)
            if metadata["identity"] == identity:
                return metadata
        raise ValueError("artifact already exists or resume identity changed")
    work = output.with_name(output.name + ".work")
    protect_output(work, paths, [features])
    work.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(output.name + ".lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("training output is locked") from exc
        state_path = work / "state.json"
        if state_path.exists():
            if not resume or json.loads(state_path.read_text())["identity"] != identity:
                raise ValueError("work directory exists; use --resume with identical inputs/config")
        elif resume or any(work.iterdir()):
            raise ValueError("cannot resume missing/incomplete state manifest")
        atomic_json(state_path, {"identity": identity})
        full_truth = truth
        if smoke:
            chosen = sorted(truth, key=lambda s: tie_hash(config["seed"], s))[
                : config["smoke_entities"]
            ]
            truth = {s: truth[s] for s in sorted(chosen)}
        assignments, _ = create_folds(
            truth, n_splits=config["fold_count"], random_state=config["seed"]
        )
        atomic_json(work / "folds.json", assignments)
        db = open_store(work / "pairs.sqlite")
        try:
            LOG.info("Validating/staging %d feature shards", len(files))
            run_id = stage_features(db, files, ids, config["batch_size"], full_truth, set(truth))
            db.executemany(
                "UPDATE pairs SET fold=? WHERE s1=?", [(f, s) for s, f in assignments.items()]
            )
            db.commit()
            diagnostics = candidate_diagnostics(db, truth)
            iterations = []
            for fold in range(config["fold_count"]):
                LOG.info("OOF fold %d/%d", fold + 1, config["fold_count"])
                validation = {s for s, f in assignments.items() if f == fold}
                training = set(truth) - validation
                model_path = work / f"fold-{fold}.cbm"
                scores_path = work / f"oof-{fold}.parquet"
                checkpoint = work / f"fold-{fold}.json"
                if checkpoint.exists():
                    state = json.loads(checkpoint.read_text())
                    if (
                        state["validation_s1"] != sorted(validation)
                        or state["training_s1"] != sorted(training)
                        or sha256(model_path) != state["model_sha256"]
                        or sha256(scores_path) != state["scores_sha256"]
                    ):
                        raise ValueError("corrupt fold checkpoint")
                else:
                    # Inner S1 fold for early stopping; outer validation labels are never
                    # passed to the fitter or its stopping/model-selection state.
                    inner_truth = {s: truth[s] for s in sorted(training)}
                    inner_assignments = (
                        create_folds(inner_truth, n_splits=2, random_state=config["seed"] + fold)[0]
                        if len(training) >= 2
                        else {}
                    )
                    inner_eval = {s for s, f in inner_assignments.items() if f == 0}
                    inner_train = training - inner_eval
                    inner_used = False
                    best_iterations = config["model"]["iterations"]
                    if inner_eval and inner_train:
                        sample(db, inner_train, config["sampling"], config["seed"], "inner_train")
                        sample(db, inner_eval, config["sampling"], config["seed"], "inner_eval")
                        tr, y = sampled_frame(db, "inner_train")
                        ev, ey = sampled_frame(db, "inner_eval")
                        if len(set(y)) == 2 and len(set(ey)) == 2:
                            inner_model = fit(
                                tr, y, config["model"], config["seed"] + fold, (ev, ey)
                            )
                            best_iterations = inner_model.tree_count_
                            inner_used = True
                            del inner_model
                        del tr, y, ev, ey
                    sampling = sample(db, training, config["sampling"], config["seed"])
                    tr, y = sampled_frame(db)
                    if set(tr.s1_id) & validation:
                        raise ValueError("fold leakage detected")
                    parameters = {**config["model"], "iterations": best_iterations}
                    model = fit(tr, y, parameters, config["seed"] + fold)
                    del tr, y
                    save_model(model, model_path)
                    score_fold(db, model, fold, scores_path, config["batch_size"])
                    state = {
                        "fold": fold,
                        "training_s1": sorted(training),
                        "validation_s1": sorted(validation),
                        "iterations": best_iterations,
                        "inner_early_stopping": inner_used,
                        "inner_training_s1": sorted(inner_train),
                        "inner_validation_s1": sorted(inner_eval),
                        "sampling": sampling,
                        "model_sha256": sha256(model_path),
                        "scores_sha256": sha256(scores_path),
                    }
                    atomic_json(checkpoint, state)
                    del model
                iterations.append(state["iterations"])
                import_scores(db, scores_path, fold, assignments, config["batch_size"])
            LOG.info("Tuning decoder on strictly OOF probabilities")
            decoder, score, searches = tune_decoder(
                db, truth, config["decoder_search"], config["max_decode_rows"]
            )
            searches.to_parquet(work / "decoder_search.parquet", index=False)
            scored = reduced_scores(db, decoder, config["max_decode_rows"])
            predictions = Decoder(**decoder).decode(scored, set(truth))
            measured, pair_diag = evaluate_macro_f0_5_efficient(truth, predictions)
            if abs(score - measured) > 1e-9:
                raise ValueError("decoder reduction disagrees with threshold search")
            fold_scores = {}
            for fold in range(config["fold_count"]):
                subset = {s: truth[s] for s, f in assignments.items() if f == fold}
                fold_scores[str(fold)] = evaluate_macro_f0_5_efficient(
                    subset, {s: predictions[s] for s in subset}
                )[0]
            per_source = {}
            for source in ("S2", "S3"):
                n = sum(t.startswith(source + "-") for ts in truth.values() for t in ts)
                tp = sum(
                    t.startswith(source + "-") and t in truth[s]
                    for s, ts in predictions.items()
                    for t in ts
                )
                per_source[source] = tp / n if n else None
            summary = {
                "macro_f0_5": measured,
                "scope": "smoke" if smoke else "full",
                "fold_scores": fold_scores,
                "per_source_positive_recall": per_source,
                "predicted_pair_count": sum(map(len, predictions.values())),
                "false_positives_on_no_match": sum(
                    len(predictions[s]) for s in truth if not truth[s]
                ),
                "pair_diagnostics": pair_diag,
                "note": "OOF decoder-selection score; not a nested unbiased final estimate",
            }
            sampling = sample(db, set(truth), config["sampling"], config["seed"])
            tr, y = sampled_frame(db)
            final_parameters = {
                **config["model"],
                "iterations": sorted(iterations)[len(iterations) // 2],
            }
            LOG.info("Fitting final model on %d sampled pairs", len(tr))
            model = fit(tr, y, final_parameters, config["seed"])
            save_model(model, work / "model.cbm")
            del tr, y, model
        finally:
            db.close()
        (work / "pairs.sqlite").unlink(missing_ok=True)
        metadata = {
            "identity": identity,
            "parameters": final_parameters,
            "seed": config["seed"],
            "fold_strategy": "create_folds v1 S1 cardinality/composition; inner S1 stopping",
            "fold_count": config["fold_count"],
            "sampling_policy": config["sampling"],
            "sampling_diagnostics": sampling,
            "decoder": decoder,
            "input_fingerprints": fingerprints,
            "candidate_diagnostics": diagnostics,
            "oof_summary": summary,
            "training_run_id": run_id,
        }
        return seal_artifact(work, output, metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/ranker_v1.yaml"))
    parser.add_argument("--data-paths", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        result = train(
            args.features,
            args.output,
            args.config,
            args.data_paths,
            smoke=args.smoke,
            resume=args.resume,
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"ranker training failed: {exc}\n")
    print(json.dumps(result["oof_summary"], indent=2))


if __name__ == "__main__":
    main()
