"""Synthetic behavioral and end-to-end coverage for the ranker lane."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from entity_resolution.catboost_ranker import predict
from entity_resolution.config import load_data_paths
from entity_resolution.decode import Decoder
from entity_resolution.feature_schema import OUTPUT_SCHEMA, feature_names
from entity_resolution.inference_pipeline import infer
from entity_resolution.model_artifact import load_artifact
from entity_resolution.negative_sampling import sample, sampled_frame
from entity_resolution.training_pipeline import reduced_scores, train, tune_decoder
from entity_resolution.training_schema import (
    candidate_diagnostics,
    feature_inputs,
    labels,
    model_inputs,
    open_store,
    source_ids,
    stage_features,
)

REPO = Path(__file__).resolve().parents[1]


def feature(s1, target, positive=False, exact=False):
    row = {
        field.name: ("channel" if pa.types.is_string(field.type) else 0) for field in OUTPUT_SCHEMA
    }
    row.update(s1_id=s1, target_id=target, target_source=target[:2], run_id="synthetic")
    row.update(
        retrieval_score=0.95 if positive else 0.3,
        name_char_similarity=0.95 if positive else 0.2,
        exact_match=int(positive or exact),
    )
    return row


def write_features(path, rows, shard_size=11):
    path.mkdir()
    for index, start in enumerate(range(0, max(1, len(rows)), shard_size)):
        pq.write_table(
            pa.Table.from_pylist(rows[start : start + shard_size], schema=OUTPUT_SCHEMA),
            path / f"part-{index:06d}.parquet",
        )
    return path


@pytest.fixture
def fixture_data(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    configs = tmp_path / "configs"
    configs.mkdir()
    s1 = [f"S1-{i:02d}" for i in range(24)] + ["S1-empty"]
    targets = [f"S{src}-{i:02d}" for src in (2, 3) for i in range(32)]
    mapping = {}
    for mode in ("train", "test"):
        for source, members in ((1, s1), (2, targets[:32]), (3, targets[32:])):
            key = f"{mode}_source{source}"
            path = raw / (key + ".tsv")
            path.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                + "".join(f"{s}\tName\tAddress\tUS\n" for s in members)
            )
            mapping[key] = str(path)
    truth = {
        s: ([f"S2-{i:02d}", f"S3-{i:02d}"] if i % 3 != 0 else []) for i, s in enumerate(s1[:-1])
    }
    truth["S1-empty"] = ["S2-31"]
    gt = raw / "train_ground_truth.tsv"
    gt.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        + "".join(f"{s}\t{','.join(ts)}\n" for s, ts in truth.items())
    )
    mapping["train_ground_truth"] = str(gt)
    data_paths = configs / "data_paths.yaml"
    data_paths.write_text(yaml.safe_dump(mapping))
    rows = []
    for i, s in enumerate(s1[:-1]):
        for src in (2, 3):
            for j in (i, 28, 29, 30):
                target = f"S{src}-{j:02d}"
                rows.append(feature(s, target, target in truth[s], exact=j == 28))
    config = yaml.safe_load((REPO / "configs/ranker_v1.yaml").read_text())
    config["fold_count"] = 3
    config["model"].update(iterations=5, depth=2, thread_count=1, early_stopping_rounds=2)
    config["smoke_entities"] = 100
    config["batch_size"] = 13
    config["sampling"].update(per_s1_negative_cap=4, minimum_negatives=2)
    config["decoder_search"] = dict(thresholds=[0.3, 0.8], top_k=[1, 2], refinement_step=0.05)
    config_path = configs / "ranker.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return dict(
        root=tmp_path,
        rows=rows,
        truth=truth,
        data_paths=data_paths,
        config_path=config_path,
        config=config,
        mapping=mapping,
        features=write_features(tmp_path / "features", rows),
    )


def staged(data, root=None, truth=True):
    db = open_store(":memory:")
    paths = load_data_paths(data["data_paths"])
    files, _ = feature_inputs(root or data["features"], "train")
    stage_features(
        db, files, source_ids(paths, "train"), batch_size=7, truth=data["truth"] if truth else None
    )
    return db


def test_labels_and_feature_contract(fixture_data):
    data = fixture_data
    frame = pa.Table.from_pylist(data["rows"], schema=OUTPUT_SCHEMA).to_pandas()
    y = labels(frame, data["truth"])
    assert y.sum() == 32
    assert np.array_equal(y, labels(frame, data["truth"]))
    x = model_inputs(frame)
    assert list(x.columns) == list(feature_names())
    assert not {"s1_id", "target_id", "run_id"} & set(x.columns)
    assert "target_source" in x.columns
    with pytest.raises(ValueError, match="order/schema"):
        model_inputs(frame[frame.columns[::-1]])


def test_sampling_and_ceiling(fixture_data):
    data = fixture_data
    db = staged(data)
    diagnostics = candidate_diagnostics(db, data["truth"])
    assert diagnostics["total_truth_pairs"] == 33
    assert diagnostics["retrieved_truth_pairs"] == 32
    assert diagnostics["missed_truth_pairs"] == 1
    assert diagnostics["candidate_free_s1"] == 1
    assert diagnostics["no_match_s1"] == 8
    policy = data["config"]["sampling"]
    d = sample(db, set(data["truth"]), policy, 42)
    frame, y = sampled_frame(db)
    assert sum(y) == 32 == d["positives"]
    no_match = frame[frame.s1_id == "S1-00"]
    assert len(no_match) == 2
    assert set(no_match.target_source) == {"S2", "S3"}
    assert set(no_match.target_id) == {"S2-28", "S3-28"}
    assert frame.assign(label=y).query("label == 0").groupby("s1_id").size().max() <= 4
    sample(db, set(data["truth"]), policy, 42)
    pd.testing.assert_frame_equal(frame, sampled_frame(db)[0])
    with pytest.raises(ValueError, match="max_sampled_rows"):
        sample(db, set(data["truth"]), {**policy, "max_sampled_rows": 1}, 42)
    db.close()


def test_shard_boundary_determinism(fixture_data):
    data = fixture_data
    shuffled = list(reversed(data["rows"]))
    other = write_features(data["root"] / "other", shuffled, shard_size=17)
    db1, db2 = staged(data), staged(data, other)
    for db in (db1, db2):
        sample(db, set(data["truth"]), data["config"]["sampling"], 42)
    pd.testing.assert_frame_equal(sampled_frame(db1)[0], sampled_frame(db2)[0])
    assert sampled_frame(db1)[1] == sampled_frame(db2)[1]
    db1.close()
    db2.close()


@pytest.mark.parametrize("failure", ["duplicate", "nan", "ownership", "run", "missing", "order"])
def test_feature_rejection(fixture_data, failure):
    data = fixture_data
    rows = [dict(r) for r in data["rows"]]
    if failure == "duplicate":
        rows.append(rows[0])
    elif failure == "nan":
        rows[0]["retrieval_score"] = float("nan")
    elif failure == "ownership":
        rows[0]["target_source"] = "S3"
    elif failure == "run":
        rows[0]["run_id"] = "wrong"
    elif failure == "missing":
        rows[0]["strongest_channel"] = None
    other = write_features(data["root"] / "bad", rows)
    if failure == "order":
        path = next(other.glob("*.parquet"))
        table = pq.read_table(path)
        pq.write_table(table.select(table.column_names[::-1]), path)
    with pytest.raises(ValueError):
        staged(data, other)


def test_sql_reduction_matches_authoritative_decoder(fixture_data):
    db = staged(fixture_data)
    rng = np.random.default_rng(42)
    keys = db.execute("SELECT s1,target,source FROM pairs").fetchall()
    # Ties and shared targets exercise ownership runner-up and lexical behavior.
    db.executemany(
        "UPDATE pairs SET score=? WHERE s1=? AND target=? AND source=?",
        [(float(rng.choice([0.3, 0.5, 0.7, 0.9])), *key) for key in keys],
    )
    all_rows = reduced_scores(db, {}, 1000)
    truth = fixture_data["truth"]
    for threshold in (0.3, 0.8):
        for k in (1, 2, 4):
            for ownership, margin in ((False, None), (True, 0), (True, 0.2)):
                config = dict(
                    global_threshold=threshold,
                    top_k_per_source=k,
                    enforce_target_ownership=ownership,
                    score_margin=margin,
                )
                reduced = reduced_scores(db, config, 1000)
                assert Decoder(**config).decode(reduced, set(truth)) == Decoder(**config).decode(
                    all_rows, set(truth)
                )
    search = fixture_data["config"]["decoder_search"]
    first = tune_decoder(db, truth, search, 1000)
    second = tune_decoder(db, truth, search, 1000)
    assert first[:2] == second[:2]
    with pytest.raises(ValueError, match="max_decode_rows"):
        reduced_scores(db, {}, 1)
    db.close()


@pytest.fixture
def trained(fixture_data):
    data = fixture_data
    output = data["root"] / "model"
    meta = train(data["features"], output, data["config_path"], data["data_paths"], smoke=True)
    return data, output, meta


def test_tiny_training_roundtrip_oof_and_resume(trained):
    data, output, meta = trained
    model, loaded = load_artifact(output)
    assert meta == loaded
    frame = pa.Table.from_pylist(data["rows"], schema=OUTPUT_SCHEMA).to_pandas()
    scores = predict(model, frame)
    assert np.isfinite(scores).all()
    assert scores.min() >= 0 and scores.max() <= 1
    assignment = json.loads((output / "folds.json").read_text())
    observed = set()
    for fold in range(2):
        checkpoint = json.loads((output / f"fold-{fold}.json").read_text())
        assert not set(checkpoint["training_s1"]) & set(checkpoint["validation_s1"])
        assert not set(checkpoint["inner_training_s1"]) & set(checkpoint["inner_validation_s1"])
        assert set(checkpoint["inner_training_s1"]) <= set(checkpoint["training_s1"])
        assert set(checkpoint["inner_validation_s1"]) <= set(checkpoint["training_s1"])
        oof = pq.read_table(output / f"oof-{fold}.parquet").to_pandas()
        assert all(assignment[s] == fold for s in oof.s1_id)
        assert set(oof.s1_id) <= set(checkpoint["validation_s1"])
        assert not set(oof.s1_id) & set(checkpoint["training_s1"])
        observed.update(zip(oof.s1_id, oof.target_id, strict=True))
        from catboost import CatBoostClassifier

        fold_model = CatBoostClassifier()
        fold_model.load_model(str(output / f"fold-{fold}.cbm"))
        heldout = frame[frame.s1_id.isin(checkpoint["validation_s1"])].sort_values(
            ["s1_id", "target_id"]
        )
        np.testing.assert_array_equal(oof.score, predict(fold_model, heldout))
    assert len(observed) == len(frame)
    assert (
        train(
            data["features"],
            output,
            data["config_path"],
            data["data_paths"],
            smoke=True,
            resume=True,
        )
        == meta
    )
    with pytest.raises(ValueError, match="already exists"):
        train(data["features"], output, data["config_path"], data["data_paths"], smoke=True)
    # Simulate interruption after the fold checkpoints; resume must reuse and verify them.
    work = output.with_name(output.name + ".work")
    shutil.copytree(output, work)
    shutil.rmtree(output)
    before = (work / "fold-0.cbm").stat().st_mtime_ns
    resumed = train(
        data["features"], output, data["config_path"], data["data_paths"], smoke=True, resume=True
    )
    assert (output / "fold-0.cbm").stat().st_mtime_ns == before
    assert resumed["oof_summary"] == meta["oof_summary"]


@pytest.mark.parametrize("corruption", ["model", "missing", "order", "version", "incomplete"])
def test_artifact_rejection(trained, corruption):
    _, output, _ = trained
    path = output / "metadata.json"
    meta = json.loads(path.read_text())
    if corruption == "model":
        (output / "model.cbm").write_bytes(b"bad")
    elif corruption == "missing":
        (output / "model.cbm").unlink()
    elif corruption == "order":
        meta["feature_names"].reverse()
    elif corruption == "version":
        meta["schema_version"] = -1
    else:
        meta["complete"] = False
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load_artifact(output)


def test_inference_no_truth_zero_candidates_submission_and_determinism(trained):
    data, model_path, _ = trained
    # Make every training file unavailable, not just ground truth.
    for key, path in data["mapping"].items():
        if key.startswith("train"):
            Path(path).unlink()
    first = infer(
        data["features"],
        model_path,
        data["root"] / "output",
        "first",
        data["data_paths"],
        batch_size=7,
    )
    second = infer(
        data["features"],
        model_path,
        data["root"] / "output",
        "second",
        data["data_paths"],
        batch_size=17,
    )
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    matches = pd.read_csv(first / "matching_results.tsv", sep="\t", keep_default_na=False)
    candidates = pd.read_csv(first / "candidate_pairs.tsv", sep="\t", keep_default_na=False)
    assert set(matches.source1_entity_id) == set(data["truth"])
    assert matches.loc[matches.source1_entity_id == "S1-empty", "matched_entity_ids"].item() == ""
    for (_, match), (_, candidate) in zip(matches.iterrows(), candidates.iterrows(), strict=True):
        m = set(filter(None, match.matched_entity_ids.split(",")))
        c = set(filter(None, candidate.candidate_entity_ids.split(",")))
        assert m <= c
        assert all(t.startswith(("S2-", "S3-")) for t in c)
    with pytest.raises(ValueError, match="already exists"):
        infer(data["features"], model_path, data["root"] / "output", "first", data["data_paths"])


def test_empty_candidates_inference(trained):
    data, model_path, _ = trained
    empty = write_features(data["root"] / "empty_features", [])
    result = infer(empty, model_path, data["root"] / "output", "empty", data["data_paths"])
    frame = pd.read_csv(result / "matching_results.tsv", sep="\t", keep_default_na=False)
    assert len(frame) == len(data["truth"])
    assert frame.matched_entity_ids.eq("").all()


@pytest.mark.parametrize("change", ["dtype", "missing_column", "extra_column"])
def test_parquet_schema_contract(fixture_data, change):
    data = fixture_data
    path = next(data["features"].glob("*.parquet"))
    table = pq.read_table(path)
    if change == "dtype":
        index = table.column_names.index("retrieval_score")
        table = table.set_column(index, "retrieval_score", table[index].cast(pa.string()))
    elif change == "missing_column":
        table = table.drop(["retrieval_score"])
    else:
        table = table.append_column("label", pa.array([0] * len(table)))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="schema"):
        feature_inputs(data["features"], "train")


def test_incomplete_feature_manifest(fixture_data):
    root = fixture_data["features"]
    (root / "manifest.json").write_text(json.dumps({"complete": False}))
    with pytest.raises(ValueError, match="incomplete"):
        feature_inputs(root, "train")


def test_corrupt_resume_checkpoint(trained):
    data, output, _ = trained
    work = output.with_name(output.name + ".work")
    output.rename(work)
    (work / "fold-0.cbm").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt fold"):
        train(
            data["features"],
            output,
            data["config_path"],
            data["data_paths"],
            smoke=True,
            resume=True,
        )


def test_training_deterministic_across_shards(trained):
    data, original, _ = trained
    other = write_features(data["root"] / "resharded", list(reversed(data["rows"])), 23)
    output = data["root"] / "other_model"
    train(other, output, data["config_path"], data["data_paths"], smoke=True)
    frame = pa.Table.from_pylist(data["rows"], schema=OUTPUT_SCHEMA).to_pandas()
    first, first_meta = load_artifact(original)
    second, second_meta = load_artifact(output)
    np.testing.assert_array_equal(predict(first, frame), predict(second, frame))
    assert first_meta["decoder"] == second_meta["decoder"]
    assert first_meta["oof_summary"] == second_meta["oof_summary"]
    for fold in range(2):
        pd.testing.assert_frame_equal(
            pd.read_parquet(original / f"oof-{fold}.parquet"),
            pd.read_parquet(output / f"oof-{fold}.parquet"),
        )
