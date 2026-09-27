import json
import random

import pandas as pd
import pytest

from entity_resolution.decode import Decoder, search_thresholds
from entity_resolution.folds import create_folds, save_folds
from entity_resolution.ground_truth import GroundTruthParserError, load_ground_truth
from entity_resolution.metrics import (
    compute_per_s1_f0_5,
    evaluate_macro_f0_5_efficient,
    evaluate_macro_f0_5_reference,
)
from entity_resolution.submission import SubmissionError, write_submission


def test_metrics_manual_cases():
    assert compute_per_s1_f0_5(set(), set()) == 1.0
    assert compute_per_s1_f0_5(set(), {"S2-1"}) == 0.0
    assert compute_per_s1_f0_5({"S2-1"}, {"S2-1"}) == 1.0
    assert pytest.approx(compute_per_s1_f0_5({"S2-1", "S3-1"}, {"S2-1"}), 0.0001) == 0.8333
    assert pytest.approx(compute_per_s1_f0_5({"S2-1"}, {"S2-1", "S3-1"}), 0.0001) == 0.555555
    assert pytest.approx(
        compute_per_s1_f0_5({"S2-A", "S2-B"}, {"S2-A", "S2-B", "S3-C"}), 0.0001
    ) == 0.714285


def test_metrics_property_randomized():
    random.seed(42)
    s2_pool = [f"S2-{i}" for i in range(100)]
    s3_pool = [f"S3-{i}" for i in range(100)]
    pool = s2_pool + s3_pool

    for _ in range(1000):
        t_len = random.randint(0, 10)
        p_len = random.randint(0, 10)
        truth = set(random.sample(pool, t_len))
        pred = set(random.sample(pool, p_len))
        gt = {"S1-1": list(truth)}
        pd_ = {"S1-1": list(pred)}
        ref = evaluate_macro_f0_5_reference(gt, pd_)
        eff, _ = evaluate_macro_f0_5_efficient(gt, pd_)
        assert pytest.approx(ref, abs=1e-7) == eff


def test_metrics_strict_mode():
    gt = {"S1-1": []}
    pred = {"S1-1": [], "S1-2": ["S2-1"]}
    with pytest.raises(ValueError, match="unknown keys"):
        evaluate_macro_f0_5_efficient(gt, pred, strict=True)
    with pytest.raises(ValueError, match="unknown keys"):
        evaluate_macro_f0_5_reference(gt, pred, strict=True)


def test_metrics_empty_inputs():
    ref = evaluate_macro_f0_5_reference({}, {})
    eff, diag = evaluate_macro_f0_5_efficient({}, {})
    assert ref == 0.0
    assert eff == 0.0
    assert "tp" in diag
    assert diag["score_distribution_counts"] == [0, 0, 0, 0, 0]


def test_ground_truth_parser(tmp_path):
    tsv = tmp_path / "train_ground_truth.tsv"
    tsv.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\t\nS1-2\tS2-1\nS1-3\tS2-2,S3-2\n")
    gt = load_ground_truth(tsv)
    assert gt == {"S1-1": [], "S1-2": ["S2-1"], "S1-3": ["S2-2", "S3-2"]}

    tsv2 = tmp_path / "dup.tsv"
    tsv2.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\t\nS1-1\tS2-1\n")
    with pytest.raises(GroundTruthParserError):
        load_ground_truth(tsv2)


def test_folds(tmp_path):
    gt = {f"S1-{i}": [] for i in range(2)}

    # n_splits greater than entity count raises ValueError
    with pytest.raises(
        ValueError,
        match="must be between 2 and number of ground_truth S1 entities"
    ):
        create_folds(gt, n_splits=5, random_state=42)

    gt = {f"S1-{i}": [] for i in range(20)}
    fa, diag = create_folds(gt, n_splits=5, random_state=42)
    assert len(fa) == 20
    # verify assignment in 0..4
    assert all(0 <= v <= 4 for v in fa.values())

    save_folds(fa, diag, tmp_path / "folds", seed=42, n_splits=5)
    assert (tmp_path / "folds" / "folds_metadata.json").exists()

    with open(tmp_path / "folds" / "folds_metadata.json") as f:
        meta = json.load(f)
    assert meta["seed"] == 42
    assert meta["n_splits"] == 5
    assert "diagnostics" in meta

    with pytest.raises(ValueError):
        create_folds({}, n_splits=5)

    with pytest.raises(ValueError):
        create_folds(gt, n_splits=1)


def test_decoder():
    all_s1 = {"S1-1", "S1-2", "S1-3", "S1-4", "S1-5"}

    # NaN and infinite scores
    candidates = pd.DataFrame([
        {"s1_id": "S1-1", "target_id": "S2-1", "target_source": "S2", "score": float("nan")},
    ])
    dec = Decoder()
    with pytest.raises(ValueError, match="finite and numeric|finite"):
        dec.decode(candidates, all_s1)

    # Nonnumeric scores
    candidates = pd.DataFrame([
        {"s1_id": "S1-1", "target_id": "S2-1", "target_source": "S2", "score": "abc"},
    ])
    with pytest.raises(ValueError, match="numeric"):
        dec.decode(candidates, all_s1)

    # Missing columns
    candidates = pd.DataFrame([
        {"s1_id": "S1-1", "target_id": "S2-1", "score": 1.0},
    ])
    with pytest.raises(ValueError, match="Missing required columns: {'target_source'}"):
        dec.decode(candidates, all_s1)

    # target_source/prefix disagreement
    candidates = pd.DataFrame([
        {"s1_id": "S1-1", "target_id": "S2-1", "target_source": "S3", "score": 0.5},
    ])
    with pytest.raises(ValueError, match="disagrees"):
        dec.decode(candidates, all_s1)

    # Unknown S1
    candidates = pd.DataFrame([
        {"s1_id": "S1-99", "target_id": "S2-1", "target_source": "S2", "score": 0.5},
    ])
    with pytest.raises(ValueError, match="Unknown S1"):
        dec.decode(candidates, all_s1)

    # Duplicate rows consuming top-k / duplicate candidate-pair rows
    candidates = pd.DataFrame([
        {"s1_id": "S1-1", "target_id": "S2-1", "target_source": "S2", "score": 0.9},
        {"s1_id": "S1-1", "target_id": "S2-1", "target_source": "S2", "score": 0.5}, # Duplicate
        {"s1_id": "S1-1", "target_id": "S2-2", "target_source": "S2", "score": 0.8},
    ])
    dec2 = Decoder(top_k_per_source=1)
    res2 = dec2.decode(candidates, all_s1)
    # The duplicate S2-1 with lower score is dropped.
    # S2-1 is rank 1. S2-2 is rank 2. Top-k=1 -> gets S2-1.
    assert res2["S1-1"] == ["S2-1"]

    # score_margin without ownership raises ValueError
    with pytest.raises(ValueError, match="score_margin requires enforce_target_ownership"):
        Decoder(score_margin=0.1, enforce_target_ownership=False)

    # Ownership-margin rejection
    candidates = pd.DataFrame([
        {"s1_id": "S1-1", "target_id": "S2-2", "target_source": "S2", "score": 0.8},
        {"s1_id": "S1-2", "target_id": "S2-2", "target_source": "S2", "score": 0.7},
    ])
    dec3 = Decoder(enforce_target_ownership=True, score_margin=0.2)
    res3 = dec3.decode(candidates, all_s1)
    # margin=0.2, difference is 0.1 -> NEITHER gets it.
    assert res3["S1-1"] == []

    # If margin is 0.05, S1-1 gets it
    dec4 = Decoder(enforce_target_ownership=True, score_margin=0.05)
    res4 = dec4.decode(candidates, all_s1)
    assert res4["S1-1"] == ["S2-2"]

    # top-k happens before ownership
    # S1-3 has top_k=1. It prefers S2-3 (0.9) over S2-4 (0.8). S2-4 is discarded.
    # S1-4 only wants S2-4 (0.5). S1-4 wins S2-4 because S1-3 discarded it!
    candidates = pd.DataFrame([
        {"s1_id": "S1-3", "target_id": "S2-3", "target_source": "S2", "score": 0.9},
        {"s1_id": "S1-3", "target_id": "S2-4", "target_source": "S2", "score": 0.8},
        {"s1_id": "S1-4", "target_id": "S2-4", "target_source": "S2", "score": 0.5},
    ])
    dec5 = Decoder(top_k_per_source=1, enforce_target_ownership=True)
    res5 = dec5.decode(candidates, all_s1)
    assert res5["S1-3"] == ["S2-3"]
    assert res5["S1-4"] == ["S2-4"]

    # shuffled candidate-row order produces identical decoding
    candidates_large = pd.DataFrame([
        {
            "s1_id": f"S1-{i}",
            "target_id": f"S2-{j}",
            "target_source": "S2",
            "score": random.random()
        }
        for i in range(1, 5) for j in range(1, 10)
    ])
    res_orig = dec5.decode(candidates_large, all_s1)
    res_shuf = dec5.decode(candidates_large.sample(frac=1, random_state=42), all_s1)
    assert res_orig == res_shuf


def test_decoder_tie_breaking_and_empty_search():
    with pytest.raises(ValueError, match="Empty search space"):
        search_thresholds(pd.DataFrame(), {}, [], set())

    all_s1 = {"S1-1", "S1-2"}
    gt = {"S1-1": ["S2-1"], "S1-2": ["S2-1"]}
    # enable ownership with genuinely competing S1 owners
    cand = pd.DataFrame([
        {"s1_id": "S1-1", "target_id": "S2-1", "target_source": "S2", "score": 0.9},
        {"s1_id": "S1-2", "target_id": "S2-1", "target_source": "S2", "score": 0.8},
    ])

    # Tie-breaking logic prefers higher threshold, larger margin, lower k
    space = [
        {"global_threshold": 0.1, "score_margin": 0.05,
         "enforce_target_ownership": True, "top_k_per_source": 5},
        {"global_threshold": 0.5, "score_margin": 0.05,
         "enforce_target_ownership": True, "top_k_per_source": 5},
        {"global_threshold": 0.5, "score_margin": 0.2,
         "enforce_target_ownership": True, "top_k_per_source": 5},
        {"global_threshold": 0.5, "score_margin": 0.2,
         "enforce_target_ownership": True, "top_k_per_source": 2},
    ]

    best_params, best_score, _ = search_thresholds(cand, gt, space, all_s1)
    # The margin=0.05 gives S2-1 to S1-1. S1-1 score=1.0, S1-2 score=0.0. Macro=0.5
    # The margin=0.2 will give S2-1 to neither (0.9-0.8 < 0.2). Both get 0. Macro=0
    # Wait, if margin 0.2 gives 0 score, it WON'T WIN!
    # Let's adjust space so that score is same, to test tie-breaking.
    space_tied = [
        {"global_threshold": 0.1, "score_margin": 0.05,
         "enforce_target_ownership": True, "top_k_per_source": 5},
        {"global_threshold": 0.2, "score_margin": 0.05,
         "enforce_target_ownership": True, "top_k_per_source": 5},
    ]
    best_params2, _, _ = search_thresholds(cand, gt, space_tied, all_s1)
    assert best_params2["global_threshold"] == 0.2


def test_submission_writer(tmp_path):
    all_s1 = {"S1-1", "S1-2"}
    valid_targets = {"S2-1", "S3-1"}
    cand = {"S1-1": ["S2-1", "S3-1"], "S1-2": []}
    match = {"S1-1": ["S2-1"], "S1-2": []}

    out = write_submission(all_s1, valid_targets, cand, match, tmp_path, "run1")
    assert (out / "matching_results.tsv").exists()

    # valid-prefix but nonexistent target ID
    cand2 = {"S1-1": ["S2-99"]}
    match2 = {"S1-1": ["S2-99"]}
    with pytest.raises(SubmissionError, match="Invalid or unknown target ID"):
        write_submission(all_s1, valid_targets, cand2, match2, tmp_path, "run2")
    assert not (tmp_path / "run2").exists() # leaves no run directory

    # duplicate candidate-pair rows canonicalization or rejection
    cand3 = {"S1-1": ["S2-1", "S2-1"]}
    match3 = {"S1-1": ["S2-1"]}
    with pytest.raises(SubmissionError, match="Duplicate candidates"):
        write_submission(all_s1, valid_targets, cand3, match3, tmp_path, "run3")


def test_decoder_ownership_margin():
    all_s1 = {"S1-1", "S1-2", "S1-3"}
    dec = Decoder(enforce_target_ownership=True, score_margin=0.2)

    candidates = pd.DataFrame([
        # 1. A target with only one eligible owner survives when score_margin > 0.
        {"s1_id": "S1-1", "target_id": "S2-1", "target_source": "S2", "score": 0.9},

        # 2. A target with two owners is rejected when gap < margin.
        {"s1_id": "S1-2", "target_id": "S2-2", "target_source": "S2", "score": 0.8},
        {"s1_id": "S1-3", "target_id": "S2-2", "target_source": "S2", "score": 0.7},

        # 3. A target with two owners is retained for the higher-scoring owner when gap >= margin.
        {"s1_id": "S1-2", "target_id": "S2-3", "target_source": "S2", "score": 0.9},
        {"s1_id": "S1-3", "target_id": "S2-3", "target_source": "S2", "score": 0.6},
    ])

    res = dec.decode(candidates, all_s1)

    assert "S2-1" in res["S1-1"]
    assert "S2-2" not in res["S1-2"] and "S2-2" not in res["S1-3"]
    assert "S2-3" in res["S1-2"] and "S2-3" not in res["S1-3"]

    # 4. Result remains identical after shuffling the input candidate rows.
    res_shuffled = dec.decode(candidates.sample(frac=1, random_state=42), all_s1)
    assert res == res_shuffled

    # Binary-exact equality
    dec_exact = Decoder(enforce_target_ownership=True, score_margin=0.25)
    cand_exact = pd.DataFrame([
        {"s1_id": "S1-1", "target_id": "S2-4", "target_source": "S2", "score": 0.75},
        {"s1_id": "S1-2", "target_id": "S2-4", "target_source": "S2", "score": 0.50},
    ])
    res_exact = dec_exact.decode(cand_exact, all_s1)
    assert "S2-4" in res_exact["S1-1"]

    # Decimal floating-point equality
    dec_dec = Decoder(enforce_target_ownership=True, score_margin=0.20)
    cand_dec = pd.DataFrame([
        {"s1_id": "S1-1", "target_id": "S2-5", "target_source": "S2", "score": 0.70},
        {"s1_id": "S1-2", "target_id": "S2-5", "target_source": "S2", "score": 0.50},
    ])
    res_dec = dec_dec.decode(cand_dec, all_s1)
    assert "S2-5" in res_dec["S1-1"]
