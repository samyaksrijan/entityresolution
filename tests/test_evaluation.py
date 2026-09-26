
import pandas as pd
import pytest

from entity_resolution.decode import Decoder
from entity_resolution.folds import create_folds
from entity_resolution.ground_truth import GroundTruthParserError, load_ground_truth
from entity_resolution.metrics import (
    compute_per_s1_f0_5,
    evaluate_macro_f0_5_efficient,
    evaluate_macro_f0_5_reference,
)
from entity_resolution.submission import SubmissionError, write_submission


def test_metrics_manual_cases():
    # 1. empty truth and empty prediction
    assert compute_per_s1_f0_5(set(), set()) == 1.0
    
    # 2. empty truth and false-positive prediction
    assert compute_per_s1_f0_5(set(), {"S2-1"}) == 0.0
    
    # 3. one true match
    assert compute_per_s1_f0_5({"S2-1"}, {"S2-1"}) == 1.0
    
    # 4. partial multi-match recall (truth: 2, pred: 1, tp: 1)
    # precision = 1.0, recall = 0.5
    # F0.5 = 1.25 * 1 * 0.5 / (0.25 * 1 + 0.5) = 0.625 / 0.75 = 0.8333...
    assert pytest.approx(compute_per_s1_f0_5({"S2-1", "S3-1"}, {"S2-1"}), 0.0001) == 0.8333
    
    # 5. extra false positives (truth: 1, pred: 2, tp: 1)
    # precision = 0.5, recall = 1.0
    # F0.5 = 1.25 * 0.5 * 1 / (0.25 * 0.5 + 1) = 0.625 / 1.125 = 0.5555...
    assert pytest.approx(compute_per_s1_f0_5({"S2-1"}, {"S2-1", "S3-1"}), 0.0001) == 0.555555
    
def test_metrics_equivalence():
    gt = {
        "S1-1": [],
        "S1-2": [],
        "S1-3": ["S2-1"],
        "S1-4": ["S2-2", "S3-2"],
        "S1-5": ["S3-3"]
    }
    
    pred = {
        "S1-1": [],
        "S1-2": ["S2-99"],
        "S1-3": ["S2-1"],
        "S1-4": ["S2-2"],
        "S1-5": ["S3-3", "S2-88"]
    }
    
    ref_score = evaluate_macro_f0_5_reference(gt, pred)
    eff_score, diag = evaluate_macro_f0_5_efficient(gt, pred)
    
    assert pytest.approx(ref_score, 1e-6) == eff_score
    assert diag["tp"] == 3
    assert diag["fp"] == 2
    assert diag["fn"] == 1
    assert diag["no_match_entities"] == 2
    assert diag["false_positives_on_no_match_entities"] == 1

def test_ground_truth_parser(tmp_path):
    tsv = tmp_path / "train_ground_truth.tsv"
    tsv.write_text("source1_entity_id\tmatched_entity_ids\n"
                   "S1-1\t\n"
                   "S1-2\tS2-1\n"
                   "S1-3\tS2-2,S3-2\n")
    
    gt = load_ground_truth(tsv)
    assert gt == {
        "S1-1": [],
        "S1-2": ["S2-1"],
        "S1-3": ["S2-2", "S3-2"]
    }
    
    # duplicates test
    tsv2 = tmp_path / "dup.tsv"
    tsv2.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\t\nS1-1\tS2-1\n")
    with pytest.raises(GroundTruthParserError):
        load_ground_truth(tsv2)

def test_folds():
    gt = {f"S1-{i}": [] for i in range(20)}
    for i in range(20, 40):
        gt[f"S1-{i}"] = ["S2-1"]
        
    fa, diag = create_folds(gt, n_splits=5, random_state=42)
    assert len(fa) == 40
    
    # no entity in multiple folds implicitly checked because dict only maps to one
    
    # Leakage test: check sizes roughly equal
    for fold in range(5):
        assert sum(1 for v in fa.values() if v == fold) == 8

def test_decoder():
    candidates = pd.DataFrame([
        {"s1_id": "S1-1", "target_id": "S2-1", "target_source": "S2", "score": 0.9},
        {"s1_id": "S1-1", "target_id": "S3-1", "target_source": "S3", "score": 0.8},
        {"s1_id": "S1-2", "target_id": "S2-2", "target_source": "S2", "score": 0.4},
        {"s1_id": "S1-3", "target_id": "S3-2", "target_source": "S3", "score": 0.9},
        {"s1_id": "S1-4", "target_id": "S3-2", "target_source": "S3", "score": 0.95}, # conflict
    ])
    
    all_s1 = {"S1-1", "S1-2", "S1-3", "S1-4", "S1-5"} # S1-5 has no candidates
    
    # Global threshold
    dec = Decoder(global_threshold=0.5)
    res = dec.decode(candidates, all_s1)
    assert res["S1-1"] == ["S2-1", "S3-1"]
    assert res["S1-2"] == []
    assert res["S1-5"] == []
    
    # Ownership
    dec2 = Decoder(global_threshold=0.5, enforce_target_ownership=True)
    res2 = dec2.decode(candidates, all_s1)
    # S3-2 should go to S1-4 because it has 0.95 vs 0.9
    assert res2["S1-4"] == ["S3-2"]
    assert res2["S1-3"] == []
    
    # Top K
    dec3 = Decoder(global_threshold=0.0, top_k_per_source=1)
    # Wait, S1-1 has S2-1 and S3-1, both are top 1 for their sources!
    res3 = dec3.decode(candidates, all_s1)
    assert "S2-1" in res3["S1-1"] and "S3-1" in res3["S1-1"]

def test_submission_writer(tmp_path):
    all_s1 = {"S1-1", "S1-2"}
    cand = {"S1-1": ["S2-1", "S3-1"], "S1-2": ["S2-2"]}
    match = {"S1-1": ["S2-1"], "S1-2": []}
    
    out = write_submission(all_s1, cand, match, tmp_path, "run1")
    assert (out / "matching_results.tsv").exists()
    assert (out / "candidate_pairs.tsv").exists()
    
    # subset error
    match2 = {"S1-1": ["S2-99"], "S1-2": []}
    with pytest.raises(SubmissionError):
        write_submission(all_s1, cand, match2, tmp_path, "run2")
