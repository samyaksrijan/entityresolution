from typing import Any


def compute_per_s1_f0_5(truth: set[str], pred: set[str]) -> float:
    if not truth and not pred:
        return 1.0
    if not truth and pred:
        return 0.0
    
    tp = len(truth & pred)
    if tp == 0:
        return 0.0
        
    precision = tp / len(pred)
    recall = tp / len(truth)
    
    return 1.25 * precision * recall / (0.25 * precision + recall)


def evaluate_macro_f0_5_reference(
    ground_truth: dict[str, list[str]],
    predictions: dict[str, list[str]],
    strict: bool = False
) -> float:
    if not ground_truth:
        return 0.0
        
    if strict:
        extra_keys = set(predictions.keys()) - set(ground_truth.keys())
        if extra_keys:
            raise ValueError(f"Predictions contain unknown keys, e.g.: {list(extra_keys)[:3]}")
            
    total_score = 0.0
    for s1_id, truth_list in ground_truth.items():
        truth_set = set(truth_list)
        pred_set = set(predictions.get(s1_id, []))
        total_score += compute_per_s1_f0_5(truth_set, pred_set)
        
    return total_score / len(ground_truth)


def evaluate_macro_f0_5_efficient(
    ground_truth: dict[str, list[str]],
    predictions: dict[str, list[str]],
    strict: bool = False
) -> tuple[float, dict[str, Any]]:
    """
    Efficient implementation suitable for millions of S1 entities.
    Also returns diagnostics.
    """
    if strict and ground_truth:
        extra_keys = set(predictions.keys()) - set(ground_truth.keys())
        if extra_keys:
            raise ValueError(f"Predictions contain unknown keys, e.g.: {list(extra_keys)[:3]}")
            
    empty_diagnostics = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "predicted_pairs": 0,
        "truth_pairs": 0,
        "no_match_entities": 0,
        "false_positives_on_no_match_entities": 0,
        "score_distribution_counts": [0, 0, 0, 0, 0],
        "score_distribution_bins": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    }
    
    if not ground_truth:
        return 0.0, empty_diagnostics
        
    total_score = 0.0
    
    # Diagnostics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_pred_pairs = 0
    total_truth_pairs = 0
    no_match_entities = 0
    fp_on_no_match = 0
    
    hist = [0, 0, 0, 0, 0]
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    for s1_id, truth_list in ground_truth.items():
        truth_set = set(truth_list)
        pred_set = set(predictions.get(s1_id, []))
        
        t_len = len(truth_set)
        p_len = len(pred_set)
        
        total_truth_pairs += t_len
        total_pred_pairs += p_len
        
        if t_len == 0:
            no_match_entities += 1
            if p_len == 0:
                score = 1.0
            else:
                score = 0.0
                total_fp += p_len
                fp_on_no_match += p_len
        else:
            tp = len(truth_set & pred_set)
            fp = p_len - tp
            fn = t_len - tp
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            if tp == 0:
                score = 0.0
            else:
                precision = tp / p_len
                recall = tp / t_len
                score = 1.25 * precision * recall / (0.25 * precision + recall)
                
        total_score += score
        
        # Incrementally build distribution
        for i in range(5):
            if i < 4 and score < bins[i+1] or i == 4 and score <= bins[i+1]:
                hist[i] += 1
                break
        
    macro_f0_5 = total_score / len(ground_truth)
    
    diagnostics = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "predicted_pairs": total_pred_pairs,
        "truth_pairs": total_truth_pairs,
        "no_match_entities": no_match_entities,
        "false_positives_on_no_match_entities": fp_on_no_match,
        "score_distribution_counts": hist,
        "score_distribution_bins": bins,
    }
    
    return macro_f0_5, diagnostics
