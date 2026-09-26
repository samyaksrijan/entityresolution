import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _get_cardinality_label(truth_list: list[str]) -> str:
    n = len(truth_list)
    if n >= 4:
        return "4+"
    return str(n)


def _get_composition_label(truth_list: list[str]) -> str:
    if not truth_list:
        return "no_match"
    has_s2 = any(x.startswith("S2-") for x in truth_list)
    has_s3 = any(x.startswith("S3-") for x in truth_list)
    if has_s2 and has_s3:
        return "both"
    elif has_s2:
        return "s2_only"
    elif has_s3:
        return "s3_only"
    return "unknown"


def create_folds(
    ground_truth: dict[str, list[str]],
    countries: dict[str, str] | None = None,
    n_splits: int = 5,
    random_state: int = 42
) -> tuple[dict[str, int], pd.DataFrame]:
    """
    Create deterministic S1-level fold assignment.
    """
    if not ground_truth:
        raise ValueError("Cannot create folds for empty ground truth.")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
        
    s1_ids = sorted(list(ground_truth.keys()))
    strata = []
    
    for s1 in s1_ids:
        truth = ground_truth[s1]
        card = _get_cardinality_label(truth)
        comp = _get_composition_label(truth)
        country = countries.get(s1, "unknown") if countries else "unknown"
        strata.append(f"{country}_{card}_{comp}")
        
    strata_counts = pd.Series(strata).value_counts()
    
    valid_s1_ids = []
    valid_strata = []
    rare_s1_ids = []
    
    for s1, s in zip(s1_ids, strata, strict=True):
        if strata_counts[s] >= n_splits:
            valid_s1_ids.append(s1)
            valid_strata.append(s)
        else:
            rare_s1_ids.append(s1)

    fold_assignment = {}
    
    if valid_s1_ids:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        X = np.zeros(len(valid_s1_ids))
        for fold_idx, (_, val_idx) in enumerate(skf.split(X, valid_strata)):
            for idx in val_idx:
                fold_assignment[valid_s1_ids[idx]] = fold_idx
                
    # Deterministic round-robin assignment for rare strata
    # since rare_s1_ids is already sorted
    for i, s1 in enumerate(rare_s1_ids):
        fold_assignment[s1] = i % n_splits
            
    fold_labels = [fold_assignment[s1] for s1 in s1_ids]
            
    df = pd.DataFrame({
        "s1_id": s1_ids,
        "stratum": strata,
        "fold": fold_labels
    })
    
    diagnostics = df.groupby(["fold", "stratum"]).size().unstack(fill_value=0)
    
    return fold_assignment, diagnostics


def save_folds(
    fold_assignment: dict[str, int],
    diagnostics: pd.DataFrame,
    output_dir: Path,
    seed: int,
    n_splits: int,
    format: str = "parquet"
) -> None:
    """
    Save fold assignments and metadata to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # deterministic order
    sorted_s1_ids = sorted(list(fold_assignment.keys()))
    records = [{"s1_id": s1, "fold": fold_assignment[s1]} for s1 in sorted_s1_ids]
    
    df = pd.DataFrame(records)
    
    if format == "parquet":
        df.to_parquet(output_dir / "folds.parquet", index=False)
    else:
        df.to_csv(output_dir / "folds.tsv", sep="\t", index=False)
        
    metadata = {
        "strategy_version": "1.0",
        "n_entities": len(fold_assignment),
        "n_splits": n_splits,
        "seed": seed,
        "format": format,
        "stratum_definitions": ["country", "cardinality", "composition"],
        "diagnostics": diagnostics.to_dict(orient="index")
    }
    
    with open(output_dir / "folds_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
