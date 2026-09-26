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
    
    Returns:
        fold_assignment: Dict mapping S1 ID to fold index (0 to n_splits-1)
        diagnostics: DataFrame showing distribution of strata across folds
    """
    s1_ids = sorted(list(ground_truth.keys()))
    strata = []
    
    for s1 in s1_ids:
        truth = ground_truth[s1]
        card = _get_cardinality_label(truth)
        comp = _get_composition_label(truth)
        country = countries.get(s1, "unknown") if countries else "unknown"
        strata.append(f"{country}_{card}_{comp}")
        
    # We use StratifiedKFold to handle stratification
    # If a stratum has fewer members than n_splits, StratifiedKFold might warn/error.
    # To avoid errors with very small strata, we group small strata.
    strata_counts = pd.Series(strata).value_counts()
    valid_strata = [s if strata_counts[s] >= n_splits else "rare" for s in strata]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_assignment = {}
    fold_labels = []
    
    # StratifiedKFold splits indices
    for s1 in s1_ids:
        fold_assignment[s1] = -1 # Placeholder
        
    X = np.zeros(len(s1_ids))
    y = np.array(valid_strata)
    
    # We want to assign each S1 to exactly one validation fold
    for fold_idx, (_train_idx, val_idx) in enumerate(skf.split(X, y)):
        for idx in val_idx:
            fold_assignment[s1_ids[idx]] = fold_idx
            
    for s1 in s1_ids:
        fold_labels.append(fold_assignment[s1])
            
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
    format: str = "parquet"
) -> None:
    """
    Save fold assignments and metadata to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    records = [{"s1_id": k, "fold": v} for k, v in fold_assignment.items()]
    df = pd.DataFrame(records)
    
    if format == "parquet":
        df.to_parquet(output_dir / "folds.parquet", index=False)
    else:
        df.to_csv(output_dir / "folds.tsv", sep="\t", index=False)
        
    metadata = {
        "n_entities": len(fold_assignment),
        "n_folds": df["fold"].nunique(),
        "format": format
    }
    
    with open(output_dir / "folds_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
