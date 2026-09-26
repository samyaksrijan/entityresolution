
import pandas as pd


class Decoder:
    def __init__(
        self,
        global_threshold: float | None = None,
        s2_threshold: float | None = None,
        s3_threshold: float | None = None,
        top_k_per_source: int | None = None,
        score_margin: float | None = None,
        enforce_target_ownership: bool = False,
        min_evidence_score: float | None = None,
    ):
        self.global_threshold = global_threshold
        self.s2_threshold = s2_threshold
        self.s3_threshold = s3_threshold
        self.top_k_per_source = top_k_per_source
        self.score_margin = score_margin
        self.enforce_target_ownership = enforce_target_ownership
        self.min_evidence_score = min_evidence_score

    def _get_threshold(self, source: str) -> float:
        if source == "S2":
            return self.s2_threshold if self.s2_threshold is not None else (
                self.global_threshold or 0.0)
        elif source == "S3":
            return self.s3_threshold if self.s3_threshold is not None else (
                self.global_threshold or 0.0)
        return self.global_threshold or 0.0

    def decode(self, candidates: pd.DataFrame, all_s1_ids: set[str]) -> dict[str, list[str]]:
        """
        Candidates DataFrame must have: s1_id, target_id, target_source, score
        """
        # Ensure determinism: sort by score descending, then s1_id ascending, target_id ascending
        df = candidates.copy()
        df.sort_values(
            by=["score", "s1_id", "target_id"], ascending=[False, True, True], inplace=True
        )
        
        # Apply min evidence if provided
        if self.min_evidence_score is not None:
            df = df[df["score"] >= self.min_evidence_score]

        # Apply target ownership
        if self.enforce_target_ownership:
            df = df.drop_duplicates(subset=["target_id"], keep="first")
            
        # Apply thresholds
        valid_rows = []
        for _, row in df.iterrows():
            if row["score"] >= self._get_threshold(row["target_source"]):
                valid_rows.append(row)
                
        if not valid_rows:
            return {s1: [] for s1 in all_s1_ids}
            
        filtered_df = pd.DataFrame(valid_rows)
        
        # Apply top-k and margin
        results = {s1: [] for s1 in all_s1_ids}
        
        for s1_id, group in filtered_df.groupby("s1_id"):
            for source in ["S2", "S3"]:
                source_group = group[group["target_source"] == source]
                if source_group.empty:
                    continue
                    
                scores = source_group["score"].tolist()
                targets = source_group["target_id"].tolist()
                
                k = self.top_k_per_source if self.top_k_per_source is not None else len(targets)
                
                # Apply margin
                selected_targets = []
                for i in range(min(k, len(targets))):
                    if (self.score_margin is not None and i + 1 < len(scores) 
                            and scores[i] - scores[i+1] < self.score_margin):
                        break # margin violated, stop taking more
                    selected_targets.append(targets[i])
                    
                results[s1_id].extend(selected_targets)
                
        # Deterministic output order
        for s1 in results:
            results[s1] = sorted(results[s1])
            
        return results

def search_thresholds(
    candidates: pd.DataFrame,
    ground_truth: dict[str, list[str]],
    search_space: list[dict[str, float]],
    all_s1_ids: set[str]
) -> tuple[dict[str, float], float, pd.DataFrame]:
    """
    Search over combinations of thresholds and parameters to maximize macro F0.5.
    Returns (best_params, best_score, diagnostics_df)
    """
    from entity_resolution.metrics import evaluate_macro_f0_5_efficient
    
    results = []
    best_score = -1.0
    best_params = None
    
    for params in search_space:
        decoder = Decoder(**params)
        predictions = decoder.decode(candidates, all_s1_ids)
        score, _ = evaluate_macro_f0_5_efficient(ground_truth, predictions)
        
        results.append({
            **params,
            "score": score
        })
        
        # Deterministic tie-breaking:
        # prefer more conservative (higher thresholds, higher margin, lower k)
        is_better = False
        if score > best_score + 1e-9:
            is_better = True
        elif abs(score - best_score) <= 1e-9 and best_params is not None:
            # Tie breaking logic
            curr_global = params.get("global_threshold") or 0.0
            best_global = best_params.get("global_threshold") or 0.0
            if curr_global > best_global:
                is_better = True
            elif curr_global == best_global:
                curr_margin = params.get("score_margin") or 0.0
                best_margin = best_params.get("score_margin") or 0.0
                if curr_margin > best_margin:
                    is_better = True
                
        if is_better or best_params is None:
            best_score = score
            best_params = params.copy()
            
    return best_params, best_score, pd.DataFrame(results)
