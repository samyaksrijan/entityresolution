import numpy as np
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
        if top_k_per_source is not None and top_k_per_source < 1:
            raise ValueError("top_k_per_source must be >= 1")
        if score_margin is not None and score_margin < 0:
            raise ValueError("score_margin must be >= 0")
            
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
        required_cols = {"s1_id", "target_id", "target_source", "score"}
        if not required_cols.issubset(candidates.columns):
            raise ValueError(f"Missing required columns: {required_cols}")
            
        if not np.isfinite(candidates["score"]).all():
            raise ValueError("Scores must be finite.")
            
        for _, row in candidates.iterrows():
            if row["s1_id"] not in all_s1_ids:
                raise ValueError(f"Unknown S1 ID: {row['s1_id']}")
            src = row["target_source"]
            if src not in ("S2", "S3"):
                raise ValueError(f"Invalid target_source: {src}")
            if not row["target_id"].startswith(f"{src}-"):
                raise ValueError(f"target_source {src} disagrees with target_id {row['target_id']}")

        df = candidates.copy()
        df.sort_values(
            by=["score", "s1_id", "target_id"], ascending=[False, True, True], inplace=True
        )
        
        # Deduplicate (s1_id, target_id) retaining the maximum score deterministically
        df.drop_duplicates(subset=["s1_id", "target_id"], keep="first", inplace=True)
        
        # Apply min evidence if provided
        if self.min_evidence_score is not None:
            df = df[df["score"] >= self.min_evidence_score]

        # Apply thresholds
        valid_rows = []
        for _, row in df.iterrows():
            if row["score"] >= self._get_threshold(row["target_source"]):
                valid_rows.append(row)
                
        if not valid_rows:
            return {s1: [] for s1 in all_s1_ids}
            
        filtered_df = pd.DataFrame(valid_rows)
        
        # Apply top-k per source BEFORE ownership
        selected_after_topk = []
        for _, group in filtered_df.groupby("s1_id"):
            for source in ["S2", "S3"]:
                source_group = group[group["target_source"] == source]
                if source_group.empty:
                    continue
                k = self.top_k_per_source
                if k is None:
                    k = len(source_group)
                selected_after_topk.append(source_group.head(k))
                
        if not selected_after_topk:
            return {s1: [] for s1 in all_s1_ids}
            
        df_topk = pd.concat(selected_after_topk)
        
        # Apply target ownership and margin
        if self.enforce_target_ownership:
            final_rows = []
            for _, group in df_topk.groupby("target_id"):
                group = group.sort_values(by=["score", "s1_id"], ascending=[False, True])
                if len(group) == 1:
                    final_rows.append(group.iloc[0])
                else:
                    best_score = group.iloc[0]["score"]
                    second_best = group.iloc[1]["score"]
                    if self.score_margin is not None:
                        if best_score - second_best >= self.score_margin:
                            final_rows.append(group.iloc[0])
                    else:
                        final_rows.append(group.iloc[0])
            if final_rows:
                df_final = pd.DataFrame(final_rows)
            else:
                df_final = pd.DataFrame(columns=df_topk.columns)
        else:
            df_final = df_topk
            
        results = {s1: [] for s1 in all_s1_ids}
        for _, row in df_final.iterrows():
            results[row["s1_id"]].append(row["target_id"])
            
        for s1 in results:
            results[s1] = sorted(list(set(results[s1])))
            
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
    
    if not search_space:
        raise ValueError("Empty search space")
        
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
        
        is_better = False
        if score > best_score + 1e-9:
            is_better = True
        elif abs(score - best_score) <= 1e-9 and best_params is not None:
            # Deterministic tie-breaking: prefer higher thresholds, larger margin, lower top-k
            def _get_val(p, key, default=0.0):
                v = p.get(key)
                return v if v is not None else default
                
            curr_g = _get_val(params, "global_threshold")
            best_g = _get_val(best_params, "global_threshold")
            if curr_g > best_g:
                is_better = True
            elif curr_g == best_g:
                curr_s2 = _get_val(params, "s2_threshold")
                best_s2 = _get_val(best_params, "s2_threshold")
                if curr_s2 > best_s2:
                    is_better = True
                elif curr_s2 == best_s2:
                    curr_s3 = _get_val(params, "s3_threshold")
                    best_s3 = _get_val(best_params, "s3_threshold")
                    if curr_s3 > best_s3:
                        is_better = True
                    elif curr_s3 == best_s3:
                        curr_margin = _get_val(params, "score_margin")
                        best_margin = _get_val(best_params, "score_margin")
                        if curr_margin > best_margin:
                            is_better = True
                        elif curr_margin == best_margin:
                            curr_k = _get_val(params, "top_k_per_source", 999999)
                            best_k = _get_val(best_params, "top_k_per_source", 999999)
                            if curr_k < best_k:
                                is_better = True
                
        if is_better or best_params is None:
            best_score = score
            best_params = params.copy()
            
    return best_params, best_score, pd.DataFrame(results)
