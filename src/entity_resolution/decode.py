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
        if score_margin is not None and not enforce_target_ownership:
            raise ValueError("score_margin requires enforce_target_ownership")

        self.global_threshold = global_threshold
        self.s2_threshold = s2_threshold
        self.s3_threshold = s3_threshold
        self.top_k_per_source = top_k_per_source
        self.score_margin = score_margin
        self.enforce_target_ownership = enforce_target_ownership
        self.min_evidence_score = min_evidence_score

    def decode(self, candidates: pd.DataFrame, all_s1_ids: set[str]) -> dict[str, list[str]]:
        required_cols = {"s1_id", "target_id", "target_source", "score"}
        missing = required_cols - set(candidates.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if not pd.api.types.is_numeric_dtype(candidates["score"]):
            raise ValueError("Scores must be numeric.")
        if not np.isfinite(candidates["score"]).all():
            raise ValueError("Scores must be finite.")

        s1_mask = candidates["s1_id"].isin(all_s1_ids)
        if not s1_mask.all():
            raise ValueError("Unknown S1 ID in candidates.")

        src_mask = candidates["target_source"].isin({"S2", "S3"})
        if not src_mask.all():
            raise ValueError("Invalid target_source.")

        prefix_ok = (
            ((candidates["target_source"] == "S2") &
             candidates["target_id"].str.startswith("S2-")) |
            ((candidates["target_source"] == "S3") &
             candidates["target_id"].str.startswith("S3-"))
        )
        if not prefix_ok.all():
            raise ValueError("target_source disagrees with target_id prefix.")

        # 1. Deduplicate pairs retaining maximum score deterministically
        df = candidates.copy()
        df.sort_values(
            by=["score", "s1_id", "target_id"],
            ascending=[False, True, True],
            inplace=True,
            kind="stable"
        )
        df.drop_duplicates(subset=["s1_id", "target_id"], keep="first", inplace=True)

        # 2. Apply min evidence if provided
        if self.min_evidence_score is not None:
            df = df[df["score"] >= self.min_evidence_score]

        # 3. Apply thresholds
        global_t = self.global_threshold or 0.0
        s2_t = self.s2_threshold if self.s2_threshold is not None else global_t
        s3_t = self.s3_threshold if self.s3_threshold is not None else global_t

        threshold_mask = (
            ((df["target_source"] == "S2") & (df["score"] >= s2_t)) |
            ((df["target_source"] == "S3") & (df["score"] >= s3_t))
        )
        df = df[threshold_mask]

        # 4. Apply top-k per source BEFORE ownership
        if self.top_k_per_source is not None and not df.empty:
            df["_rank"] = df.groupby(["s1_id", "target_source"]).cumcount()
            df = df[df["_rank"] < self.top_k_per_source].drop(columns=["_rank"])

        # 5. Apply target ownership and margin
        if self.enforce_target_ownership and not df.empty:
            df["_owner_rank"] = df.groupby("target_id").cumcount()

            if self.score_margin is not None:
                best = df.loc[
                    df["_owner_rank"].eq(0)
                ].set_index("target_id")["score"]

                second = (
                    df.loc[df["_owner_rank"].eq(1)]
                    .set_index("target_id")["score"]
                    .reindex(best.index)
                )

                margin_ok = second.isna() | (best - second).ge(self.score_margin)

                df_best = df[df["_owner_rank"] == 0].copy()
                valid_targets = margin_ok[margin_ok].index
                df_final = df_best[df_best["target_id"].isin(valid_targets)]
                df_final = df_final.drop(columns=["_owner_rank"])
            else:
                df_final = df[df["_owner_rank"] == 0].drop(columns=["_owner_rank"])
        else:
            df_final = df

        # Group back to dictionary deterministically
        results = {s1: [] for s1 in all_s1_ids}
        if not df_final.empty:
            df_final = df_final.sort_values(by=["s1_id", "target_id"])
            grouped = df_final.groupby("s1_id")["target_id"].apply(list).to_dict()
            for s1, tids in grouped.items():
                results[s1] = tids

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
