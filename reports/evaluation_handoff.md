# Evaluation Handoff

## 1. Branch Name
`lane-evaluation`

## 2. Changed Files
- `src/entity_resolution/ground_truth.py` (new)
- `src/entity_resolution/metrics.py` (new)
- `src/entity_resolution/folds.py` (new)
- `src/entity_resolution/decode.py` (new)
- `src/entity_resolution/submission.py` (new)
- `tests/test_evaluation.py` (new)

## 3. Test and Lint Results
- `uv run pytest tests/test_evaluation.py`: 6 tests passed.
- `uv run ruff check .`: All checks passed.

## 4. Five Manually Calculated Metric Examples

1. **Empty truth and empty prediction**
   - Truth: `[]`, Prediction: `[]`
   - Calculation: True negative singleton case, F0.5 = 1.0.
   - Computed Output: 1.0

2. **Empty truth and false-positive prediction**
   - Truth: `[]`, Prediction: `["S2-1"]`
   - Calculation: False positive on singleton case, F0.5 = 0.0.
   - Computed Output: 0.0

3. **One true match**
   - Truth: `["S2-1"]`, Prediction: `["S2-1"]`
   - Calculation: TP=1, FP=0, FN=0. Precision=1.0, Recall=1.0. F0.5 = 1.0.
   - Computed Output: 1.0

4. **Partial multi-match recall**
   - Truth: `["S2-1", "S3-1"]`, Prediction: `["S2-1"]`
   - Calculation: TP=1, FP=0, FN=1. Precision=1.0, Recall=0.5. F0.5 = 1.25 * 1.0 * 0.5 / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.8333.
   - Computed Output: 0.8333

5. **Extra false positives**
   - Truth: `["S2-1"]`, Prediction: `["S2-1", "S3-1"]`
   - Calculation: TP=1, FP=1, FN=0. Precision=0.5, Recall=1.0. F0.5 = 1.25 * 0.5 * 1.0 / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 0.5555.
   - Computed Output: 0.5555

## 5. Fold Distribution Checks
- The S1-level fold assignment stratifies across combinations of country, truth cardinality (0, 1, 2, 3, 4+), and target-source composition (no match, S2 only, S3 only, both).
- We use `StratifiedKFold` which handles generating distributions that mirror the dataset. `folds.py` generates distribution diagnostics broken down by fold and stratum. Small strata that are less than `n_splits` size are grouped into a "rare" stratum bucket to prevent stratification failures.

## 6. Decoder Behavior and Tie-Breaking
- **Threshold Search**: Tie-breaking prefers more conservative settings if exact matches on the F0.5 score exist up to `1e-9`. Specifically, it favors higher global thresholds, and then higher margins.
- **Decoding**: We apply multiple deterministic filters:
  - We sort initial candidates by `score` (descending), then `s1_id` (ascending), then `target_id` (ascending) to guarantee determinism.
  - Optional `enforce_target_ownership` keeps only the highest-scoring S1 per target.
  - We apply source-specific thresholds, top-K selection, and a score margin.

## 7. Unresolved Organizer-Format Assumptions
- "minimum-evidence rule" vs "global score threshold": The decoder allows both a `min_evidence_score` that drops candidates globally prior to target ownership logic, and specific `global/s2/s3_threshold`s.
- Score margin definition: It was implemented as a simple "distance to next highest score" threshold when picking top-K matches per source.
- Target ownership: Implemented as greedy assignment strictly sorted by score.

## 8. Confirmations
Raw data, ingestion code (`io.py`), normalization, profiling, blocking, configs (`configs/data_paths.yaml`), existing profiling reports, and existing tests were not modified.
