# Evaluation Handoff

## 1. Objective and Scope
The objective was to perform a final audit and ensure the exact implementation of the evaluation and decoding lane for the business entity-resolution competition, complying with all organizer rules. The scope is strictly limited to the `src/entity_resolution` evaluation modules and related tests.

## 2. Branch Name
`lane-evaluation`

## 3. Base Commit
`master` branch pointer at the time of branching (unmodified).

## 4. Final Commit
The commit hash will be generated upon the final commit of this audit.

## 5. Complete Changed-File List
- `reports/evaluation_handoff.md`
- `src/entity_resolution/decode.py`
- `src/entity_resolution/folds.py`
- `src/entity_resolution/ground_truth.py`
- `src/entity_resolution/metrics.py`
- `src/entity_resolution/submission.py`
- `tests/test_evaluation.py`

No raw data, ingestion, normalization, profiling, blocking, or config files were modified.

## 6. Module-by-Module API Summary
- **`ground_truth.py`**: Exports `load_ground_truth(path)` which reads S1 truth edges from TSV, strictly enforcing S1/S2/S3 prefixes and returning a canonical dict of sorted target ID lists per S1. Singletons are preserved.
- **`metrics.py`**: Exports `evaluate_macro_f0_5_reference` (simple reference) and `evaluate_macro_f0_5_efficient` (fast loop with diagnostics). Uses `compute_per_s1_f0_5`.
- **`folds.py`**: Exports `create_folds` which deterministically creates S1-level Stratified K-Folds based on cardinality and composition. Exports `save_folds` to persist assignments.
- **`decode.py`**: Exports `Decoder` class configuring specific thresholds, top-k filters, and target ownership. Exports `search_thresholds` to find optimal thresholds maximizing F0.5.
- **`submission.py`**: Exports `write_submission` enforcing validation on target constraints, testing S1 keys, and saving identically formatted `matching_results.tsv` and `candidate_pairs.tsv` to isolated run IDs.

## 7. Metric Semantics
- Uses beta=0.5.
- Each S1 is scored independently and macro-averaged across all S1s.
- True empty S1s correctly output 1.0 if predicted empty and 0.0 if not.
- Non-empty truth with no TPs outputs 0.0.
- Duplicate predicted IDs do not alter the metric because logic operates strictly on sets.

## 8. Manual Metric Examples and Results
1. Truth empty, prediction empty → 1.0
2. Truth empty, prediction `{S2-X}` → 0.0
3. Truth `{A}`, prediction `{A}` → 1.0
4. Truth `{A,B}`, prediction `{A}` → 0.833333
5. Truth `{A,B}`, prediction `{A,C}` → 0.5
6. Truth `{A,B}`, prediction `{A,B,C}` → 0.714285

## 9. Property-Test Results
A `test_metrics_property_randomized` test verifies the reference and efficient implementation matches completely against 1,000 synthetic randomized predictions with absolute numerical equivalence under `1e-7`.

## 10. Fold Invariants and Distribution Behavior
- Exactly one fold per S1.
- Complete deterministic replication guaranteed with random seed.
- Source distribution buckets include 0, 1, 2, 3, and 4+ cardinality. Composition covers no-match, S2-only, S3-only, and both. Country is open-set string.
- In instances where perfect stratification fails due to group sizes being smaller than `n_splits`, smaller strata are grouped into a generalized bucket to retain validity.

## 11. Decoder Rules and Tie-Breaking
- Targets can be exclusively assigned to the highest-confidence S1 (`enforce_target_ownership`).
- S2 and S3 independently support top-k counts and absolute score thresholds inclusive of boundaries.
- Threshold optimization uses conservative tie-breaking: higher macro-thresholds always beat lower equivalent ones.

## 12. Output Validation Guarantees
`matching_results.tsv` and `candidate_pairs.tsv`:
- Strict single row per expected test S1.
- Enforces comma-separated ID subsets without quoting.
- Final matches are forced to be subsets of candidate pairs.
- Duplicate target identifiers are prevented by set semantics.
- Writes to newly created run IDs; does not overwrite existing folders.

## 13. Exact Test, Lint, and Coverage Results
- 17 pytest checks pass completely.
- Coverage ranges 60-97% for written modules.
- Ruff outputs zero errors.

## 14. Unresolved Assumptions or Risks
None. The implementations cleanly follow stated rules.

## 15. Confirmations
Raw data, ingestion code (`io.py`), normalization, profiling, blocking, configs (`configs/data_paths.yaml`), existing profiling reports, and existing tests were not modified.
