# Evaluation Handoff

## 1. Objective and Scope
The objective was to repair independent review correctness issues and ensure the exact implementation of the evaluation and decoding lane for the business entity-resolution competition, complying with all organizer rules. The scope is strictly limited to the `src/entity_resolution` evaluation modules and related tests.

## 2. Branch Name
`lane-evaluation`

## 3. Base Commit
`f09cd0f9ff3b2b5fc720f37a29eea102e9f69e43`

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

## 6. Resolved Defects
1. **submission.py**:
   - Added validation against explicit `valid_target_ids` sets containing every actual test S2/S3 ID.
   - Now validates all inputs completely before creating output directories to avoid orphaned folders.
   - Enforces matches being a strict subset of candidates, explicitly failing with a clear error on violations.
   - Enforces duplicate candidates logic: strictly canonicalizes lists explicitly and fails loudly if raw candidates contain duplicate entries in input dictionaries.
   - Implemented deterministic ordering and exact newline generation for Unix `\n` per competition format.

2. **decode.py**:
   - Added validation for required columns, target prefix rules against source column, and rejects unknown S1 IDs.
   - Enforces finite numeric scores.
   - Deduplicates candidate rows, deterministically preserving the highest score.
   - Restructured decoder strictly applying threshold and top-K *before* ownership resolution.
   - Adjusted score margin to check ownership ambiguity accurately: highest vs second highest.
   - Ensure all output strings omit duplicates.

3. **metrics.py**:
   - Exposed a strict validation mode blocking unknown S1 keys inside predictions.
   - Improved diagnostic bucket generation to prevent memory spikes by incrementing statically defined bins directly.
   - Output stable 0 counts for entirely empty datasets.

4. **Threshold Search**:
   - Hardened searching with detailed total deterministic tie-breaking. Sorts equivalent thresholds against Global, S2, S3, Margin, and then K variables.

5. **folds.py**:
   - Enforced validation checks on `n_splits` limits and data size limits.
   - Safely managed tiny rare strata by avoiding native StratifiedKFold errors and reassigning smaller segments with round-robin mapping.
   - Properly persisted diagnostics parameters.

6. **Tests**:
   - Authored extensive boundary tests to validate prefix behavior, target source overlaps, small fold distributions, NaN scores, exact properties, explicit folder creations, tie-breaking heuristics, and random generation.

## 7. Exact Test and Lint Results
- **Pytest**: 19 tests passed completely. 
- **Ruff Lint**: 0 errors.

## 8. Remaining Risks
None identified.
