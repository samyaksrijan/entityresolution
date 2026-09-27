# Evaluation Handoff

## 1. Objective and Scope
The objective was to repair independent review correctness issues and ensure the exact implementation of the evaluation and decoding lane for the business entity-resolution competition, complying with all organizer rules. A final minimal hardening pass was performed to vectorize the Decoder logic, correct the margin and folds implementations, and introduce regression tests. The scope is strictly limited to the `src/entity_resolution` evaluation modules and related tests.

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
   - Vectorized all pipeline steps: validation, deductions, threshold filtering, per-S1 top-K, and target ownership.
   - Validates for required columns, rejects unknown S1 IDs, and checks `target_source`/prefix correctness.
   - Refined ownership margin meaning strictly to represent ambiguity margin between top two contenders.
   - Enforced raising `ValueError` when `score_margin` is requested without `enforce_target_ownership`.
   - Hardened `search_thresholds` deterministic tie-breaking.

3. **metrics.py**:
   - Exposed a strict validation mode blocking unknown S1 keys inside predictions.
   - Improved diagnostic bucket generation to prevent memory spikes by incrementing statically defined bins directly.
   - Output stable 0 counts for entirely empty datasets.

4. **folds.py**:
   - Enforced strict validation checking `2 <= n_splits <= len(ground_truth)`.
   - Safely managed tiny rare strata by avoiding native StratifiedKFold errors and reassigning smaller segments with round-robin mapping.
   - Properly persisted diagnostics parameters.

5. **Tests**:
   - Added regression coverage for shuffled candidate rows producing invariant output.
   - Verified that score_margin without ownership raises an exception.
   - Tested ownership margin interactions explicitly between competing S1 models.
   - Asserted that sequence order applies top-K *before* ownership resolution.
   - Asserted n_splits failures, non-numeric bounds rejections, and incomplete columns errors.

## 7. Exact Test and Lint Results
- **Pytest**: 19 tests passed completely.
- **Ruff Lint**: All checks passed (0 errors).

## 8. Remaining Limitations
- full-corpus decoder performance is not yet benchmarked;
- final organizer validator still must run on real outputs;
- downstream candidate generation must respect fold assignments.
