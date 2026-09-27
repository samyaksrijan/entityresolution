# Ranker lane handoff

## Scope and file contract

Branch: `lane-ranker-core`. Base: `492b693fad1b968c2f6597bbbad7162a6fc28f86`.
Only the nine requested files were added. No authoritative evaluation, folds,
decoder, submission, retrieval, candidate, feature-generation, dependency or raw
files were changed. No branches were merged. No external data was used.

- `training_schema.py`: input fingerprints, strict Parquet validation, source-ID
  membership, labels, disk-backed feature staging and candidate diagnostics.
- `negative_sampling.py`: bounded per-S1 hard-negative selection and diagnostics.
- `catboost_ranker.py`: ordered CatBoost inputs, CPU fitting and probabilities.
- `model_artifact.py`: native models, JSON contracts, hashes and atomic publication.
- `training_pipeline.py`: CLI, S1 folds, inner early stopping, OOF shards,
  resumable checkpoints, decoder search and final model.
- `inference_pipeline.py`: CLI, bounded scoring, lazy candidate mapping,
  authoritative decoder/submission calls and atomic output publication.
- `configs/ranker_v1.yaml`: conservative CPU, sampling and search defaults.
- `tests/test_ranker_pipeline.py`: synthetic behavioral and end-to-end tests.
- `reports/ranker_handoff.md`: this handoff.

Feature input is a Parquet file or a directory of `part-*.parquet` shards.
Every shard must match `OUTPUT_SCHEMA`, including column order and Arrow types.
This deliberately accepts the materializer's exact types, rather than silently
casting ostensibly compatible input. A present materializer manifest must be
complete, have the correct train/test mode and schema version, and match all
shard names, row counts and SHA-256 hashes. Bare shards are accepted with the
current schema contract; use manifested inputs in production.

The model uses exactly `feature_names()` in its declared order. Projection from
`OUTPUT_SCHEMA` into that model order is explicit; unknown/missing/reordered
input columns fail. Both `CATEGORICAL_FEATURES` are passed by name. `s1_id`,
`target_id` and `run_id` are never inputs, nor are numeric ID components.
`target_source` is an intentional categorical feature. SQLite JSON transport
reconstructs numeric Python values; it does not alter the feature list or values.

Source membership is loaded only through `load_data_paths`. TSV parsing uses
explicit tabs. Input checks reject missing values, non-finite numbers, inconsistent
run IDs, invalid source ownership, absent source IDs and duplicate pairs across
all batches/shards. Output paths beneath the configured organizer root or
intersecting feature/model inputs are rejected.

## Leakage protections and scoring

Labels are exactly target membership in authoritative `load_ground_truth`
results. Training requires exact S1 coverage and valid truth targets. It never
creates missing positive candidates. `create_folds` assigns deterministic S1
folds using its existing cardinality/composition strategy, without country input.
All rows of a validation S1 stay outside its model's training sample.

Each outer fold creates a separate inner S1 split using only outer-training
truth. When both inner partitions contain both classes, the inner model performs
early stopping. Its tree count is then used to refit on all outer-training
sampled pairs. Outer validation labels never enter model fitting, stopping,
sampling quotas or learned state. If a usable inner split is absent, configured
iterations are used; a one-class outer-training pool fails with an actionable
error. The final model uses the median outer-model iteration count and all
sampled training entities. No test files are read by training.

Every held-out candidate, including unsampled negatives, receives OOF probability.
Fold checkpoints record outer and inner S1 sets, native-model hashes, prediction
hashes, selected iterations and sampling diagnostics. The union of OOF scores
is the only input to decoder selection. All training S1 IDs participate in macro
F0.5, including no-match, candidate-free and retrieval-missed entities.

The default search has 45 coarse settings (five thresholds, three top-k values,
and three ownership/margin settings), followed by seven local settings including
source-specific threshold changes. It uses `search_thresholds`, `Decoder`, and
`evaluate_macro_f0_5_efficient` unchanged. The full seven-field selected decoder
configuration is persisted. No independent probability-0.5 decision rule exists.
Minimum evidence is unused because it would merely duplicate probability gating.

SQL first filters thresholds and applies the decoder's exact score/lexical top-k
order. For final decoding, ownership requires at most the best two eligible owners
per target. The authoritative decoder then handles decisions. During tuning,
SQL only filters at the minimum searched threshold and maximum searched top-k;
all ownership and individual-setting decisions remain in `search_thresholds`.
Synthetic tests verify equivalence, including shared targets and ties.

Reported metrics distinguish total/retrieved/missed truth pairs, candidate recall,
full-truth S1 coverage (including vacuously covered no-match S1), S2/S3 counts,
no-match and candidate-free counts, and the candidate oracle macro F0.5. The
oracle predicts only retrieved truth edges and ignores ownership restrictions;
it is an upper bound, not the learned model score. OOF reports contain pair
counts/TP/FP/FN, per-source recall, no-match false positives and fold scores.
Fold scores are slices of the globally ownership-decoded OOF union. The selected
OOF score is used for decoder selection and is not an independently held-out or
nested estimate of the entire selection process.

## Sampling

Defaults: 12 negatives per retrieved positive, minimum six negatives per S1,
maximum 36 negatives per S1, and a 250,000-row sampled-training guard.
Every retrieved positive is retained. No-match S1 and S1 whose positives were
missed receive the minimum allocation when negatives exist. Each available target
source gets one hard negative before filling remaining slots by hardness.

Exact retrieval/name/name-and-address indicators rank first, followed by the
strongest existing retrieval/fusion/string evidence. SHA-256 of seed and complete
pair identifiers breaks ties, independently of input order and shard boundaries.
Exact-looking false positives therefore receive priority within the cap; the cap
can still exclude some when there are too many. The source reservation can retain
a weaker negative to preserve diversity. IDs are used only for keys and seeded
tie-breaking, never for model features.

Only a bounded short list per source/S1 is held during negative selection.
Sampled keys stay in SQLite. The global cap fails safely before fitting if the
policy cannot fit, rather than discarding positives or silently dropping S1s.
Increase the cap on suitable hardware or lower negative quotas. Setting the cap
to null explicitly removes this guard and is not recommended on the local machine.

## Artifact, resume and inference

Artifact format version 1 contains:

- `model.cbm`, `fold-N.cbm`: native CatBoost models, no pickled objects.
- `metadata.json`: complete marker, format/schema versions, schema fingerprint,
  ordered/categorical features, effective final hyperparameters, seed, fold
  strategy/count, sampling policy/diagnostics, decoder, input fingerprints,
  candidate diagnostics, OOF summary and package versions.
- `folds.json`, `fold-N.json`: assignments and per-fold provenance/checkpoints.
- `oof-N.parquet`: bounded-write OOF predictions with IDs, probability and fold.
- `decoder_search.parquet`: the actual coarse/local search results.
- `state.json`: exact resume identity.

All artifact payload files are SHA-256 checked at load, and native-model feature
names/categorical positions must match the live feature contract. Incomplete,
corrupt and incompatible artifacts fail. Output is assembled in a sibling
`<output>.work`, protected by `<output>.lock`, and atomically renamed only after
verification. Checkpoints and JSON manifests use temporary files plus atomic
rename. Interrupted folds replay; completed hash-verified folds resume. Resume
requires identical config, feature/source fingerprints, consumed implementation
files and package versions. A completed artifact can be verified/reused with
`--resume`; ordinary overwrite is refused. SQLite staging is rebuilt on resume,
so resume saves model fitting but does not skip input validation/scanning.

Inference reads only configured test source TSVs, feature inputs and the model
artifact. It never opens training TSVs or ground truth. Scoring uses bounded
batches, produces atomic `scored/part-*.parquet` plus a complete manifest, applies
the stored OOF decoder, and passes a lazy SQLite candidate mapping to
`write_submission`. Every test S1 is emitted in both TSV files. Matches are
validated as candidate subsets and valid S2/S3 targets by the existing writer.
A complete submission directory is published atomically; existing run IDs fail.

## Exact commands

If the default uv cache is inaccessible in this sandbox, first run:

```bash
export UV_CACHE_DIR=/tmp/ranker-uv-cache
```

Smoke on available real materialized training data (200 S1 by seeded hash;
2 folds, at most 12 iterations, depth 3 and two threads):

```bash
uv run python -m entity_resolution.training_pipeline \
  --features artifacts/pair_features/train_max_score \
  --output artifacts/models/ranker_v1_smoke \
  --config configs/ranker_v1.yaml \
  --data-paths configs/data_paths.yaml --smoke
```

Production training; append `--resume` only for an existing compatible run:

```bash
uv run python -m entity_resolution.training_pipeline \
  --features artifacts/pair_features/train_max_score \
  --output artifacts/models/ranker_v1 \
  --config configs/ranker_v1.yaml \
  --data-paths configs/data_paths.yaml
```

Inference/submission:

```bash
uv run python -m entity_resolution.inference_pipeline \
  --features artifacts/pair_features/test_max_score \
  --model artifacts/models/ranker_v1 \
  --output output --run-id ranker_v1_submission \
  --data-paths configs/data_paths.yaml
```

Expected production output: `artifacts/models/ranker_v1/` and
`output/ranker_v1_submission/{matching_results.tsv,candidate_pairs.tsv,manifest.json,scored/}`.

The following synthetic CLI smoke was actually executed in this worktree. Its
fixture can be reproduced once in a fresh ignored directory with:

```bash
uv run python - <<'PY'
from pathlib import Path
import runpy
root = Path('artifacts/ranker_smoke_fixture').resolve()
root.mkdir(parents=True, exist_ok=False)
runpy.run_path('tests/test_ranker_pipeline.py')['fixture_data'].__wrapped__(root)
PY
uv run python -m entity_resolution.training_pipeline \
  --features artifacts/ranker_smoke_fixture/features \
  --output artifacts/models/ranker_v1_synthetic_smoke \
  --config artifacts/ranker_smoke_fixture/configs/ranker.yaml \
  --data-paths artifacts/ranker_smoke_fixture/configs/data_paths.yaml --smoke
uv run python -m entity_resolution.inference_pipeline \
  --features artifacts/ranker_smoke_fixture/features \
  --model artifacts/models/ranker_v1_synthetic_smoke \
  --output output --run-id ranker_v1_synthetic_smoke \
  --data-paths artifacts/ranker_smoke_fixture/configs/data_paths.yaml
```

These generated fixtures/artifacts/submissions are local and git-ignored; they are
not production deliverables or part of the commit. A second training invocation
with `--smoke --resume` was also tested.

## Memory, remaining risks and cloud experiment

Defaults are CPU-only, four threads, 400 iterations, depth six, 64 borders and
CatBoost `used_ram_limit: 2gb`. CatBoost's RAM setting is advisory, not a hard
process memory limit. Feature scans use 2,048 rows per batch and a 16 MiB SQLite
cache. The full feature pool is staged on disk as JSON payloads, which can be
substantially larger than compressed Parquet. Reserve ample local scratch disk.
Smoke validates/scans all input shards but retains full payloads only for selected
S1s; it bounds model work, not all validation I/O.

CatBoost fitting requires a sampled pool in RAM. Inner training and evaluation
pools may coexist, each protected by the sampled-row cap. Source ID sets, truth,
fold membership, predictions and submission S1 lists scale with entity count;
this implementation does not claim constant memory with respect to all IDs.
Submission candidates are read one S1 at a time. Decoder DataFrames are guarded
at 500,000 rows after exact SQL filtering; SQL sorting uses disk-backed temporary
storage. Exceeding that guard fails with instructions instead of attempting an
unbounded allocation. Overall peak RSS and production throughput have not been
measured. Very large source registries can still require cloud memory.

Start cloud execution on a CPU machine with 32 GiB RAM and local SSD scratch,
using the unchanged four-thread model configuration. After measuring actual row
counts/RSS, raise `sampling.max_sampled_rows` and `max_decode_rows` in a separate
ranker YAML, or reduce negative quotas while retaining source/no-match coverage.
Eight CPU threads may be tested in a separate run; no GPU is required. These are
initial engineering settings, not measured capacity or vendor recommendations.

First full-data experiment: provision immutable organizer TSVs, materialize train
features upstream, run the 200-S1 smoke, inspect retrieval misses and class coverage,
then run five-fold OOF training under a new artifact directory. Compare learned
OOF macro F0.5 to candidate oracle and no-match FP counts; inspect every fold and
selected decoder. Only then score the separately materialized test features.
Threshold transfer from OOF models to the final refit and sampled-negative
probability calibration remain empirical risks. Tiny/one-class datasets may need
a different seed/subset or cannot support binary training; no artificial labels
are introduced to conceal that limitation.

Rollback: keep earlier model/submission directories intact, select the previous
artifact for inference under a new run ID, and retain this run for inspection.
If reverting code is necessary, revert the ranker commit rather than touching
raw/candidate/feature artifacts. To retry incompatible settings, choose a new
output directory; do not force resume or remove checkpoints from a run in use.

## Validation evidence

Synthetic tests exercise labels, positive retention, deterministic/source-diverse
hard negatives, no-match entities, missing truth edges, identifier exclusion,
outer/inner S1 separation, strict OOF prediction provenance, tiny real CatBoost
fits, save/load, schema/dtype/order changes, corrupt artifacts and checkpoints,
deterministic decoder selection, exact SQL reduction, empty candidate inputs,
valid deterministic submissions, and inference after every training file is
removed. End-to-end predictions and OOF scores match across reordered/resharded
inputs. No call-only mocks substitute for training or inference.

CLI synthetic smoke: 25 S1 entities, 192 candidates, 33 truth edges, 32 retrieved
positives, 112 sampled final-training rows, two folds. Measured synthetic OOF macro
F0.5 = 0.96; pair candidate recall = 32/33; candidate oracle macro F0.5 = 0.96;
32 predicted pairs, zero false positives and one retrieval-missed truth edge.
This is an intentionally simple synthetic check, not a competition estimate.

Full-data OOF score: **not measured on full data**. The configured
`student_resource/` directory and real pair-feature shards are absent in this
worktree. No full-data fitting, production throughput measurement or competition
submission upload was attempted. No authoritative-module blocker was found.

Final checks (with `UV_CACHE_DIR=/tmp/ranker-uv-cache`):

```text
uv run pytest
........................................................................ [ 52%]
.................................................................        [100%]
137 passed in 30.01s

uv run ruff check .
All checks passed!

git diff --check
(no output; exit 0)
```
