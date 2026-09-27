# Retrieval rescue: final integration handoff — 2026-09-27

`fast_submission` is the hard-deadline fallback. It produces both complete submission TSVs directly from label-free exact lookups, without any K20/K40 run or sparse artifacts. K20, K40 and stretch retain their original per-channel K80 retrieval depth and are **not verified to finish inside the submission window**. No new retrieval experiment or full local production run was performed during this integration repair.

The highest completed diagnostic mean oracle is **0.9911914026818882**, with held-out folds 3–4 pooled oracle **0.9907835387476042**. These are retrieval ceilings, not achieved model/submission scores. The fast fallback has no measured accuracy score and deliberately trades recall for execution speed.

## Delivery and verification

Branch: `lane-retrieval-rescue`. Repair parent: `2d9f41f559a931ee092b420b422daab9f2f9f372`; original base: `f9708345608e77940d34da12110a8cb62a47d873`. The final repair is one commit; its full hash and actual push/clean-tree outcome are returned with delivery (a commit cannot embed its own hash). Obtain it with `git rev-parse HEAD` after checkout. No master merge or pull request.

Complete pytest suite: **91 passed** (final suite result; see delivery for elapsed time). Ruff: **All checks passed**. Whitespace check: `git diff --check` passed. Verification commands:

```sh
UV_CACHE_DIR=/tmp/retrieval-rescue-uv MPLCONFIGDIR=/tmp/retrieval-rescue-mpl uv run pytest
UV_CACHE_DIR=/tmp/retrieval-rescue-uv uv run ruff check .
```

Exact files in the final repair commit relative to `2d9f41f`:

- `configs/production_candidates.yaml`
- `configs/retrieval_rescue.yaml`
- `src/entity_resolution/candidate_fusion.py`
- `src/entity_resolution/production_candidates.py`
- `src/entity_resolution/retrieval_rescue.py`
- `tests/test_production_candidates.py`
- `tests/test_retrieval_rescue.py`
- `reports/retrieval_rescue_handoff.md`

Every preexisting uncommitted change was reviewed. Retained: channel-union provenance, grouped/source metrics, bounded streaming evaluation, disk-backed experiment reproduction helpers, per-field sparse budget overrides, bounded LRU cache and their tests. Corrected: the uncommitted emergency per-channel K20 overrides were removed, restoring K80 evidence; the unmeasured name-K512 override was removed from `max_score`. No blanket reset/discard was used. The former 76-test report was stale; the pre-repair suite had 85 tests and the repair adds six cases.

## Fast submission contract

A separate `ExactOnlyRetriever` shares the normalized ID-registry and exact-probe helpers with production. It scans only the configured mode's three source TSVs, validates prefixes and duplicate IDs, creates a disk-backed SQLite registry and two Unicode-preserving exact indexes, then probes exact name and exact address within country/source. Normalization retains the existing null-like handling, Unicode case folding and punctuation/whitespace treatment. All exact pairs survive; combined evidence is deduplicated with both channels preserved. It does not cap exact evidence to K40. Its specialized exact fusion produces the same schema/provenance as regular exact fusion without per-query fuzzy gates.

It skips **all** vectorizer loading/fitting/fingerprinting, TF-IDF transforms, sparse matrix construction/serialization/loading, country-by-block sparse top-N multiplication, sparse cutoff-tie recomputation, missing-address sparse rescue, and fuzzy quality/budget selection. It does not wait for or require an existing sparse index. Fast mode uses Unicode-preserving exact matches only; accent-folded exact and approximate evidence remain available in the unchanged sparse profiles.

For a test run, the CLI automatically calls the existing submission writer, defaulting to `output` if `--submission-output` is absent. Candidates include name-only, address-only and combined exact matches. Final matches require **both exact name AND exact address**. The unchanged authoritative `Decoder` applies global target ownership with threshold 1 and margin .01; equal-score competing owners are rejected. Decoder calls are batched over complete target groups, including owners from different candidate shards. Every S1 gets a row in both TSVs, including zero-evidence queries.

The no-sparse test replaces both sparse retriever constructors, cached build/retrieve/sparse methods, vectorizer load/fit, sparse matrix load/save and sparse top-N with functions that fail if called. It runs the real fast CLI through schema-valid full-fixture export, with inaccessible labels and vectorizer assets. Additional tests cover fast train/test isolation, interruption/resume and corruption, batch invariance, exact overflow/provenance, original profile retrieval depths, rescue rejection and cross-shard ownership conflicts. No synthetic accuracy result is presented as a real score.

## Exact production commands and paths

On the cloud machine, make `configs/data_paths.yaml` resolve the immutable organizer files. The fast command needs only those source TSVs and the installed repository environment. Sparse commands additionally require the four existing `artifacts/blocking/S{2,3}_{name,address}_vectorizer.joblib` assets. No added dependencies or external data are required.

Fast, including complete submission export:

```sh
uv run python -m entity_resolution.production_candidates --mode test --profile fast_submission --run-id test_fast_submission --data-paths configs/data_paths.yaml --output artifacts/production_candidates/test_fast_submission --submission-output output
```

Expected submission files: **`output/test_fast_submission/matching_results.tsv`** and **`output/test_fast_submission/candidate_pairs.tsv`**. They are generated by this command; no prior K20/K40 execution is needed.

K20 (original evidence depth preserved), including conservative export:

```sh
uv run python -m entity_resolution.production_candidates --mode test --profile emergency_k20 --run-id test_emergency_k20 --data-paths configs/data_paths.yaml --threads 16 --output artifacts/production_candidates/test_emergency_k20 --submission-output output
```

K40, including conservative export:

```sh
uv run python -m entity_resolution.production_candidates --mode test --profile safe_k40 --run-id test_safe_k40 --data-paths configs/data_paths.yaml --threads 16 --output artifacts/production_candidates/test_safe_k40 --submission-output output
```

Stretch K400 with the original missing-address K20 rescue, including conservative export:

```sh
uv run python -m entity_resolution.production_candidates --mode test --profile stretch_adaptive --enable-rescue --run-id test_stretch_adaptive --data-paths configs/data_paths.yaml --threads 16 --output artifacts/production_candidates/test_stretch_adaptive --submission-output output
```

For interrupted generation, repeat the exact command with `--resume`. A completed export directory is intentionally not overwritten: resume generation/export only when `output/<run_id>` does not already exist, or choose another submission root. To generate training candidates, change mode, run ID and candidate directory to train equivalents and omit `--submission-output`; labels are still never read by generation.

Each candidate directory contains `part-*.parquet`, `manifest.json`, and `index/{index.json,records.sqlite}`. Sparse profiles additionally have fitted vectors and matrix blocks. Export builds `submission.sqlite` beside the shards and writes to `output/<run_id>/`. Atomic shard validation, run/config/input/code/dependency fingerprints, contiguous resume validation and exclusive run-directory locking apply to the fast profile too. Code changes intentionally invalidate resume from older implementation hashes.

`competitive_adaptive` preserves the base K80 channel union plus gated missing-address K80 at .20. `max_score` preserves that base union plus gated missing-address K2048 at .18, with explicit source-aware RRF weights and retained per-channel scores/ranks. Its 32-GiB optional sparse LRU is byte-bounded, uses 250-query shards and never truncates complementary channels to meet a cosmetic budget. Neither profile includes unmeasured name/address expansions.

No query/shard-range distributed execution was added. Existing shards support single-writer resume; independently produced partial runs must not be concatenated into a submission without a validated coverage/manifest merge and global ownership decode. Adding that contract was deferred to protect closure.

## Runtime, memory and accuracy limits

**Fast eliminates the multi-day sparse bottleneck; full-data completion time is not measured or guaranteed.** It still pays source SHA-256 reads, one normalized registry-building scan, SQLite index construction, a query scan with indexed probes, exact shard writes and global submission export. Complexity is dominated by source rows, index sorting and actual exact hits, not query-by-corpus similarity work. On a CPU cloud machine with local SSD, plan for tens of minutes and allow an hour or more for millions of records; this is a planning allowance, not a benchmark. Slow/network storage or common exact keys can take longer. The tiny-fixture tests verify behavior, not deadline throughput. Start this fallback first and inspect actual completion/progress before relying on a wall-clock deadline.

Fast preparation uses a 32-MiB SQLite page cache and 10,000-row parser batches; candidate shards are 1,000 queries. Ordinary process memory is expected to be roughly 0.5–3 GiB, unmeasured at full scale. The existing writer sorts all S1 IDs in RAM (O(S1)); pair mappings and global ownership groups are disk-backed. Pathological exact collisions can raise memory and output volume; the existing two-million-evidence-row guard fails explicitly, without silently dropping candidates. Smaller shards can address aggregate-shard overflow. An individual ownership group above 100,000 rows also fails visibly. Provision several times the raw TSV size in SSD space for registry, temporary indexes, exact evidence and export maps. The planned 128-GB machine has ample nominal RAM for the ordinary fast path; RAM alone does not remove disk/runtime limits.

Fast accuracy will be substantially below the approximate-retrieval oracle: typos, transliteration, field edits, missing fields and cross-country discrepancies can lose matches, and name-only/address-only candidates are intentionally not emitted as final matches. No validation tuning or new accuracy experiment was run for it.

**K20, K40 and stretch are not verified to finish inside the deadline.** They still build and search four sparse channels at per-channel K80, regardless of the smaller final fusion K. Historical four-channel retrieval took about 1,435 seconds per 5,000 queries; an unoptimized linear extrapolation was about 138 hours for test (176 hours for train) at four threads. Cached matrices, a warm LRU or 16 cloud threads may improve this, but no full-run speedup or deadline guarantee was measured. Do not use these profiles as the hard-deadline fallback.

The high-score union is also expensive: 2,529.2548 mean candidates extrapolates to billions of test pairs. A 128-GB machine can stream the bounded generator, but downstream feature storage and runtime must be budgeted separately; no full-scale feasibility claim is made. Local evaluation peaks were about 1.9 GiB. The max-score production LRU alone permits 32 GiB plus query/result/native-library overhead, so it is intended for the cloud, not the 8-GB laptop.

## Completed leakage-safe measurements (frozen; no new experiments)

These results use the existing five grouped folds, 1,000 queries each: 5,000 diagnostic S1 queries and 13,415 positive edges; folds 3–4 have 2,000 queries and 5,371 edges. The authoritative ground-truth parser and per-S1 macro F0.5 evaluator are unchanged. Empty-owner and empty-candidate queries remain in the denominator. Source scores project truth onto one source while keeping every query. These folds were repeatedly diagnosed, so they are **reused validation**, not untouched final validation. No test labels, external augmentation or numeric-ID ranking features were used.

The highest mean improves over the quoted original **0.98846 by 0.0027314026818882065**, and over its exact recomputation **0.9884639999033469 by 0.0027274027785413324**. Candidate-conditioned theoretical downstream ceiling is **0.9911914026818882** across these folds and **0.9907835387476042** on pooled folds 3–4, assuming perfect discrimination/decoding. The actual ranker will ordinarily score lower. 0.990 was exceeded on the pooled populations; 0.995 and 0.997 were not reached, and fold 2 remains below .990.

| Configuration | Mean fold oracle | Held-out oracle | Covered / 13,415 | Full-owner / 5,000 | Mean candidates | Candidate rows |
|---|---:|---:|---:|---:|---:|---:|
| emergency_k20 | 0.9431537287735587 | 0.9442734676448599 | 11816 | 3958 | 25.344 | 126720 |
| safe_k40 | 0.9573522961155909 | 0.9575072883864754 | 12248 | 4200 | 43.9498 | 219749 |
| fixed_k80 diagnostic | 0.9693540280479404 | 0.9681438587138225 | 12538 | 4377 | 81.8334 | 409167 |
| original per-channel K80 union | 0.9884639999033469 | 0.9889090426515118 | 13018 | 4664 | 319.0568 | 1595284 |
| rescue K20 union | 0.9898801781695251 | 0.9892983824783517 | 13055 | 4701 | 351.8832 | 1759416 |
| stretch_adaptive + rescue K20 | 0.9898009573903044 | 0.9891241400541092 | 13051 | 4697 | 342.9982 | 1714991 |
| competitive_adaptive (rescue K80) | 0.990703296449095 | 0.9903713296924279 | 13080 | 4724 | 464.1224 | 2320612 |
| rescue K512, threshold .20 | 0.9911420939261278 | 0.9907436001915366 | 13096 | 4738 | 1124.235 | 5621175 |
| rescue K512, threshold .18 | 0.9911516177356517 | 0.9907674097153462 | 13097 | 4739 | 1148.7282 | 5743641 |
| rescue K2048, threshold .20 | 0.9911818788723645 | 0.9907597292237947 | 13098 | 4740 | 2385.8466 | 11929233 |
| max_score (rescue K2048, .18) | 0.9911914026818882 | 0.9907835387476042 | 13099 | 4741 | 2529.2548 | 12646274 |

The saved early JSON key `competitive_adaptive` refers to the old adaptive K400/rescue-K20 experiment; current code calls that measurement `legacy_adaptive_k400`. The current named `competitive_adaptive` matches `rescue_k80_union` and was replayed through production gating, fusion, schema validation and atomic shard writing: 2,320,612 rows, exactly 0.990703296449095 mean oracle, 108.883732583 seconds for fusion/write, 62,901,092 Parquet bytes and 1,412.390625 MiB process peak. The max-score union was measured from production-backend rescue evidence plus cached original channels; full end-to-end production replay for max was not completed. Tiny fixtures test its configured pipeline. Old diagnostic exact channels were capped at 80; production retains all exact matches, so full-run counts and small backend tie differences can differ.

| Fold | Original union oracle | Competitive oracle | Max-score oracle | Max covered edges | Max full-owner queries |
|---|---:|---:|---:|---:|---:|
| 0 | 0.9883227458815694 | 0.9911506679594916 | 0.9914840012928248 | 2617/2676 | 951 |
| 1 | 0.9905618937222198 | 0.9934731491334753 | 0.9935640582243843 | 2626/2679 | 955 |
| 2 | 0.9856172746099217 | 0.9881500057676529 | 0.9893418763970235 | 2609/2689 | 938 |
| 3 | 0.9889788659711574 | 0.9909429422174916 | 0.9913371655226495 | 2614/2673 | 949 |
| 4 | 0.9888392193318664 | 0.9897997171673641 | 0.990229911972559 | 2633/2698 | 948 |

| Population | Source | Max source oracle | Covered edges | Recall | Full-owner queries | Candidate rows |
|---|---|---:|---:|---:|---:|---:|
| all | S2 | 0.9920903679653679 | 6640/6772 | 0.980507974010632 | 4885 | 6265721 |
| all | S3 | 0.9902847689075632 | 6459/6643 | 0.972301670931808 | 4844 | 6380553 |
| held_out | S2 | 0.9925239448051949 | 2681/2729 | 0.9824111396115793 | 1957 | 2494097 |
| held_out | S3 | 0.9908142586580089 | 2566/2642 | 0.9712339137017411 | 1934 | 2548972 |

The max configuration improves on the original union in **all five folds and both sources**. Fold standard deviation is 0.001419969502400521; minimum is 0.9893418763970235. Compared with the K512/.18 rescue union, however, K2048/.18 gains only two positive edges and 0.000039784946236465224 mean oracle, concentrated in two folds, while mean volume grows from 1,148.7282 to 2,529.2548. That marginal gain is weak evidence of generalization; it is retained for the requested maximum-score option, while K80 rescue is the practical score/runtime option. Lowering threshold .20 to .18 adds one edge in one fold. No claim of uniform gain is made for either small marginal change.

Max cost on all queries: **12,646,274 rows**, mean **2,529.2548**, median **2,262.5**, p95 **4,412**, maximum **4,535**; 13,099/13,415 edges covered; full/partial/zero owners **4,741/242/17**. Versus K40: 851 positives recovered, none lost, 12,426,525 added pairs, 14,602.26204465335 candidates per newly covered edge. Rescue retrieval alone took **508.3121742500225 seconds**; evaluation-process peak **1,939.40625 MiB**. That timing excludes original-channel retrieval/index construction and is not a full generation runtime.

Held-out max cost: **5,043,069 rows**, mean **2,521.5345**, median **2,271.5**, p95 **4,412**, max **4,509**; full/partial/zero owners **1,897/94/9**. It covers 5,247/5,371 edges, leaves **124** unavailable, and gains 34 over the original full union (no original-channel fusion truncation in union mode). The missing-address gate activated 4,844/5,000 diagnostic queries, so it should not be described as selective or low-volume.

## Missed-edge audit and deliberately deferred work

The original K40 held-out audit had 464 misses: 306 fusion losses and 158 absent-channel edges. The table below preserves the earlier audited taxonomy and sanitized examples, with **historical K400/rescue-K20** remaining counts. Those remaining counts are not the final max-score taxonomy. A new detailed taxonomy for max was deliberately deferred; its aggregate remaining count is 124 above. Flags overlap and describe symptoms, not proven causes.

| Class | Baseline misses | % of 464 | Existing channel present | Newly recovered by rescue channel beyond old union | Remaining after selected stretch | Sanitized representative |
|---|---:|---:|---:|---:|---:|---|
| missing_or_null_name | 0 | 0.00% | 0 | 0 | 0 | — |
| missing_address | 59 | 12.72% | 25 | 8 | 26 | `04f959f39d5a63c7`; name tokens 4/4; missing address=True |
| both_fields_sparse | 0 | 0.00% | 0 | 0 | 0 | — |
| accent_unicode | 2 | 0.43% | 2 | 0 | 0 | `a845f37bf755d12c`; name tokens 2/2; missing address=False |
| legal_suffix | 51 | 10.99% | 48 | 1 | 2 | `0316dd4743519a3e`; name tokens 2/3; missing address=False |
| token_reorder | 11 | 2.37% | 10 | 0 | 1 | `091903523b39290c`; name tokens 4/4; missing address=False |
| token_segmentation | 5 | 1.08% | 5 | 0 | 0 | `a845f37bf755d12c`; name tokens 2/2; missing address=False |
| abbreviation_initials | 0 | 0.00% | 0 | 0 | 0 | — |
| typographical_corruption | 262 | 56.47% | 205 | 6 | 53 | `012dacbfed50dfcd`; name tokens 3/2; missing address=False |
| digit_address_conflict | 173 | 37.28% | 94 | 0 | 81 | `012dacbfed50dfcd`; name tokens 3/2; missing address=False |
| cross_language_like | 82 | 17.67% | 29 | 0 | 53 | `004f519ecbe528d1`; name tokens 4/12; missing address=False |
| fusion_truncation | 306 | 65.95% | 306 | 0 | 3 | `002f0803e51c1eec`; name tokens 3/1; missing address=False |
| absent_every_channel | 158 | 34.05% | 0 | 8 | 150 | `004f519ecbe528d1`; name tokens 4/12; missing address=False |
| other_unknown | 44 | 9.48% | 33 | 0 | 11 | `002f0803e51c1eec`; name tokens 3/1; missing address=False |

Fusion/channel preservation and the existing missing-address name index were the supported rescue mechanisms. The retained source-aware RRF uses per-source/channel weights and quotas, and retains provenance instead of summing incomparable cosine scores. Short-name, missing-field, margin and agreement gates are label-free. No new transliteration, embeddings, external API, ID-derived model signal or digit-dropping rule was introduced.

Before integration repair, name-K512, name-K1024 and address-K512 retrieval jobs had been launched and later exited; their completed evidence sidecars are present, but **no oracle evaluation or selection claim** was made for those expansions. Their reproduction helpers are preserved. Further evaluation of those outputs, stripped legal suffixes, alternate short n-grams, transliteration, a fresh untouched validation sample, distributed range merging and full-cloud throughput checks were deliberately deferred because the user stopped experimentation and required closure. No further retrieval job was launched during repair.

Machine-readable completed evidence is in ignored `artifacts/retrieval_rescue/score_evaluation_k512.json`, `score_evaluation_k2048.json`, `verified_competitive_adaptive/manifest.json`, and the earlier `evaluation.json`/`context.json`, with source/sample/fold/artifact fingerprints. The discarded lower-depth emergency diagnostic remains an ignored historical artifact and is not the delivered profile. This checkout used `artifacts/retrieval_rescue/data_paths.yaml` to resolve the sibling organizer files through the same data-path loader; raw bytes and existing blocking artifacts were not modified. Generated data is not committed. Full test submissions were produced only on fixtures; full train/test generation was not run locally.
