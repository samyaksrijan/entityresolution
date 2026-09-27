# Retrieval rescue handoff — 2026-09-27

**The production and fallback submission paths are implemented and tested. Retrieval remains inadequate for the requested competition score.** The selected stretch reaches held-out oracle macro F0.5 **0.989124140**, covering **5,218/5,371** positive edges. Even the complete existing-plus-rescue union reaches only **0.989298382**. A classifier cannot exceed that candidate oracle on this population. No official submission score or full-corpus production performance is claimed.

Implementation commit: `f064d110cdf22f39b16149925c62042740d65a90`. Base: `f9708345608e77940d34da12110a8cb62a47d873`. Branch: `lane-retrieval-rescue`. The final branch tip adds this report in a separate documentation commit. Git status was clean after the implementation commit; the final clean-tree and origin-tip checks accompany delivery. No merge or pull request was made.

`UV_CACHE_DIR=/tmp/retrieval-rescue-uv MPLCONFIGDIR=/tmp/retrieval-rescue-mpl uv run pytest`: **76 passed** (4.76 seconds on the final configuration). `UV_CACHE_DIR=/tmp/retrieval-rescue-uv uv run ruff check .`: **All checks passed**. The complete implementation diff and whitespace checks were reviewed. Only the following nine owned files changed:

- `src/entity_resolution/production_candidates.py`
- `src/entity_resolution/retrieval_rescue.py`
- `src/entity_resolution/candidate_fusion.py`
- `src/entity_resolution/missed_edge_analysis.py`
- `configs/production_candidates.yaml`
- `configs/retrieval_rescue.yaml`
- `tests/test_production_candidates.py`
- `tests/test_retrieval_rescue.py`
- `reports/retrieval_rescue_handoff.md`

The tests cover both modes with an inaccessible ground-truth path, source/identifier validation, compact Arrow dtypes, empty/null-like inputs, exact overflow, deduplication and provenance, source/channel reservations, rank weights, stable ties across target block sizes and thread counts, parser/query batch invariance, adaptive gates, enabled/disabled rescue, atomic validation failure, interruption before and after shard rename, resume incompatibility and corruption, fitted-vocabulary reuse, country partitioning, raw-output guards, and the existing decoder/submission writer. Existing retrieval APIs and behavior remain intact.

**Measurement population and selection.** All rows below use the same 2,000 S1 queries in existing grouped held-out folds 3–4, with 5,371 positive edges. Development folds 0–2 contain 3,000 queries. The original authoritative fold creation and ground-truth parser were reused. Every oracle is checked against `evaluate_macro_f0_5_reference`; no metric, fold, decoder, or submission-writer implementation changed. The sample is truth-stratified and has already been diagnosed: these are diagnostic held-out measurements, not untouched final validation. The existing uniform sample was not evaluated. No test ownership, test labels, external business data, or numeric entity-ID features were used.

The earlier full-diagnostic top-80 result (97.041% recall, 0.98846 oracle, 319.06 mean) covered all 5,000 sampled queries. Its separately measured held-out ceiling below must not be confused with that full-sample statistic. `fixed_k80` means 80 total selected pairs; `top80_union_ceiling` means the entire union of per-source/per-view top-80 evidence.

| Configuration | Covered / 5,371 | Edge recall | S2 recall | S3 recall | Full / partial / zero owner queries | Oracle macro F0.5 |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_v2_k40` | 4,907 | 91.361% | 92.671% | 90.008% | 1,681 / 270 / 49 | 0.957507288 |
| `fixed_k20` | 4,736 | 88.177% | 89.886% | 86.412% | 1,583 / 357 / 60 | 0.944273468 |
| `fixed_k40` | 4,907 | 91.361% | 92.671% | 90.008% | 1,681 / 270 / 49 | 0.957507288 |
| `source_k40` | 4,908 | 91.380% | 92.781% | 89.932% | 1,683 / 268 / 49 | 0.957495283 |
| `weighted_source_k40` | 4,731 | 88.084% | 92.781% | 83.232% | 1,567 / 383 / 50 | 0.948252602 |
| `channel_k40` | 4,908 | 91.380% | 92.671% | 90.045% | 1,682 / 269 / 49 | 0.957552743 |
| `fixed_k80` | 5,009 | 93.260% | 94.247% | 92.241% | 1,745 / 221 / 34 | 0.968143859 |
| `stretch_adaptive` | 5,209 | 96.984% | 97.582% | 96.366% | 1,863 / 127 / 10 | 0.988637398 |
| `top80_union_ceiling` | 5,213 | 97.058% | 97.655% | 96.442% | 1,866 / 124 / 10 | 0.988909043 |
| `rescue_union_ceiling` | 5,221 | 97.207% | 97.838% | 96.556% | 1,874 / 116 / 10 | 0.989298382 |
| `rescue_stretch` | 5,210 | 97.002% | 97.728% | 96.253% | 1,868 / 122 / 10 | 0.988727519 |
| `rescue_adaptive_k400` | 5,218 | 97.151% | 97.801% | 96.480% | 1,871 / 119 / 10 | 0.989124140 |

Queries with no true owners count as fully covered under the authoritative metric. Every query remains in the denominator, including queries with no candidates. Counts and percentiles include those queries.

| Configuration | Mean | Median | p95 | Maximum | Candidate rows | New positives / positives lost vs K40 | New candidate pairs vs K40 | Candidates added / new positive | Seconds | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_v2_k40` | 43.7810 | 40.0 | 64.0 | 187 | 87,562 | 0 / 0 | 0 | — | 1.69 | 1245.3 |
| `fixed_k20` | 25.1420 | 20.0 | 64.0 | 187 | 50,284 | 0 / 171 | 0 | — | 37.61 | 1245.3 |
| `fixed_k40` | 43.7810 | 40.0 | 64.0 | 187 | 87,562 | 0 / 0 | 0 | — | 36.72 | 1245.3 |
| `source_k40` | 43.7810 | 40.0 | 64.0 | 187 | 87,562 | 5 / 4 | 1,459 | 291.80 | 36.32 | 1245.3 |
| `weighted_source_k40` | 43.7810 | 40.0 | 64.0 | 187 | 87,562 | 19 / 195 | 16,420 | 864.21 | 36.24 | 1245.3 |
| `channel_k40` | 43.7810 | 40.0 | 64.0 | 187 | 87,562 | 1 / 0 | 1 | 1.00 | 41.16 | 1245.3 |
| `fixed_k80` | 81.6670 | 80.0 | 80.0 | 187 | 163,334 | 102 / 0 | 75,772 | 742.86 | 38.56 | 1245.3 |
| `stretch_adaptive` | 308.2085 | 318.0 | 320.0 | 320 | 616,417 | 302 / 0 | 528,889 | 1,751.29 | 40.47 | 1245.3 |
| `top80_union_ceiling` | 318.8915 | 318.0 | 325.0 | 414 | 637,783 | 306 / 0 | 550,221 | 1,798.11 | 0.33 | 1245.3 |
| `rescue_union_ceiling` | 351.5905 | 352.0 | 361.0 | 453 | 703,181 | 314 / 0 | 615,619 | 1,960.57 | 37.62 | 1245.3 |
| `rescue_stretch` | 310.3370 | 320.0 | 320.0 | 320 | 620,674 | 303 / 0 | 533,146 | 1,759.56 | 95.19 | 1900.9 |
| `rescue_adaptive_k400` | 341.8780 | 352.0 | 361.0 | 400 | 683,756 | 311 / 0 | 596,228 | 1,917.13 | 78.23 | 1317.5 |

New candidate pairs are set additions, not net row-count changes. The selected stretch adds 596,228 held-out pairs and removes 34 baseline negative pairs, for a net increase of 596,194. It loses no baseline positive edges. Times for fusion rows measure selection across all 5,000 sampled queries, excluding shared artifact loading. Rescue union time includes the one retrieval pass; rescue-stretch time includes retrieval and fusion. The final K400 re-budget time includes recreating the K40 reference and re-fusing saved rescue evidence, with no new retrieval. Peak RSS is each process’s cumulative high-water mark, not an isolated per-configuration allocation.

| Configuration | Development oracle | Development full-owner queries | Development edge recall |
|---|---:|---:|---:|
| `baseline_v2_k40` | 0.957248968 | 2,519 | 91.261% |
| `fixed_k20` | 0.942407236 | 2,375 | 88.016% |
| `fixed_k40` | 0.957248968 | 2,519 | 91.261% |
| `source_k40` | 0.956776820 | 2,517 | 91.186% |
| `weighted_source_k40` | 0.945214530 | 2,334 | 87.581% |
| `channel_k40` | 0.957360079 | 2,521 | 91.285% |
| `fixed_k80` | 0.970160808 | 2,632 | 93.598% |
| `stretch_adaptive` | 0.988151432 | 2,797 | 97.016% |
| `top80_union_ceiling` | 0.988167305 | 2,798 | 97.029% |
| `rescue_union_ceiling` | 0.990268042 | 2,827 | 97.389% |
| `rescue_stretch` | 0.989779031 | 2,821 | 97.290% |
| `rescue_adaptive_k400` | 0.990252169 | 2,826 | 97.377% |

The reliable K40 baseline was frozen before rescue work. Source reservations at K40 mostly traded positive edges; the existing S3 weighting prescription regressed. Channel reservations added two development edges and one held-out edge, a marginal diagnostic change that was not promoted into the frozen fallback. Among stretch variants, selection used development oracle, full-owner coverage, edge recall, p95, mean, then runtime. The single post-rescue budget repair was justified by development coverage lost during fusion (29 newly available channel positives but only 22 net positives retained at K320), not by a test-label search.

| Selected stretch split | Oracle F0.5 |
|---|---:|
| Development | 0.990252169 |
| Held-out combined | 0.989124140 |
| Held-out fold 3 | 0.989117124 |
| Held-out fold 4 | 0.989131157 |

| Threshold | Selected development | Selected held-out | Full rescue union held-out |
|---|---|---|---|
| 0.990 | Reached | Not reached | Not reached |
| 0.995 | Not reached | Not reached | Not reached |
| 0.997 | Not reached | Not reached | Not reached |

**One targeted mechanism.** Existing views already supplied Unicode-preserving and accent-folded exact name/address matches and Unicode name/address character TF-IDF (char_wb 3–5, min_df 2, 50,000 features, cosine threshold 0.20, K80 per source/view). No redundant normalization view or neural dependency was added. The new channel retrieves name top-20 separately within each source’s missing-address target subset, using the existing fitted name vocabulary. Diagnostic retrieval reused `SparseTopNRetriever`; production filters and reuses its cached name matrices. The subset contains 168,967 S2 and 175,916 S3 target records.

The label-free gate uses missing fields, normalized lengths (<5 name or <8 address), minimum within-channel top-1/top-2 margin (<0.05), maximum channel agreement (<2), exact evidence, and viable fuzzy views (<2). Any weak signal expands the budget; rescue additionally requires a usable name. Missing evidence activates the safe expanded fallback. Both adaptive budgets and rescue activate for 4,844/5,000 queries (96.88%). This dataset makes the gate broad; it does not support a low-volume claim.

The new channel adds 98,734 development pairs and recovers 29 previously absent development edges (3,404.62 pairs per new positive), within the prespecified 10,000-pair cost cap. Held-out it adds 65,398 previously absent pairs and recovers 8 of the 34 original missing-address/top-80-absent positive edges (8,174.75 pairs per new positive). At K320, seven old covered positives are displaced, leaving only one net held-out improvement over the previous adaptive configuration. The K400 re-budget restores those seven and one additional original-channel edge. No second retrieval mechanism, threshold sweep, or further sparse pass was run. The mechanism is retained only as an optional stretch setting; its absolute validation gain is small.

**Diagnosis and remaining misses.** The K40 baseline missed 464 held-out edges: 306 were fusion losses and 158 were absent from all old channels. The deterministic flags overlap and are heuristic descriptions, not causal labels. After the selected stretch, **153 edges remain missed: 150 channel failures and 3 fusion/gating losses**; none of the baseline positives is newly lost. No missing-name or both-fields-sparse record occurred among these baseline misses; those paths are still covered synthetically.

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

“Existing channel present” answers whether fusion could have recovered that edge without new retrieval. Script mismatch is only “cross-language-like,” not verified translation. Typographical corruption uses normalized-name similarity 70–<100%; digit conflict compares nonempty address digit sequences. Examples contain only content hashes and token/missingness counts.

| Class | Proposed mechanism / decision | Expected extra candidate cost |
|---|---|---|
| Missing/null-like name | Existing address-only retrieval; no new view justified by this audit | 0 new channels |
| Missing address | Implemented missing-address-target name subindex | At most 20 per source per gated query before deduplication |
| Both fields sparse | Input insufficiency; preserve zero-candidate queries and report them | No unsupported candidate expansion |
| Accent/Unicode | Existing accent-folded exact view already supplies both flagged edges | 0 new channels |
| Legal suffix | Existing helper could support a future stripped-name view; not evaluated in this sprint | Would require a separately capped view; cost unmeasured |
| Token reorder | Existing char view and fusion already recover most; no new view | 0 here |
| Segmentation | Existing char view plus higher budget recovers all five flagged edges | 0 new channels |
| Abbreviation/initials | No observed misses justify a new view | 0 here |
| Typographical corruption | Existing char retrieval plus bounded rescue; shorter n-grams remain future work | Implemented rescue bounded at 40 raw pairs/query; other costs unmeasured |
| Digit/address conflict | Preserve digits; no large digit-token expansion (prior feasibility showed extreme volume) | 0 here; a future view needs a strict cap |
| Cross-language-like | No existing lightweight transliteration implementation was found; not added | Not evaluated |
| Fusion truncation | Source/channel reservations, RRF, then adaptive K400 repair | Measured volume tables above |
| Absent every channel | Missing-address subindex recovers eight; remaining failures need new evidence | 40 raw pairs/gated query for the implemented mechanism |
| Other/unknown | Retain as unresolved; no invented mechanism | Unmeasured |

**Production contracts.** `safe_k40` is frozen global weighted RRF with uniform weights and a nominal total K40; `emergency_k20` uses K20. `stretch_adaptive` uses base K40, expanded K400, source reservations of 20/20, and the optional missing-address rescue flag. All preserve exact matches and identical content/evidence boundary ties beyond nominal caps. Source reservations precede channel reservations; unused seats return to global RRF order. Exact preservation takes precedence when a nominal budget cannot satisfy every reservation.

Output has the eight requested fields with Arrow strings, float32 scores, int16 ranks, and boolean exact flags, plus `support_json` and float32 `fusion_score`. The primary `channel`/score/rank describes one deterministic observation; `support_json` retains every distinct channel’s score, rank, and exact flag. Raw cosine scores are not summed across channels. IDs are used only for integrity, joins, and final serialization, never as ranking features.

Raw source files are only read through the configured data-path resolver with explicit TSV separators. A SQLite ID registry rejects duplicates across parser batches and validates ownership. Exact lookup is disk-backed. Target normalization and float32 sparse blocks are cached once; sparse top-N results are bounded by query/target block sizes. Cutoff ties use a bounded one-query sparse row, never a dense query-by-corpus product. No full Cartesian product is built.

A shard is written to `.parquet.tmp`, schema/content validated, flushed, and atomically renamed. The JSON manifest commits it afterward. Resume verifies input/vocabulary hashes, effective configuration hash, schema, implementation hashes, dependency versions, shard hashes/row counts, and contiguous shard identity. A crash after rename but before manifest update replays exactly the next uncommitted shard. Unknown orphan files fail. Completed shards are reused; incomplete index construction is rebuilt rather than trusted. An OS lock prevents concurrent writers. Query-count histograms include empty queries and support exact mean, median, p95, and maximum; per-shard source/exact counts, time, RSS, gates, and timestamps are recorded.

**Cloud commands.** First make `configs/data_paths.yaml` resolve the immutable organizer data on the cloud machine. Copy the four existing `artifacts/blocking/S{2,3}_{name,address}_vectorizer.joblib` files. Production defaults reuse these small fitted assets and pin their hashes. No labels are stored in or loaded from them. Setting `runtime.vectorizer_root: null` explicitly fits new bounded samples, but those vocabularies have not received the reported validation measurements. No new package dependency is required.

K40 production:

```sh
uv run python -m entity_resolution.production_candidates --mode test --profile safe_k40 --run-id test_safe_k40 --data-paths configs/data_paths.yaml --threads 16 --output artifacts/production_candidates/test_safe_k40
```

K20 emergency, including the conservative fallback submission:

```sh
uv run python -m entity_resolution.production_candidates --mode test --profile emergency_k20 --run-id test_emergency_k20 --data-paths configs/data_paths.yaml --threads 16 --output artifacts/production_candidates/test_emergency_k20 --submission-output output
```

Selected stretch:

```sh
uv run python -m entity_resolution.production_candidates --mode test --profile stretch_adaptive --enable-rescue --run-id test_stretch_adaptive --data-paths configs/data_paths.yaml --threads 16 --output artifacts/production_candidates/test_stretch_adaptive
```

Resume K40 with exactly the same effective settings:

```sh
uv run python -m entity_resolution.production_candidates --mode test --profile safe_k40 --run-id test_safe_k40 --data-paths configs/data_paths.yaml --threads 16 --output artifacts/production_candidates/test_safe_k40 --resume
```

For train generation, replace `--mode test`, the run ID, and output directory with their train equivalents; the generation path still never loads labels. To export the fallback from completed K40 shards, repeat its exact command with both `--resume` and `--submission-output output`. The fallback uses the unchanged `Decoder` with global target ownership and a 0.01 margin on exact-name AND exact-address evidence, then the unchanged `write_submission`. Ambiguous equal-score owners are rejected. This is a tested format-valid submission path, not a trained or calibrated high-score model.

For each production run, candidate output is `artifacts/production_candidates/<run_id>/part-*.parquet`; the authoritative manifest is `artifacts/production_candidates/<run_id>/manifest.json`, and the cache manifest is `index/index.json` beneath that run. Submission output is `output/<run_id>/matching_results.tsv` and `candidate_pairs.tsv`. The disk-backed export map is `artifacts/production_candidates/<run_id>/submission.sqlite`. Export deliberately retains the existing writer’s O(number of S1 IDs) in-memory sort, while candidate-pair mappings and ownership grouping remain on disk.

Reproduce the diagnostic sequence (one retrieval mechanism, then one cached budget repair):

```sh
uv run python -m entity_resolution.retrieval_rescue --data-paths configs/data_paths.yaml
uv run python -m entity_resolution.retrieval_rescue --data-paths configs/data_paths.yaml --run-rescue
uv run python -m entity_resolution.retrieval_rescue --data-paths configs/data_paths.yaml --rebudget 400
```

This checkout has no local `student_resource` directory. Its executed diagnostic commands used `--data-paths artifacts/retrieval_rescue/data_paths.yaml`, an ignored local configuration resolving the original sibling checkout’s raw files through the same resolver. Existing blocking artifacts were read through the preexisting symlinks. Raw files and existing blocking artifacts were not changed. The new cache and machine-readable report are under `artifacts/retrieval_rescue/`: `context.json`, `evaluation.json`, `rescue_evidence.parquet`, and the small query/audit/missing-address target caches. `evaluation.log`, `rescue_evaluation.log`, and `rebudget.log` capture the executed runs. These generated files are ignored by Git; the measurements and commands are preserved here for the committed handoff.

**Resources and integration limits.** Local diagnostic peak RSS was 1,900.9 MiB (about 1.86 GiB); no full train/test generation was attempted on the 8 GB laptop. With default 1,000-query shards, 20,000-target matrix blocks, and 50,000 vocabulary features, plan roughly 1–4 GiB process RSS for ordinary production records, subject to cloud measurement. A 16-vCPU / 32-GB CPU machine with fast SSD and about 256 GB free disk is a reasonable starting allocation for both safe and stretch artifacts, not a measured minimum.

- `safe_k40` extrapolates to 96,616,830 train and 75,852,509 test candidate rows. At an assumed 80–128 compressed bytes/pair including provenance, test output would be about 5.7–9.0 GiB, excluding indexes.
- `emergency_k20` extrapolates to 55,483,894 train and 43,559,621 test candidate rows. At an assumed 80–128 compressed bytes/pair including provenance, test output would be about 3.2–5.2 GiB, excluding indexes.
- `stretch_adaptive` extrapolates to 754,463,550 train and 592,318,678 test candidate rows. At an assumed 80–128 compressed bytes/pair including provenance, test output would be about 44.1–70.6 GiB, excluding indexes.

These volume estimates use a stratified diagnostic sample and are not full-run measurements. In particular production retains all exact matches, while old diagnostic exact channels were capped at 80; full-run exact volume can be larger. The explicit two-million-evidence-row guard fails visibly on pathological shards instead of dropping hard queries. Use smaller shards or an appropriately reviewed higher cloud cap if that guard is reached. K20 reduces downstream output, but retains the shared K80 per-view retrieval setup and does not remove indexing cost.

The prior four sparse channels took about 1,435 seconds for 5,000 queries. A simple old-backend extrapolation is about 176 hours for train or 138 hours for test at its four-thread setting; this is deliberately conservative and includes costs now cached. For a 16-vCPU cloud run, reserve roughly 1–3 days per full mode plus index construction until real shard timing is available. There is no verified full-run time or guaranteed linear thread speedup. Read per-shard timing and index completion on the cloud and update the estimate before relying on a deadline.

Downstream feature/ranker integration must consume the deduplicated pair schema and parse `support_json` for channel-specific features; the primary raw retrieval score is not calibrated across channels. The model lane must still train leakage-safe scores and use the authoritative ownership decoder. Full-corpus cache construction, output volume, throughput, exact-overflow behavior, and full submission validation remain unverified. Small floating-point/tie differences between old diagnostic retrieval and the hardened cached backend are possible. No France/test accuracy is claimed. All held-out thresholds remain unmet, and further retrieval work must address the remaining 150 channel failures before classifier tuning can support the requested final score.
