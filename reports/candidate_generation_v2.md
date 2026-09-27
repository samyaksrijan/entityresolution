# Candidate generation v2

## Scope and evidence validation

The fixed 5,000-S1 truth-stratified diagnostic sample was reused byte-for-byte. All selection tuning used development folds 0–2; folds 3–4 remained held out. Cosine values were not compared across channels: fusion uses reciprocal ranks, exact flags, support counts, and content-derived tie keys.

Validated 1,693,164 evidence rows and 1,595,284 distinct candidate pairs. All required artifacts passed schema, row-count, run-ID, query-coverage, rank, duplicate, finiteness, source-separation, and configuration-fingerprint checks.

| Artifact | Rows | Queries | Maximum rank | SHA-256 |
|---|---:|---:|---:|---|
| S2_exact | 45,549 | 3,173 | 80 | `9b6fa4dfde3a8d431be06ce5ee01b21b0f164d8c3485eb5157eacdab32ac940b` |
| S2_name | 400,000 | 5,000 | 80 | `6ea8caf4271c819b6425b177caf4d1f2713468861f6d3c274f0aa06a7fe03e96` |
| S2_address | 400,000 | 5,000 | 80 | `67c1cabbe471d25a70cbc56b2eae2b268eed76b755c6299039aeeef754609f46` |
| S3_exact | 47,651 | 2,911 | 80 | `2490cb3e027196848d9582076f0d170cd103ce31439a7c4bf6f632fe4671a4b0` |
| S3_name | 400,000 | 5,000 | 80 | `12c046c7238f7e47f1ba93171695b635cee303ff21aebe9116283813becabec8` |
| S3_address | 399,964 | 5,000 | 80 | `15bdcdf42f953d8e257011ed47ce05509755a926bf374deb6096f1822e8fee1c` |

## Leakage-safe selection

Selected on development folds only: `rrf|global=40|rrf_k=60`.
Exact name/address candidates from either normalization view are always retained before the fuzzy budget. Identical content/evidence boundary ties are expanded; target IDs never break ties.

| Split | Edge recall | All truth | Oracle F0.5 | Mean | Median | p90 | p95 | p99 | Max | Zero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Development (measured) | 91.261% | 83.967% | 0.95725 | 44.06 | 40.0 | 40.0 | 65.0 | 171.0 | 187 | 0.000% |
| Held-out folds 3–4 (measured) | 91.361% | 84.050% | 0.95751 | 43.78 | 40.0 | 40.0 | 64.0 | 155.0 | 187 | 0.000% |

### Held-out fold stability

| Fold | Edge recall | All truth | Oracle F0.5 | Mean | p95 | Gate |
|---:|---:|---:|---:|---:|---:|---|
| 3 | 90.797% | 83.700% | 0.95382 | 43.92 | 73.0 | fail |
| 4 | 91.920% | 84.400% | 0.96119 | 43.64 | 60.1 | fail |

### Held-out target-source results

| Source | Positive edges | Edge recall | Oracle F0.5 | Mean candidates |
|---|---:|---:|---:|---:|
| S2 | 2,729 | 92.671% | 0.96999 | 21.70 |
| S3 | 2,642 | 90.008% | 0.95973 | 22.08 |

Mean/worst held-out fold values are recorded in the JSON report. The held-out candidate-volume gate passes, but the recall and oracle gates fail.

## Held-out subgroup results

### country

| Group | S1 | Edge recall | All truth | Oracle F0.5 | Mean | p95 |
|---|---:|---:|---:|---:|---:|---:|
| India | 963 | 90.843% | 81.724% | 0.96231 | 40.06 | 40.0 |
| US | 1,037 | 91.847% | 86.210% | 0.95304 | 47.24 | 101.0 |
### cardinality_bucket

| Group | S1 | Edge recall | All truth | Oracle F0.5 | Mean | p95 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 56 | 0.000% | 100.000% | 1.00000 | 44.00 | 45.5 |
| 1 | 309 | 95.146% | 95.146% | 0.95146 | 43.64 | 42.8 |
| 2 | 592 | 92.145% | 86.824% | 0.95693 | 45.00 | 82.2 |
| 3 | 559 | 90.519% | 80.143% | 0.95221 | 43.39 | 62.0 |
| 4+ | 484 | 91.050% | 76.240% | 0.96328 | 42.81 | 40.0 |
### truth_composition

| Group | S1 | Edge recall | All truth | Oracle F0.5 | Mean | p95 |
|---|---:|---:|---:|---:|---:|---:|
| S2_only | 690 | 92.266% | 86.812% | 0.95584 | 44.66 | 74.6 |
| S3_only | 596 | 90.381% | 83.054% | 0.94446 | 42.60 | 48.0 |
| both_sources | 658 | 91.376% | 80.699% | 0.96746 | 43.91 | 73.1 |
| no_match | 56 | 0.000% | 100.000% | 1.00000 | 44.00 | 45.5 |
### address_state

| Group | S1 | Edge recall | All truth | Oracle F0.5 | Mean | p95 |
|---|---:|---:|---:|---:|---:|---:|
| present | 2,000 | 91.361% | 84.050% | 0.95751 | 43.78 | 64.0 |
### target_address_state

| Group | S1 | Edge recall | All truth | Oracle F0.5 | Mean | p95 |
|---|---:|---:|---:|---:|---:|---:|
| all_true_target_addresses_present | 1,736 | 92.253% | 85.887% | 0.95869 | 43.89 | 71.0 |
| any_true_target_missing | 208 | 85.513% | 64.423% | 0.93617 | 42.79 | 40.0 |
| no_true_target | 56 | 0.000% | 100.000% | 1.00000 | 44.00 | 45.5 |
### exact_name_state

| Group | S1 | Edge recall | All truth | Oracle F0.5 | Mean | p95 |
|---|---:|---:|---:|---:|---:|---:|
| available | 932 | 93.755% | 86.481% | 0.97941 | 44.64 | 77.0 |
| unavailable | 1,068 | 88.959% | 81.929% | 0.93839 | 43.03 | 49.9 |
### exact_address_state

| Group | S1 | Edge recall | All truth | Oracle F0.5 | Mean | p95 |
|---|---:|---:|---:|---:|---:|---:|
| available | 766 | 94.107% | 86.945% | 0.98382 | 42.66 | 48.0 |
| unavailable | 1,234 | 89.521% | 82.253% | 0.94118 | 44.48 | 77.3 |

## Fusion plateau and Pareto frontier

The full existing top-80 union is an absolute ceiling for any fusion subset: its previously measured diagnostic recall is 97.041% and oracle is 0.98846, already below the 98.5%/0.99 gates at 319.06 mean candidates. Therefore no reordering or pruning of this evidence can meet the recall gate.

Development Pareto configurations:

- `exact_first|S2=20,S3=20|rrf_k=60`: recall 91.236%, oracle 0.95742, mean 44.07, p95 65.0.
- `exact_first|S2=25,S3=25|rrf_k=60`: recall 92.031%, oracle 0.96214, mean 53.45, p95 65.0.
- `exact_first|S2=30,S3=20|rrf_k=60`: recall 91.895%, oracle 0.96244, mean 53.48, p95 65.0.
- `exact_first|global=100|rrf_k=60`: recall 94.281%, oracle 0.97383, mean 101.33, p95 100.0.
- `exact_first|global=20|rrf_k=60`: recall 88.016%, oracle 0.94241, mean 25.48, p95 65.0.
- `exact_first|global=30|rrf_k=60`: recall 90.440%, oracle 0.95325, mean 34.74, p95 65.0.
- `exact_first|global=30|rrf_k=60,rescue=20:no_exact_evidence+single_support_channel`: recall 90.564%, oracle 0.95389, mean 38.08, p95 65.0.
- `exact_first|global=40|rrf_k=60`: recall 91.261%, oracle 0.95725, mean 44.06, p95 65.0.
- `exact_first|global=50|rrf_k=60`: recall 91.982%, oracle 0.96176, mean 53.43, p95 65.0.
- `exact_first|global=75|rrf_k=60`: recall 93.324%, oracle 0.96866, mean 77.15, p95 75.0.
- `rrf|S2=20,S3=20|rrf_k=60`: recall 91.236%, oracle 0.95742, mean 44.07, p95 65.0.
- `rrf|S2=25,S3=25|rrf_k=60`: recall 92.031%, oracle 0.96214, mean 53.45, p95 65.0.
- `rrf|S2=30,S3=20|rrf_k=60`: recall 91.895%, oracle 0.96244, mean 53.48, p95 65.0.
- `rrf|global=100|rrf_k=60`: recall 94.281%, oracle 0.97383, mean 101.33, p95 100.0.
- `rrf|global=20|rrf_k=60`: recall 88.016%, oracle 0.94241, mean 25.48, p95 65.0.
- `rrf|global=30|rrf_k=60`: recall 90.440%, oracle 0.95325, mean 34.74, p95 65.0.
- `rrf|global=30|rrf_k=60,rescue=20:no_exact_evidence+single_support_channel`: recall 90.564%, oracle 0.95389, mean 38.08, p95 65.0.
- `rrf|global=40|rrf_k=60`: recall 91.261%, oracle 0.95725, mean 44.06, p95 65.0.
- `rrf|global=50|rrf_k=60`: recall 91.982%, oracle 0.96176, mean 53.43, p95 65.0.
- `rrf|global=75|rrf_k=60`: recall 93.324%, oracle 0.96866, mean 77.15, p95 75.0.

## Candidate-volume estimates

At the held-out mean, full train output is extrapolated to 96,616,830 pairs (2.88 GiB), and test output to 75,852,509 pairs (2.26 GiB). These are extrapolated, not full-run measurements.

## Held-out missed-edge audit

The selected fusion misses 464 held-out positive edges. Flags overlap; primary categories are mutually assigned by documented precedence.

| Category flag | Count | Percent of misses |
|---|---:|---:|
| target_address_missing | 59 | 12.72% |
| weak_or_missing_name_evidence | 366 | 78.88% |
| name_retrieved_below_selected_budget | 98 | 21.12% |
| address_retrieved_below_selected_budget | 241 | 51.94% |
| absent_from_every_existing_top80_channel | 158 | 34.05% |
| duplicate_common_normalized_signature | 12 | 2.59% |
| accent_unicode_normalization_issue | 2 | 0.43% |
| legal_suffix_or_token_order_variation | 62 | 13.36% |
| india_specific_concentration | 238 | 51.29% |
| s3_specific_concentration | 264 | 56.90% |
| multi_match_truncation | 357 | 76.94% |

All overlaps and 5 representative content-only examples are in the JSON report; edge-level flags are in `missed_edges_v2.tsv`. Entity IDs are identifiers only and were not used as features or explanations.

## Uniform-sample status

A separate uniform content-hash sample with 5,000 rows was created without truth, source composition, exact-match state, or retrieval scores. No unbiased final metric is reported because no v2 fusion configuration qualified; a new full-target scan of a rejected configuration would not be final validation. The sample is reserved for the selected post-rescue configuration.

## Next experiment

Run one name-character index restricted to targets whose address is missing on the fixed diagnostic sample, union it with existing evidence, and measure marginal held-out recall gained per added candidate. This directly targets 34 current held-out misses that are both absent from all channels and missing a target address. The accent-only pattern covers 2 misses and does not justify a broad first rescue. Reject the channel if volume rises without material held-out recall gain. No CatBoost training was started. France accuracy is not claimed because France truth is unavailable.

IMPROVE_RETRIEVAL
