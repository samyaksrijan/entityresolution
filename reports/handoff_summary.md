# Forensic-analysis handoff

## 1. Dataset scale

The six source files contain 24,229,173 rows (2.23 GiB on disk): 2,206,821 training and 1,732,544 test S1 entities. The truth file contains 2,206,821 S1 rows and 7,638,365 positive edges.

## 2. Data-quality findings

Full scans found 0 unknown S1 truth references, 0 unknown targets, 0 duplicate IDs within truth lists, and 0 S1 IDs used as targets. Conflicting target ownership is 0. Country is treated as open-set; France is a measured test-only label, not an error. Field-level empty, whitespace-only, null-like, Unicode, and duplicate measurements are in `data_profile.json`.

## 3. Ground-truth structure

There are 123,247 singleton S1 entities (5.585%). Cardinality counts are {'4+': 1058364, '3': 530841, '2': 375212, '0': 123247, '1': 119157}; source edge counts are {'S1-S2': 3693619, 'S1-S3': 3944746}; 1,776,047 S1 entities have matches in both S2 and S3. Country and missingness breakdowns are {'India': {'s1_entities': 883188, 'positive_edges': 3059843}, 'US': {'s1_entities': 1323633, 'positive_edges': 4578522}} and {'complete': {'s1_entities': 2206821, 'positive_edges': 7638365}}.

## 4. Five main matching difficulties

1. Singleton decisions affect 123,247 S1 rows (5.585%) and false positives score zero for those rows.
2. Multi-match output is material: 1,058,364 S1 rows have 4+ targets.
3. Test-only France contributes 1,694,445 records, creating open-set country and language shift.
4. Missing names/addresses reach maxima of 0.000% and 3.356% across source files.
5. Exact channels alone do not recover every edge: the strongest measured channel is `S2:address_char_tfidf` at 90.062% sampled recall (measurement scope documented in `blocking_feasibility.md`).

## 5. Five strongest opportunities

1. Exact normalized name and address produce high-precision, low-cost candidates with measured candidate distributions in the blocking report.
2. Address digits are preserved in every comparison view and digit conflicts are measured on all 7,638,365 positive edges.
3. Character TF-IDF tolerates punctuation, spacing, and transcription variation; sampled full-target recall is measured separately for S2 and S3.
4. Source-specific retrieval is justified by 3,693,619 S1-S2 versus 3,944,746 S1-S3 edges.
5. The exact per-S1 singleton-aware objective supports calibrated source/channel-specific thresholds rather than forcing a match.

## 6. Train/test shift

- S1 rows: train 2,206,821, test 1,732,544 (0.785x); missing-name shift 0.000% points and missing-address shift 0.000% points.
- S1 sampled mean name/address length shift: -0.25/+5.26 chars; non-ASCII name/address shift 2.354%/4.235%; digit-bearing name/address shift -0.446%/-0.656%; normalized-name duplicate-density shift -1.978%.
- S2 rows: train 5,034,616, test 4,887,273 (0.971x); missing-name shift 0.000% points and missing-address shift -0.708% points.
- S2 sampled mean name/address length shift: +0.61/+4.13 chars; non-ASCII name/address shift 3.804%/5.242%; digit-bearing name/address shift -1.190%/2.028%; normalized-name duplicate-density shift -0.666%.
- S3 rows: train 5,285,603, test 5,082,316 (0.962x); missing-name shift 0.000% points and missing-address shift -0.650% points.
- S3 sampled mean name/address length shift: +0.54/+1.96 chars; non-ASCII name/address shift 3.032%/5.332%; digit-bearing name/address shift -1.140%/1.694%; normalized-name duplicate-density shift -1.067%.
- Country labels: train=['India', 'US'], test=['France', 'India', 'US']; test-only=['France']. Unseen labels are measured shift, not integrity errors.
- France contributes 1,694,445 test rows across sources (measured).
- S1 sampled test-token OOV rate is 11.661%; top-100 train/test token overlap is 82.000%. France sample n=5,303, address-missing 0.000%, name/address non-ASCII 15.821%/27.456%, and address-digit rate 99.698%.
- S2 sampled test-token OOV rate is 9.316%; top-100 train/test token overlap is 84.000%. France sample n=13,965, address-missing 3.179%, name/address non-ASCII 25.156%/23.817%, and address-digit rate 92.954%.
- S3 sampled test-token OOV rate is 9.526%; top-100 train/test token overlap is 86.000%. France sample n=14,521, address-missing 3.078%, name/address non-ASCII 23.566%/24.530%, and address-digit rate 93.410%.

These are measurements. A hypothesis for the next milestone is that France requires country-aware calibration; test labels do not exist, so no France accuracy claim is made.

## 7. Blocking results

The best sampled recall was `S2:address_char_tfidf` at 90.062%, with 20.00 average candidates and 90.000% all-match coverage. Exact/digit channels were evaluated against every target; TF-IDF used sparse full-target scans with sampled queries, a 40,000-feature vocabulary, cosine >=0.35, and top-20 cap. Values are measured samples and full-volume values are extrapolations.

## 8. Measured compute recommendation

This run peaked at 1660.2 MiB process RSS on an 8 GiB host with 1506.6 MiB available at start. Use a 16-vCPU, 64-GiB RAM, 100-GiB SSD tier for the next milestone so a disk-backed/ANN candidate index, full feature tables, and cross-validation folds can coexist safely. This is a recommendation derived from measured peak plus the avoided 40.9-TiB dense-matrix requirement.

## 9. Ranked modelling recommendation

1. Gradient-boosted pair classifier on normalized similarities, token/digit evidence, missingness, country and source indicators; exclude numeric entity-ID components.
2. Calibrated source-specific decision thresholds optimized with exact macro per-S1 F0.5 and singleton handling.
3. Character/word sparse retrieval union plus exact/digit blocks, followed by hard-negative mining.
4. Country-aware calibration with France treated as unseen/open-set, never inferred from external data.
5. A compact text encoder only if sparse candidates plateau; any model must be MIT/Apache-2.0 and <=8B parameters.

## 10. Next three experiments

1. Implement leakage-safe grouped validation and the exact per-S1 macro F0.5 scorer, then baseline deterministic rules.
2. Build a persistent sparse/ANN index and measure full-S1 recall/candidate volume for the union without the query sample.
3. Train a first pair ranker with deterministic random and hard negatives, then tune singleton-aware thresholds by source and country.

## 11. Critical unknowns

- Test truth is unavailable; France generalization is a hypothesis only.
- Sampled TF-IDF recall may differ from full-query recall; sample sizes and caps are explicit in the blocking report.
- Hash-equivalent duplicate counts carry a negligible but non-zero 64-bit collision risk.
- The precision impact of candidate unions is unknown until leakage-safe validation is implemented.

## Reproduction

From the repository root: `.venv/bin/python -m entity_resolution.forensic_analysis --output-dir reports --exact-sample 500 --tfidf-sample 100`.
