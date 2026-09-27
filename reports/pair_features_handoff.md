# Pair-feature lane handoff

## Interface

`entity_resolution.feature_schema` exports `OUTPUT_SCHEMA`, `NUMERIC_FEATURES`,
`CATEGORICAL_FEATURES`, `FEATURE_DTYPES`, and `feature_names()`. The model input is
`feature_names()` in that order. `s1_id`, `target_id`, and `run_id` are identifiers;
`target_source` is both an identifier and a categorical model input. No ID-derived
numeric feature is emitted. All feature columns have fixed `float32`, `int16`,
`int8`, or string types. Each output row retains the input pair's identifiers and
position. Raw business text is absent from feature Parquet.

The input accepts the production ten columns, the older eight columns, and either
of the optional columns independently. Missing `support_json` becomes a single
primary-channel observation. Missing `fusion_score` uses `retrieval_score`. A
present but malformed support JSON fails; it never silently falls back. The
actual production support JSON has one observation per channel with `channel`,
`score`, `rank`, and `exact` fields.

## Commands

```bash
uv run python -m entity_resolution.feature_materialization \
  --mode train \
  --candidates artifacts/production_candidates/train_max_score \
  --output artifacts/pair_features/train_max_score

uv run python -m entity_resolution.feature_materialization \
  --mode train \
  --candidates artifacts/production_candidates/train_max_score \
  --output artifacts/pair_features/train_max_score \
  --resume
```

Use `--mode test` with the matching test candidate directory and a separate test
feature output. `--data-paths` defaults through `configs/pair_features.yaml` to
`configs/data_paths.yaml`. `--batch-size` and `--shard-size` override the YAML
defaults of 1000. The effective candidate batch is the smaller setting.

The input can be one Parquet file or a directory of sorted `part-*.parquet`
files. A candidate manifest, when present, must be complete and agree with the
files and hashes. Output is `part-*.parquet` plus `manifest.json`. An incomplete
run resumes only with the same source and candidate fingerprints, schema, code,
and batch settings. Completed shards are checked against their hashes and schema.
An orphaned next shard is replayed from input. Train/test source paths are loaded
only from the data-path configuration. Ground truth is never opened.

## Memory and validation

The source TSVs are parsed explicitly as tabs and staged in a disk-backed SQLite
ID lookup. Each candidate batch fetches only its S1 and target records. A
separate temporary SQLite table detects duplicate pairs across shards without
holding all keys in RAM. Both SQLite files are removed on successful completion;
an interrupted run retains the source index for resume. The resident candidate,
text, and output frames are bounded by one batch. Normalization has bounded
20,000-entry caches. Source and candidate SHA-256 checks stream over files.

The materializer rejects wrong target ownership, missing source records,
duplicate candidates, invalid scores or ranks, malformed channel observations,
changed resume inputs, incomplete retrieval manifests, and corrupt Parquet.
Feature shards are written to a temporary file, read back for schema/row-count
verification, flushed, and atomically renamed before manifest publication.

## Integration assumptions

- CatBoost should use `feature_names()` and pass `CATEGORICAL_FEATURES` as string
  categoricals. Identifiers other than `target_source` must not enter the model.
- Retrieval candidate shards have unique `(s1_id, target_id, target_source)`
  pairs. The materializer preserves their order without expecting duplicate
  per-channel rows.
- Candidate scores are channel-specific evidence; `fusion_score` is used as an
  additional feature. No model calibration or label joins occur here.
- RapidFuzz is already declared in `pyproject.toml` and present in the local
  environment. No dependency file was changed.
- Tests use synthetic source files. Full-data throughput and disk needs have not
  been measured on 8 GB hardware. Source indexing and fingerprinting scan the
  configured train or test TSVs once per new run.

## Ordered feature contract

The following table is generated from the schema source. Its order is the model
input order; `target_source` is the first categorical feature.

| Feature | Dtype |
|---|---|
| `target_source` | `string` |
| `strongest_channel` | `string` |
| `retrieval_score` | `float32` |
| `fusion_score` | `float32` |
| `reciprocal_rank` | `float32` |
| `support_best_score` | `float32` |
| `support_mean_score` | `float32` |
| `support_min_score` | `float32` |
| `support_max_score` | `float32` |
| `support_mean_rank` | `float32` |
| `name_char_similarity` | `float32` |
| `name_accent_char_similarity` | `float32` |
| `name_token_jaccard` | `float32` |
| `name_token_containment_s1` | `float32` |
| `name_token_containment_target` | `float32` |
| `name_token_set_similarity` | `float32` |
| `name_token_sort_similarity` | `float32` |
| `name_prefix_similarity` | `float32` |
| `name_suffix_similarity` | `float32` |
| `name_char_3gram_jaccard` | `float32` |
| `name_token_count_ratio` | `float32` |
| `name_char_length_ratio` | `float32` |
| `address_char_similarity` | `float32` |
| `address_token_jaccard` | `float32` |
| `address_token_containment_s1` | `float32` |
| `address_token_containment_target` | `float32` |
| `address_number_overlap` | `float32` |
| `address_token_count_ratio` | `float32` |
| `address_char_length_ratio` | `float32` |
| `combined_string_similarity` | `float32` |
| `retrieval_name_interaction` | `float32` |
| `retrieval_address_interaction` | `float32` |
| `channel_rank` | `int16` |
| `channel_support_count` | `int16` |
| `exact_support_count` | `int16` |
| `support_min_rank` | `int16` |
| `source_support_count` | `int16` |
| `s2_support_count` | `int16` |
| `s3_support_count` | `int16` |
| `name_support_count` | `int16` |
| `address_support_count` | `int16` |
| `exact_support_family_count` | `int16` |
| `fuzzy_support_count` | `int16` |
| `name_token_count_diff` | `int16` |
| `name_char_length_diff` | `int16` |
| `address_token_count_diff` | `int16` |
| `address_char_length_diff` | `int16` |
| `exact_match` | `int8` |
| `has_source_support` | `int8` |
| `has_s2_support` | `int8` |
| `has_s3_support` | `int8` |
| `has_name_support` | `int8` |
| `has_address_support` | `int8` |
| `has_exact_support` | `int8` |
| `has_fuzzy_support` | `int8` |
| `independent_channel_agreement` | `int8` |
| `name_s1_missing` | `int8` |
| `name_target_missing` | `int8` |
| `name_s1_null_like` | `int8` |
| `name_target_null_like` | `int8` |
| `name_raw_exact` | `int8` |
| `name_normalized_exact` | `int8` |
| `name_casefold_exact` | `int8` |
| `name_accent_exact` | `int8` |
| `name_punctuation_agreement` | `int8` |
| `name_legal_suffix_agreement` | `int8` |
| `name_initials_agreement` | `int8` |
| `name_acronym_agreement` | `int8` |
| `name_digit_agreement` | `int8` |
| `address_s1_missing` | `int8` |
| `address_target_missing` | `int8` |
| `address_s1_null_like` | `int8` |
| `address_target_null_like` | `int8` |
| `address_raw_exact` | `int8` |
| `address_normalized_exact` | `int8` |
| `address_accent_exact` | `int8` |
| `address_number_exact` | `int8` |
| `address_house_number_agreement` | `int8` |
| `address_postal_agreement` | `int8` |
| `both_names_usable` | `int8` |
| `both_addresses_usable` | `int8` |
| `exact_name_and_address` | `int8` |
| `strong_name_weak_address` | `int8` |
| `strong_address_weak_name` | `int8` |
| `exact_name_conflicting_address_digits` | `int8` |
| `country_agreement` | `int8` |
| `country_both_present` | `int8` |
