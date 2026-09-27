# Experiments

## 2026-09-27 — Candidate generation v1

### Scope

- Measured country agreement over all 7,638,365 positive training edges.
- Built a fixed, retrieval-independent, truth-stratified sample of 5,000 S1 entities.
- Queried that sample against every S2 and S3 training target using exact normalized name,
  exact normalized address, character TF-IDF name, and character TF-IDF address channels.
- Evaluated K = 5, 10, 20, 40, and 80 per channel and for required/all-view unions.
- Scored positive-edge recall, per-S1 all-truth coverage, candidate-count distributions, and the
  exact oracle macro per-S1 F0.5 ceiling including true no-match entities.

### Sparse backend benchmark

`sparse-dot-topn==1.2.0` (Apache-2.0) was benchmarked on 200 queries by 20,000 targets before the
full scans. The optimized result retained roughly 3,750 nonzeros per source benchmark versus
about 1.3–1.4 million nonzeros in the unbounded SciPy sparse product. Projected full name-matrix
storage is 1.95 GB for S2 and 2.05 GB for S3. Full four-matrix persistence was not safe locally.

### Main union results

| K | Edge recall | All truth recovered | Oracle macro F0.5 | Mean candidates | p95 candidates |
|---:|---:|---:|---:|---:|---:|
| 5 | 90.652% | 81.460% | 0.95992 | 19.14 | 24.00 |
| 10 | 93.343% | 86.160% | 0.97142 | 38.99 | 44.00 |
| 20 | 95.050% | 89.580% | 0.97966 | 79.00 | 85.00 |
| 40 | 96.154% | 91.700% | 0.98506 | 159.13 | 166.05 |
| 80 | 97.041% | 93.280% | 0.98846 | 319.06 | 325.05 |

At K=80, S2 recall is 97.578% with a 0.99102 oracle ceiling; S3 recall is 96.493% with a
0.98757 ceiling. The weakest major groups are India at 95.447% edge recall, S3-only truth at
96.492%, and S1 entities with any true target missing its address at 93.703%. Cardinality 4+ has
88.264% all-truth recovery. All 5,000 sampled S1 addresses are present, so no S1-missing-address
estimate is available.

### Resources and outcome

Measured exact scans took 288.0 seconds for S2 at 170.4 MiB peak RSS and 290.4 seconds for S3 at
283.5 MiB. Sparse name/address scans took 268.8/463.1 seconds for S2 and 324.0/379.3 seconds for
S3; the maximum measured process RSS was 516.9 MiB. Full detailed results, extrapolated train/test
pair and Parquet sizes, and subgroup tables are in `reports/candidate_generation_v1.json` and
`reports/candidate_generation_v1.md`.

No configuration qualifies for production. Word TF-IDF was optional and was not retained. No
final pair classifier, cross-encoder, neural model, competition candidate TSV, or test prediction
was produced.
