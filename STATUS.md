# Status

## Current milestone

- Scalable candidate-generation v1 is implemented and measured without modifying organizer data.
- Country agreement is 100% across all 7,638,365 positive edges, so retrieval uses country as a
  hard partition.
- Exact name/address and character TF-IDF name/address channels were evaluated separately for S2
  and S3 on a fixed 5,000-S1 truth-stratified sample against the full target sources.
- Sparse retrieval is batch-oriented, deterministic, persisted, resumable, and backed by
  `sparse-dot-topn==1.2.0` (Apache-2.0). Candidate provenance and address-digit pair evidence are
  retained in Parquet.
- No candidate configuration is selected for production. The best measured K=80 union reaches
  97.041% positive-edge recall and 0.98846 oracle macro F0.5, below the 98.5% and 0.99 targets,
  while also exceeding the preferred volume at 319.06 mean and 325.05 p95 candidates per S1.
- No final matcher was trained and no competition `candidate_pairs.tsv` or predictions were made.

## Next milestone

Improve candidate recall before pair-model training, focusing on India, S3-only truth, and true
targets with missing addresses. Evaluate additional bounded, business-text-only retrieval views
(for example token/phonetic or field-aware fallbacks), then rerun the fixed sample and select the
smallest union that clears the candidate recall and oracle thresholds.
