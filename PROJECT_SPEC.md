# Project contract

- Source 1 (S1) is the deduplicated reference. Resolve every S1 record against Sources 2
  and 3; an S1 record can have zero, one, or multiple matches.
- Country is an open-set string field. Training contains `US` and `India`; test additionally
  contains unseen `France`. Do not hard-code a closed country vocabulary, and include every
  test S1 entity.
- Evaluation is the macro-average of per-S1 F0.5, which weights precision more heavily than
  recall. A true singleton scores 1.0 only when predicted with an empty match list, otherwise
  0.0.
- The leaderboard file is `output/matching_results.tsv`. The final package also requires
  `output/candidate_pairs.tsv`, containing the exact final candidate set scored by the model;
  final matches must be a subset of those candidates. Both are tab-separated and contain one
  row per test S1 entity, including empty-list singletons.
- The final package also includes runnable source plus a pinned environment and a completed
  methodology document based on `Documentation_template.md`.
- External databases, APIs, geocoders, internet searches, and other business-identity lookup
  or data augmentation are strictly prohibited.
- Any final model must use an MIT or Apache-2.0 license and have no more than 8 billion
  parameters.

Authority: organizer-supplied `student_resource/README.md`.

