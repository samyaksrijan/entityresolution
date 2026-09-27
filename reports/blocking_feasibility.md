# Blocking feasibility

All target sources were retrieved separately. Exact/digit channels use a deterministic S1 sample against every target row. TF-IDF uses sparse matrices only, transforms targets in 20,000-row chunks, and retains cosine >= 0.35 with a top-20 cap. TF-IDF vocabulary fitting and recall are sampled feasibility measurements, not full-corpus guarantees.

| Source/channel | Edge recall | All-matches coverage | Avg cand. | Median | p95 | Max | Runtime s | Peak MiB | Est. full volume |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S2:address_char_tfidf | 90.062% | 90.000% | 20.00 | 20.0 | 20.0 | 20 | 219.2 | 1660.2 | 44,136,420 |
| S2:address_word_tfidf | 63.975% | 61.000% | 20.00 | 20.0 | 20.0 | 20 | 56.9 | 1660.2 | 44,136,420 |
| S2:digit_token_overlap | 80.435% | 77.000% | 90331.02 | 8946.5 | 423735.1 | 764081 | 85.0 | 1321.2 | 199,344,396,301 |
| S2:exact_digit_union | 85.240% | 81.800% | 90335.54 | 8948.5 | 423735.1 | 764081 | 87.2 | 1321.2 | 199,354,362,304 |
| S2:exact_normalized_address | 11.670% | 18.400% | 0.23 | 0.0 | 1.0 | 5 | 85.0 | 1321.2 | 503,155 |
| S2:exact_normalized_name | 21.281% | 21.000% | 4.87 | 0.0 | 27.0 | 147 | 85.0 | 1321.2 | 10,747,218 |
| S2:name_char_tfidf | 60.870% | 56.000% | 20.00 | 20.0 | 20.0 | 20 | 127.1 | 1660.2 | 44,136,420 |
| S2:name_word_tfidf | 38.509% | 36.000% | 20.00 | 20.0 | 20.0 | 20 | 41.6 | 1660.2 | 44,136,420 |
| S3:address_char_tfidf | 83.626% | 80.000% | 19.85 | 20.0 | 20.0 | 20 | 221.1 | 1660.2 | 43,805,396 |
| S3:address_word_tfidf | 52.047% | 49.000% | 20.00 | 20.0 | 20.0 | 20 | 50.9 | 1660.2 | 44,136,420 |
| S3:digit_token_overlap | 81.236% | 76.600% | 93332.99 | 10592.5 | 445478.0 | 799442 | 98.2 | 1660.2 | 205,969,202,324 |
| S3:exact_digit_union | 86.865% | 82.000% | 93337.92 | 10592.5 | 445478.0 | 799442 | 100.8 | 1660.2 | 205,980,081,952 |
| S3:exact_normalized_address | 4.084% | 12.800% | 0.08 | 0.0 | 1.0 | 2 | 98.2 | 1660.2 | 180,959 |
| S3:exact_normalized_name | 24.724% | 22.400% | 5.34 | 1.0 | 30.1 | 138 | 98.2 | 1660.2 | 11,780,010 |
| S3:name_char_tfidf | 68.421% | 60.000% | 20.00 | 20.0 | 20.0 | 20 | 140.0 | 1660.2 | 44,136,420 |
| S3:name_word_tfidf | 42.105% | 37.000% | 20.00 | 20.0 | 20.0 | 20 | 35.6 | 1660.2 | 44,136,420 |

## Resource guard

Available memory at start: 1506.6 MiB; observed process peak: 1660.2 MiB. No dense all-pairs matrix or Cartesian product was constructed. The full unsampled TF-IDF query set was not attempted: even a single dense 2,206,821 x 5,034,616 float32 matrix would require about 40.9 TiB. The safe alternative is the measured chunked sample followed by an indexed ANN implementation in the next milestone.
