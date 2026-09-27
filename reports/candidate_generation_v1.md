# Candidate generation v1

## Measurement scope

- Full-dataset measurements: country agreement over all positive edges and exact/sparse retrieval target scans.
- Sampled measurements: fixed 5,000-S1 truth-stratified evaluation sample.
- Extrapolated estimates: candidate output sizes use the sample mean times full S1 counts at 32 Parquet bytes per pair; matrix storage uses benchmark bytes per target row.
- Unverified hypothesis: France retrieval will transfer from the training-country configuration.

## Country decision

Country agreement is 100.000000% across 7,638,365 edges with 0 disagreements. Country is therefore used as a hard retrieval partition.
No disagreement sample is present because the full positive-edge scan found none.

## Backend benchmark

`{'S2': {'queries': 200, 'targets': 20000, 'features': 39778, 'target_nnz': 927899, 'target_matrix_bytes': 7503196, 'optimized_runtime_seconds': 0.06236383301438764, 'scipy_sparse_product_runtime_seconds': 0.028185708972159773, 'optimized_result_nnz': 3763, 'scipy_result_nnz': 1324595, 'estimated_full_target_matrix_bytes': 1950830960, 'license': 'Apache-2.0', 'backend_version': '1.2.0'}, 'S3': {'queries': 200, 'targets': 20000, 'features': 40199, 'target_nnz': 976664, 'target_matrix_bytes': 7893316, 'optimized_runtime_seconds': 0.009794124984182417, 'scipy_sparse_product_runtime_seconds': 0.02942958299536258, 'optimized_result_nnz': 3751, 'scipy_result_nnz': 1402830, 'estimated_full_target_matrix_bytes': 2052262160, 'license': 'Apache-2.0', 'backend_version': '1.2.0'}}`

## Evaluation sample

`{'rows': 5000, 'countries': {'US': 2593, 'India': 2407}, 'cardinality': {'2': 1480, '3': 1398, '4+': 1210, '1': 774, '0': 138}, 'truth_composition': {'S2_only': 1721, 'both_sources': 1648, 'S3_only': 1493, 'no_match': 138}, 'address_state': {'present': 5000}, 'target_address_state': {'all_true_target_addresses_present': 4337, 'any_true_target_missing': 525, 'no_true_target': 138}, 'exact_name_state': {'unavailable': 2668, 'available': 2332}, 'exact_address_state': {'unavailable': 3095, 'available': 1905}}`

## Combined union results

| Configuration | Edge recall | All truth | Missed-any | Oracle F0.5 | Mean | Median | p95 | p99 | Max | Zero | Train GiB | Test GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| combined / all_views_union / K=10 | 93.343% | 86.160% | 13.840% | 0.97142 | 38.99 | 39.0 | 44.0 | 50.0 | 56 | 0.000% | 2.56 | 2.01 |
| combined / all_views_union / K=20 | 95.050% | 89.580% | 10.420% | 0.97966 | 79.00 | 78.0 | 85.0 | 97.0 | 112 | 0.000% | 5.20 | 4.08 |
| combined / all_views_union / K=40 | 96.154% | 91.700% | 8.300% | 0.98506 | 159.13 | 158.0 | 166.1 | 188.0 | 217 | 0.000% | 10.47 | 8.22 |
| combined / all_views_union / K=5 | 90.652% | 81.460% | 18.540% | 0.95992 | 19.14 | 19.0 | 24.0 | 26.0 | 31 | 0.000% | 1.26 | 0.99 |
| combined / all_views_union / K=80 | 97.041% | 93.280% | 6.720% | 0.98846 | 319.06 | 318.0 | 325.1 | 354.0 | 440 | 0.000% | 20.98 | 16.47 |

## All source, channel, K, and union results

Full subgroup breakdowns for every row are in `candidate_generation_v1.json`.

| Configuration | Recall | All truth | Oracle | Mean | p90 | p95 | p99 | Max | Zero | Runtime s | Peak MiB | Est. train pairs | Est. test pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S2 / address_char_tfidf / K=10 | 87.197% | 86.700% | 0.94571 | 10.00 | 10.0 | 10.0 | 10.0 | 10 | 0.000% | 463.1 | 507.7 | 22,068,210 | 17,325,440 |
| S2 / address_char_tfidf / K=20 | 89.486% | 88.800% | 0.95662 | 20.00 | 20.0 | 20.0 | 20.0 | 20 | 0.000% | 463.1 | 507.7 | 44,136,420 | 34,650,880 |
| S2 / address_char_tfidf / K=40 | 90.815% | 89.920% | 0.96336 | 40.00 | 40.0 | 40.0 | 40.0 | 40 | 0.000% | 463.1 | 507.7 | 88,272,840 | 69,301,760 |
| S2 / address_char_tfidf / K=5 | 83.387% | 83.600% | 0.93161 | 5.00 | 5.0 | 5.0 | 5.0 | 5 | 0.000% | 463.1 | 507.7 | 11,034,105 | 8,662,720 |
| S2 / address_char_tfidf / K=80 | 92.026% | 90.900% | 0.96919 | 80.00 | 80.0 | 80.0 | 80.0 | 80 | 0.000% | 463.1 | 507.7 | 176,545,680 | 138,603,520 |
| S2 / all_views_union / K=10 | 94.507% | 93.720% | 0.97874 | 19.37 | 21.0 | 22.0 | 25.0 | 30 | 0.000% | 1020.0 | 507.7 | 42,744,357 | 33,557,991 |
| S2 / all_views_union / K=20 | 95.939% | 95.320% | 0.98464 | 39.40 | 41.0 | 42.0 | 49.0 | 56 | 0.000% | 1020.0 | 507.7 | 86,950,071 | 68,263,273 |
| S2 / all_views_union / K=40 | 96.737% | 96.200% | 0.98851 | 79.45 | 80.0 | 83.0 | 94.0 | 109 | 0.000% | 1020.0 | 507.7 | 175,341,638 | 137,658,244 |
| S2 / all_views_union / K=5 | 92.174% | 91.460% | 0.97156 | 9.46 | 11.0 | 12.0 | 13.0 | 17 | 0.000% | 1020.0 | 507.7 | 20,876,085 | 16,389,520 |
| S2 / all_views_union / K=80 | 97.578% | 97.060% | 0.99102 | 159.44 | 160.0 | 162.0 | 178.0 | 222 | 0.000% | 1020.0 | 507.7 | 351,848,920 | 276,231,618 |
| S2 / char_union / K=10 | 94.315% | 93.560% | 0.97840 | 19.07 | 20.0 | 20.0 | 20.0 | 20 | 0.000% | 732.0 | 507.7 | 42,086,283 | 33,041,347 |
| S2 / char_union / K=20 | 95.791% | 95.200% | 0.98398 | 38.96 | 40.0 | 40.0 | 40.0 | 40 | 0.000% | 732.0 | 507.7 | 85,977,746 | 67,499,914 |
| S2 / char_union / K=40 | 96.648% | 96.100% | 0.98807 | 78.87 | 80.0 | 80.0 | 80.0 | 80 | 0.000% | 732.0 | 507.7 | 174,047,559 | 136,642,280 |
| S2 / char_union / K=5 | 91.834% | 91.220% | 0.97047 | 9.22 | 10.0 | 10.0 | 10.0 | 10 | 0.000% | 732.0 | 507.7 | 20,345,124 | 15,972,670 |
| S2 / char_union / K=80 | 97.534% | 97.020% | 0.99093 | 158.77 | 160.0 | 160.0 | 160.0 | 160 | 0.000% | 732.0 | 507.7 | 350,386,680 | 275,083,634 |
| S2 / exact_address_accent_folded / K=10 | 24.793% | 46.300% | 0.56697 | 0.39 | 1.0 | 2.0 | 3.0 | 10 | 72.120% | 288.0 | 170.4 | 858,895 | 674,306 |
| S2 / exact_address_accent_folded / K=20 | 24.793% | 46.300% | 0.56697 | 0.39 | 1.0 | 2.0 | 3.0 | 10 | 72.120% | 288.0 | 170.4 | 858,895 | 674,306 |
| S2 / exact_address_accent_folded / K=40 | 24.793% | 46.300% | 0.56697 | 0.39 | 1.0 | 2.0 | 3.0 | 10 | 72.120% | 288.0 | 170.4 | 858,895 | 674,306 |
| S2 / exact_address_accent_folded / K=5 | 24.675% | 46.200% | 0.56658 | 0.38 | 1.0 | 2.0 | 3.0 | 5 | 72.120% | 288.0 | 170.4 | 843,888 | 662,525 |
| S2 / exact_address_accent_folded / K=80 | 24.793% | 46.300% | 0.56697 | 0.39 | 1.0 | 2.0 | 3.0 | 10 | 72.120% | 288.0 | 170.4 | 858,895 | 674,306 |
| S2 / exact_address_unicode_preserving / K=10 | 24.793% | 46.300% | 0.56697 | 0.39 | 1.0 | 2.0 | 3.0 | 10 | 72.120% | 288.0 | 170.4 | 858,895 | 674,306 |
| S2 / exact_address_unicode_preserving / K=20 | 24.793% | 46.300% | 0.56697 | 0.39 | 1.0 | 2.0 | 3.0 | 10 | 72.120% | 288.0 | 170.4 | 858,895 | 674,306 |
| S2 / exact_address_unicode_preserving / K=40 | 24.793% | 46.300% | 0.56697 | 0.39 | 1.0 | 2.0 | 3.0 | 10 | 72.120% | 288.0 | 170.4 | 858,895 | 674,306 |
| S2 / exact_address_unicode_preserving / K=5 | 24.675% | 46.200% | 0.56658 | 0.38 | 1.0 | 2.0 | 3.0 | 5 | 72.120% | 288.0 | 170.4 | 843,888 | 662,525 |
| S2 / exact_address_unicode_preserving / K=80 | 24.793% | 46.300% | 0.56697 | 0.39 | 1.0 | 2.0 | 3.0 | 10 | 72.120% | 288.0 | 170.4 | 858,895 | 674,306 |
| S2 / exact_name_accent_folded / K=10 | 23.228% | 41.660% | 0.55490 | 1.83 | 7.0 | 10.0 | 10.0 | 10 | 47.580% | 288.0 | 170.4 | 4,030,979 | 3,164,665 |
| S2 / exact_name_accent_folded / K=20 | 23.907% | 41.920% | 0.56209 | 2.52 | 7.0 | 20.0 | 20.0 | 20 | 47.580% | 288.0 | 170.4 | 5,552,362 | 4,359,081 |
| S2 / exact_name_accent_folded / K=40 | 24.660% | 42.380% | 0.56914 | 3.54 | 7.0 | 31.1 | 40.0 | 40 | 47.580% | 288.0 | 170.4 | 7,818,325 | 6,138,057 |
| S2 / exact_name_accent_folded / K=5 | 22.578% | 41.420% | 0.54942 | 1.34 | 5.0 | 5.0 | 5.0 | 5 | 47.580% | 288.0 | 170.4 | 2,957,582 | 2,321,955 |
| S2 / exact_name_accent_folded / K=80 | 24.941% | 42.520% | 0.57125 | 4.38 | 7.0 | 31.1 | 80.0 | 80 | 47.580% | 288.0 | 170.4 | 9,660,580 | 7,584,385 |
| S2 / exact_name_unicode_preserving / K=10 | 20.112% | 40.720% | 0.52887 | 1.68 | 6.0 | 10.0 | 10.0 | 10 | 50.680% | 288.0 | 170.4 | 3,715,845 | 2,917,258 |
| S2 / exact_name_unicode_preserving / K=20 | 20.629% | 40.980% | 0.53453 | 2.35 | 6.0 | 20.0 | 20.0 | 20 | 50.680% | 288.0 | 170.4 | 5,184,264 | 4,070,092 |
| S2 / exact_name_unicode_preserving / K=40 | 21.382% | 41.440% | 0.54174 | 3.26 | 6.0 | 28.0 | 40.0 | 40 | 50.680% | 288.0 | 170.4 | 7,186,733 | 5,642,203 |
| S2 / exact_name_unicode_preserving / K=5 | 19.610% | 40.520% | 0.52431 | 1.24 | 5.0 | 5.0 | 5.0 | 5 | 50.680% | 288.0 | 170.4 | 2,731,603 | 2,144,543 |
| S2 / exact_name_unicode_preserving / K=80 | 21.618% | 41.540% | 0.54355 | 3.95 | 6.0 | 28.0 | 80.0 | 80 | 50.680% | 288.0 | 170.4 | 8,725,329 | 6,850,132 |
| S2 / name_char_tfidf / K=10 | 61.326% | 65.820% | 0.79646 | 10.00 | 10.0 | 10.0 | 10.0 | 10 | 0.000% | 268.8 | 324.2 | 22,068,210 | 17,325,440 |
| S2 / name_char_tfidf / K=20 | 66.110% | 69.240% | 0.82259 | 20.00 | 20.0 | 20.0 | 20.0 | 20 | 0.000% | 268.8 | 324.2 | 44,136,420 | 34,650,880 |
| S2 / name_char_tfidf / K=40 | 69.920% | 72.040% | 0.84762 | 40.00 | 40.0 | 40.0 | 40.0 | 40 | 0.000% | 268.8 | 324.2 | 88,272,840 | 69,301,760 |
| S2 / name_char_tfidf / K=5 | 55.286% | 61.440% | 0.76572 | 5.00 | 5.0 | 5.0 | 5.0 | 5 | 0.000% | 268.8 | 324.2 | 11,034,105 | 8,662,720 |
| S2 / name_char_tfidf / K=80 | 73.302% | 74.340% | 0.86700 | 80.00 | 80.0 | 80.0 | 80.0 | 80 | 0.000% | 268.8 | 324.2 | 176,545,680 | 138,603,520 |
| S2 / required_union / K=10 | 94.315% | 93.560% | 0.97840 | 19.19 | 20.0 | 20.0 | 24.0 | 28 | 0.000% | 1020.0 | 507.7 | 42,344,923 | 33,244,401 |
| S2 / required_union / K=20 | 95.821% | 95.220% | 0.98420 | 39.15 | 40.0 | 40.0 | 47.0 | 55 | 0.000% | 1020.0 | 507.7 | 86,404,987 | 67,835,335 |
| S2 / required_union / K=40 | 96.663% | 96.120% | 0.98827 | 79.12 | 80.0 | 80.0 | 90.0 | 104 | 0.000% | 1020.0 | 507.7 | 174,605,443 | 137,080,267 |
| S2 / required_union / K=5 | 91.834% | 91.220% | 0.97047 | 9.33 | 10.0 | 11.0 | 12.0 | 15 | 0.000% | 1020.0 | 507.7 | 20,586,550 | 16,162,210 |
| S2 / required_union / K=80 | 97.534% | 97.020% | 0.99093 | 159.05 | 160.0 | 160.0 | 174.0 | 210 | 0.000% | 1020.0 | 507.7 | 351,005,031 | 275,569,093 |
| S3 / address_char_tfidf / K=10 | 78.594% | 79.920% | 0.90143 | 10.00 | 10.0 | 10.0 | 10.0 | 10 | 0.000% | 379.3 | 516.9 | 22,068,210 | 17,325,440 |
| S3 / address_char_tfidf / K=20 | 81.258% | 82.000% | 0.91349 | 20.00 | 20.0 | 20.0 | 20.0 | 20 | 0.000% | 379.3 | 516.9 | 44,136,420 | 34,650,880 |
| S3 / address_char_tfidf / K=40 | 83.682% | 83.880% | 0.92580 | 40.00 | 40.0 | 40.0 | 40.0 | 40 | 0.000% | 379.3 | 516.9 | 88,272,840 | 69,301,760 |
| S3 / address_char_tfidf / K=5 | 73.867% | 76.140% | 0.88542 | 5.00 | 5.0 | 5.0 | 5.0 | 5 | 0.000% | 379.3 | 516.9 | 11,034,105 | 8,662,720 |
| S3 / address_char_tfidf / K=80 | 85.278% | 85.260% | 0.93313 | 79.99 | 80.0 | 80.0 | 80.0 | 80 | 0.000% | 379.3 | 516.9 | 176,529,791 | 138,591,046 |
| S3 / all_views_union / K=10 | 92.157% | 91.640% | 0.96967 | 19.62 | 21.0 | 23.0 | 25.0 | 30 | 0.000% | 993.6 | 516.9 | 43,304,890 | 33,998,057 |
| S3 / all_views_union / K=20 | 94.144% | 93.720% | 0.97726 | 39.60 | 41.0 | 43.0 | 49.0 | 56 | 0.000% | 993.6 | 516.9 | 87,395,408 | 68,612,901 |
| S3 / all_views_union / K=40 | 95.559% | 95.100% | 0.98301 | 79.67 | 81.0 | 84.0 | 94.0 | 110 | 0.000% | 993.6 | 516.9 | 175,820,077 | 138,033,860 |
| S3 / all_views_union / K=5 | 89.101% | 88.800% | 0.96008 | 9.68 | 11.0 | 12.0 | 14.0 | 17 | 0.000% | 993.6 | 516.9 | 21,365,117 | 16,773,451 |
| S3 / all_views_union / K=80 | 96.493% | 95.960% | 0.98757 | 159.62 | 161.0 | 163.0 | 180.0 | 225 | 0.000% | 993.6 | 516.9 | 352,252,327 | 276,548,327 |
| S3 / char_union / K=10 | 91.615% | 91.140% | 0.96680 | 19.21 | 20.0 | 20.0 | 20.0 | 20 | 0.000% | 703.3 | 516.9 | 42,396,121 | 33,284,596 |
| S3 / char_union / K=20 | 93.828% | 93.440% | 0.97572 | 39.10 | 40.0 | 40.0 | 40.0 | 40 | 0.000% | 703.3 | 516.9 | 86,276,108 | 67,734,154 |
| S3 / char_union / K=40 | 95.273% | 94.820% | 0.98135 | 78.99 | 80.0 | 80.0 | 80.0 | 80 | 0.000% | 703.3 | 516.9 | 174,327,384 | 136,861,967 |
| S3 / char_union / K=5 | 88.288% | 88.020% | 0.95592 | 9.37 | 10.0 | 10.0 | 10.0 | 10 | 0.000% | 703.3 | 516.9 | 20,682,768 | 16,237,749 |
| S3 / char_union / K=80 | 96.252% | 95.680% | 0.98628 | 158.88 | 160.0 | 160.0 | 160.0 | 160 | 0.000% | 703.3 | 516.9 | 350,626,782 | 275,272,135 |
| S3 / exact_address_accent_folded / K=10 | 9.213% | 39.140% | 0.46726 | 0.14 | 1.0 | 1.0 | 1.0 | 3 | 86.680% | 290.4 | 283.5 | 307,631 | 241,517 |
| S3 / exact_address_accent_folded / K=20 | 9.213% | 39.140% | 0.46726 | 0.14 | 1.0 | 1.0 | 1.0 | 3 | 86.680% | 290.4 | 283.5 | 307,631 | 241,517 |
| S3 / exact_address_accent_folded / K=40 | 9.213% | 39.140% | 0.46726 | 0.14 | 1.0 | 1.0 | 1.0 | 3 | 86.680% | 290.4 | 283.5 | 307,631 | 241,517 |
| S3 / exact_address_accent_folded / K=5 | 9.213% | 39.140% | 0.46726 | 0.14 | 1.0 | 1.0 | 1.0 | 3 | 86.680% | 290.4 | 283.5 | 307,631 | 241,517 |
| S3 / exact_address_accent_folded / K=80 | 9.213% | 39.140% | 0.46726 | 0.14 | 1.0 | 1.0 | 1.0 | 3 | 86.680% | 290.4 | 283.5 | 307,631 | 241,517 |
| S3 / exact_address_unicode_preserving / K=10 | 9.213% | 39.140% | 0.46726 | 0.14 | 1.0 | 1.0 | 1.0 | 3 | 86.680% | 290.4 | 283.5 | 307,631 | 241,517 |
| S3 / exact_address_unicode_preserving / K=20 | 9.213% | 39.140% | 0.46726 | 0.14 | 1.0 | 1.0 | 1.0 | 3 | 86.680% | 290.4 | 283.5 | 307,631 | 241,517 |
| S3 / exact_address_unicode_preserving / K=40 | 9.213% | 39.140% | 0.46726 | 0.14 | 1.0 | 1.0 | 1.0 | 3 | 86.680% | 290.4 | 283.5 | 307,631 | 241,517 |
| S3 / exact_address_unicode_preserving / K=5 | 9.213% | 39.140% | 0.46726 | 0.14 | 1.0 | 1.0 | 1.0 | 3 | 86.680% | 290.4 | 283.5 | 307,631 | 241,517 |
| S3 / exact_address_unicode_preserving / K=80 | 9.213% | 39.140% | 0.46726 | 0.14 | 1.0 | 1.0 | 1.0 | 3 | 86.680% | 290.4 | 283.5 | 307,631 | 241,517 |
| S3 / exact_name_accent_folded / K=10 | 23.754% | 46.680% | 0.59496 | 2.19 | 10.0 | 10.0 | 10.0 | 10 | 47.800% | 290.4 | 283.5 | 4,838,234 | 3,798,429 |
| S3 / exact_name_accent_folded / K=20 | 24.432% | 47.040% | 0.60120 | 2.96 | 11.0 | 20.0 | 20.0 | 20 | 47.800% | 290.4 | 283.5 | 6,534,838 | 5,130,409 |
| S3 / exact_name_accent_folded / K=40 | 25.004% | 47.320% | 0.60659 | 4.01 | 11.0 | 32.0 | 40.0 | 40 | 47.800% | 290.4 | 283.5 | 8,859,945 | 6,955,818 |
| S3 / exact_name_accent_folded / K=5 | 22.490% | 46.160% | 0.58392 | 1.48 | 5.0 | 5.0 | 5.0 | 5 | 47.800% | 290.4 | 283.5 | 3,266,978 | 2,564,858 |
| S3 / exact_name_accent_folded / K=80 | 25.199% | 47.420% | 0.60819 | 4.87 | 11.0 | 32.0 | 80.0 | 80 | 47.800% | 290.4 | 283.5 | 10,755,163 | 8,443,726 |
| S3 / exact_name_unicode_preserving / K=10 | 20.367% | 45.360% | 0.56754 | 2.02 | 9.0 | 10.0 | 10.0 | 10 | 51.080% | 290.4 | 283.5 | 4,468,813 | 3,508,402 |
| S3 / exact_name_unicode_preserving / K=20 | 20.849% | 45.680% | 0.57245 | 2.73 | 9.0 | 20.0 | 20.0 | 20 | 51.080% | 290.4 | 283.5 | 6,033,007 | 4,736,429 |
| S3 / exact_name_unicode_preserving / K=40 | 21.271% | 45.860% | 0.57626 | 3.66 | 9.0 | 28.1 | 40.0 | 40 | 51.080% | 290.4 | 283.5 | 8,084,027 | 6,346,655 |
| S3 / exact_name_unicode_preserving / K=5 | 19.389% | 44.960% | 0.55861 | 1.39 | 5.0 | 5.0 | 5.0 | 5 | 51.080% | 290.4 | 283.5 | 3,070,571 | 2,410,662 |
| S3 / exact_name_unicode_preserving / K=80 | 21.391% | 45.920% | 0.57733 | 4.38 | 9.0 | 28.1 | 80.0 | 80 | 51.080% | 290.4 | 283.5 | 9,661,021 | 7,584,731 |
| S3 / name_char_tfidf / K=10 | 58.994% | 65.320% | 0.80299 | 10.00 | 10.0 | 10.0 | 10.0 | 10 | 0.000% | 324.0 | 193.3 | 22,068,210 | 17,325,440 |
| S3 / name_char_tfidf / K=20 | 64.685% | 69.280% | 0.83506 | 20.00 | 20.0 | 20.0 | 20.0 | 20 | 0.000% | 324.0 | 193.3 | 44,136,420 | 34,650,880 |
| S3 / name_char_tfidf / K=40 | 68.794% | 71.820% | 0.85792 | 40.00 | 40.0 | 40.0 | 40.0 | 40 | 0.000% | 324.0 | 193.3 | 88,272,840 | 69,301,760 |
| S3 / name_char_tfidf / K=5 | 51.799% | 60.520% | 0.76498 | 5.00 | 5.0 | 5.0 | 5.0 | 5 | 0.000% | 324.0 | 193.3 | 11,034,105 | 8,662,720 |
| S3 / name_char_tfidf / K=80 | 72.663% | 74.680% | 0.88054 | 80.00 | 80.0 | 80.0 | 80.0 | 80 | 0.000% | 324.0 | 193.3 | 176,545,680 | 138,603,520 |
| S3 / required_union / K=10 | 91.660% | 91.180% | 0.96737 | 19.39 | 20.0 | 21.0 | 24.0 | 28 | 0.000% | 993.6 | 516.9 | 42,782,315 | 33,587,791 |
| S3 / required_union / K=20 | 93.828% | 93.440% | 0.97572 | 39.28 | 40.0 | 40.0 | 46.0 | 53 | 0.000% | 993.6 | 516.9 | 86,690,991 | 68,059,872 |
| S3 / required_union / K=40 | 95.273% | 94.820% | 0.98135 | 79.25 | 80.0 | 80.0 | 89.0 | 106 | 0.000% | 993.6 | 516.9 | 174,892,330 | 137,305,498 |
| S3 / required_union / K=5 | 88.424% | 88.080% | 0.95684 | 9.52 | 10.0 | 11.0 | 13.0 | 15 | 0.000% | 993.6 | 516.9 | 21,015,998 | 16,499,363 |
| S3 / required_union / K=80 | 96.252% | 95.680% | 0.98628 | 159.18 | 160.0 | 160.0 | 174.0 | 217 | 0.000% | 993.6 | 516.9 | 351,270,733 | 275,777,691 |
| combined / all_views_union / K=10 | 93.343% | 86.160% | 0.97142 | 38.99 | 42.0 | 44.0 | 50.0 | 56 | 0.000% | 2013.6 | 516.9 | 86,049,247 | 67,556,049 |
| combined / all_views_union / K=20 | 95.050% | 89.580% | 0.97966 | 79.00 | 82.0 | 85.0 | 97.0 | 112 | 0.000% | 2013.6 | 516.9 | 174,345,479 | 136,876,174 |
| combined / all_views_union / K=40 | 96.154% | 91.700% | 0.98506 | 159.13 | 162.0 | 166.1 | 188.0 | 217 | 0.000% | 2013.6 | 516.9 | 351,161,716 | 275,692,104 |
| combined / all_views_union / K=5 | 90.652% | 81.460% | 0.95992 | 19.14 | 22.0 | 24.0 | 26.0 | 31 | 0.000% | 2013.6 | 516.9 | 42,241,202 | 33,162,971 |
| combined / all_views_union / K=80 | 97.041% | 93.280% | 0.98846 | 319.06 | 321.0 | 325.1 | 354.0 | 440 | 0.000% | 2013.6 | 516.9 | 704,101,246 | 552,779,944 |

## Best-union subgroup results

### country

| Group | S1 | Recall | All truth | Oracle F0.5 | Mean candidates |
|---|---:|---:|---:|---:|---:|
| India | 2,407 | 95.447% | 89.946% | 0.98269 | 318.21 |
| US | 2,593 | 98.539% | 96.375% | 0.99383 | 319.85 |
### cardinality_bucket

| Group | S1 | Recall | All truth | Oracle F0.5 | Mean candidates |
|---|---:|---:|---:|---:|---:|
| 0 | 138 | 0.000% | 100.000% | 1.00000 | 322.26 |
| 1 | 774 | 97.933% | 97.933% | 0.97933 | 320.14 |
| 2 | 1,480 | 97.804% | 95.811% | 0.99133 | 319.75 |
| 3 | 1,398 | 96.638% | 91.702% | 0.98749 | 318.64 |
| 4+ | 1,210 | 96.811% | 88.264% | 0.99062 | 317.64 |
### truth_composition

| Group | S1 | Recall | All truth | Oracle F0.5 | Mean candidates |
|---|---:|---:|---:|---:|---:|
| S2_only | 1,721 | 97.529% | 95.061% | 0.99033 | 319.34 |
| S3_only | 1,493 | 96.492% | 92.364% | 0.98215 | 318.87 |
| both_sources | 1,648 | 97.064% | 91.687% | 0.99127 | 318.66 |
| no_match | 138 | 0.000% | 100.000% | 1.00000 | 322.26 |
### address_state

| Group | S1 | Recall | All truth | Oracle F0.5 | Mean candidates |
|---|---:|---:|---:|---:|---:|
| present | 5,000 | 97.041% | 93.280% | 0.98846 | 319.06 |
### target_address_state

| Group | S1 | Recall | All truth | Oracle F0.5 | Mean candidates |
|---|---:|---:|---:|---:|---:|
| all_true_target_addresses_present | 4,337 | 97.540% | 94.582% | 0.99035 | 318.99 |
| any_true_target_missing | 525 | 93.703% | 80.762% | 0.96985 | 318.79 |
| no_true_target | 138 | 0.000% | 100.000% | 1.00000 | 322.26 |
### exact_name_state

| Group | S1 | Recall | All truth | Oracle F0.5 | Mean candidates |
|---|---:|---:|---:|---:|---:|
| available | 2,332 | 98.018% | 94.940% | 0.99523 | 318.87 |
| unavailable | 2,668 | 96.063% | 91.829% | 0.98255 | 319.22 |
### exact_address_state

| Group | S1 | Recall | All truth | Oracle F0.5 | Mean candidates |
|---|---:|---:|---:|---:|---:|
| available | 1,905 | 97.895% | 94.961% | 0.99474 | 318.71 |
| unavailable | 3,095 | 96.462% | 92.246% | 0.98460 | 319.27 |

## Failure analysis and caveats

- India is the weaker country at 95.447% edge recall versus 98.539% for the US.
- S3-only truth is weaker than S2-only truth: 96.492% versus 97.529% edge recall.
- Entities with any true target missing its address are the clearest address failure: 93.703% edge recall and 80.762% all-truth recovery.
- All 5,000 sampled S1 records have a present address, so an S1-missing-address subgroup could not be measured. Truth-side target address state was annotated after the fixed sample was selected and did not affect membership.
- Word TF-IDF was optional and was not retained. The implemented required character channels already exceed the preferred candidate budget before reaching the recall target.
- Address digits are retained only as evidence on generated pairs; no global digit block is used.
- France transfer remains unverified because training truth contains only India and US entities.

## Selection

Best measured union: `combined|all_views_union|K=80` with 97.041% edge recall, 0.98846 oracle macro F0.5, mean 319.06, and p95 325.1 candidates.

No configuration is selected for production: the measured local evaluation did not satisfy every recall/oracle/candidate-volume target.

## Resource and production recommendation

The normalized/exact artifacts and bounded sparse benchmark are safe locally. A fully persisted four-index sparse target build is projected from benchmark bytes and is not launched if it exceeds local free disk. Recommended production tier: 16 vCPU, 64 GiB RAM, 100 GiB SSD. Reproducible command: `python -m entity_resolution.candidate_generation --config configs/candidate_generation_v1.yaml --full-target-evaluation`.

## Next modeling milestone

First improve retrieval on India, S3-only truth, and missing-target-address cases using business-text-only views. Rerun this fixed sample and proceed to leakage-safe pair-feature development only after a union clears the recall and oracle thresholds.
