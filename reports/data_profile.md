# Dataset profile

Generated with deterministic seed `20260927`. Row-level missingness, ASCII/digit rates, countries, and hash-equivalent duplicate counts are full scans. Lengths, token counts, scripts, accents, and token frequencies use the documented deterministic 2% sample. A 64-bit pandas hash was used for duplicate equivalence; collision risk is negligible but non-zero.

## Source profiles

| Split/source | Rows | File MiB | DataFrame MiB | Missing name | Missing address | Both missing | Countries |
|---|---:|---:|---:|---:|---:|---:|---|
| train_source1 | 2,206,821 | 200.3 | 259.3 | 0.000% | 0.000% | 0.000% | {'India': 883188, 'US': 1323633} |
| train_source2 | 5,034,616 | 466.6 | 601.1 | 0.000% | 3.356% | 0.000% | {'India': 2017799, 'US': 3016817} |
| train_source3 | 5,285,603 | 480.4 | 621.5 | 0.000% | 3.328% | 0.000% | {'India': 2115547, 'US': 3170056} |
| test_source1 | 1,732,544 | 166.9 | 213.2 | 0.000% | 0.000% | 0.000% | {'France': 259452, 'India': 809986, 'US': 663106} |
| test_source2 | 4,887,273 | 485.9 | 616.4 | 0.000% | 2.648% | 0.000% | {'France': 703378, 'India': 2312565, 'US': 1871330} |
| test_source3 | 5,082,316 | 482.6 | 618.3 | 0.000% | 2.678% | 0.000% | {'France': 731615, 'India': 2405000, 'US': 1945701} |

## Country partitions

| Split/source | Country | Rows | Missing name | Missing address | Non-ASCII name | Address digits | Sample mean name/address length |
|---|---|---:|---:|---:|---:|---:|---:|
| train_source1 | US | 1,323,633 | 0.000% | 0.000% | 0.000% | 100.000% | 22.44/34.95 |
| train_source1 | India | 883,188 | 0.000% | 0.000% | 0.000% | 91.278% | 26.40/77.53 |
| train_source2 | India | 2,017,799 | 0.000% | 2.867% | 27.874% | 91.317% | 27.44/68.26 |
| train_source2 | US | 3,016,817 | 0.000% | 3.683% | 6.701% | 90.197% | 23.56/31.54 |
| train_source3 | US | 3,170,056 | 0.000% | 3.501% | 6.800% | 91.296% | 23.92/38.46 |
| train_source3 | India | 2,115,547 | 0.000% | 3.070% | 18.490% | 90.055% | 27.03/59.25 |
| test_source1 | US | 663,106 | 0.000% | 0.000% | 0.000% | 100.000% | 22.47/35.00 |
| test_source1 | France | 259,452 | 0.000% | 0.000% | 15.721% | 99.580% | 19.37/50.08 |
| test_source1 | India | 809,986 | 0.000% | 0.000% | 0.000% | 91.266% | 26.28/77.75 |
| test_source2 | India | 2,312,565 | 0.000% | 2.282% | 27.615% | 93.032% | 28.30/68.88 |
| test_source2 | France | 703,378 | 0.000% | 3.062% | 24.541% | 93.148% | 21.05/39.40 |
| test_source2 | US | 1,871,330 | 0.000% | 2.945% | 6.248% | 92.054% | 24.32/31.87 |
| test_source3 | India | 2,405,000 | 0.000% | 2.463% | 18.208% | 91.879% | 27.92/59.30 |
| test_source3 | France | 731,615 | 0.000% | 2.944% | 23.945% | 93.385% | 21.10/39.94 |
| test_source3 | US | 1,945,701 | 0.000% | 2.843% | 6.395% | 92.919% | 24.68/38.87 |

## Detailed measurements

### train_source1

Sampled rows: 44,227. Name length mean/median/p95: 24.02/24.0/37.0; address: 51.97/41.0/102.0.

Name token mean: 3.57; address token mean: 8.49. Non-ASCII name/address: 0.000%/0.025%. Digit-bearing name/address: 1.613%/96.509%.

Script counts in sample: `{'address:Latin': 44227, 'name:Latin': 44227}`. Accent rates in sample: name 0.000%, address 0.029%.

Duplicate metrics: `{'exact_name': {'unique': 1539229, 'duplicate_excess': 667592, 'duplicate_groups': 177793, 'rows_in_duplicate_groups': 845385}, 'exact_address': {'unique': 2130606, 'duplicate_excess': 76215, 'duplicate_groups': 40089, 'rows_in_duplicate_groups': 116304}, 'normalized_name': {'unique': 1520684, 'duplicate_excess': 686137, 'duplicate_groups': 180757, 'rows_in_duplicate_groups': 866894}, 'normalized_address': {'unique': 2130168, 'duplicate_excess': 76653, 'duplicate_groups': 40515, 'rows_in_duplicate_groups': 117168}, 'record': {'unique': 2206821, 'duplicate_excess': 0, 'duplicate_groups': 0, 'rows_in_duplicate_groups': 0}}`.

### train_source2

Sampled rows: 100,454. Name length mean/median/p95: 25.13/25.0/40.0; address: 46.37/37.0/96.0.

Name token mean: 4.50; address token mean: 8.09. Non-ASCII name/address: 15.187%/9.503%. Digit-bearing name/address: 5.078%/90.646%.

Script counts in sample: `{'address:Latin': 87406, 'address:Mixed': 9731, 'address:NoLetters': 3317, 'name:Devanagari': 5111, 'name:Latin': 91025, 'name:Mixed': 368, 'name:NoLetters': 8, 'name:OtherLetter': 3942}`. Accent rates in sample: name 14.834%, address 7.436%.

Duplicate metrics: `{'exact_name': {'unique': 4402009, 'duplicate_excess': 632607, 'duplicate_groups': 239779, 'rows_in_duplicate_groups': 872386}, 'exact_address': {'unique': 4337262, 'duplicate_excess': 697354, 'duplicate_groups': 421473, 'rows_in_duplicate_groups': 1118827}, 'normalized_name': {'unique': 3824065, 'duplicate_excess': 1210551, 'duplicate_groups': 330243, 'rows_in_duplicate_groups': 1540794}, 'normalized_address': {'unique': 4281382, 'duplicate_excess': 753234, 'duplicate_groups': 461195, 'rows_in_duplicate_groups': 1214429}, 'record': {'unique': 5034616, 'duplicate_excess': 0, 'duplicate_groups': 0, 'rows_in_duplicate_groups': 0}}`.

### train_source3

Sampled rows: 105,772. Name length mean/median/p95: 25.16/25.0/42.0; address: 46.75/42.0/91.0.

Name token mean: 4.12; address token mean: 7.89. Non-ASCII name/address: 11.479%/9.017%. Digit-bearing name/address: 5.124%/90.799%.

Script counts in sample: `{'address:Latin': 92904, 'address:Mixed': 9331, 'address:NoLetters': 3534, 'address:OtherLetter': 3, 'name:Devanagari': 2738, 'name:Latin': 100292, 'name:Mixed': 659, 'name:NoLetters': 9, 'name:OtherLetter': 2074}`. Accent rates in sample: name 11.180%, address 6.845%.

Duplicate metrics: `{'exact_name': {'unique': 4651609, 'duplicate_excess': 633994, 'duplicate_groups': 258276, 'rows_in_duplicate_groups': 892270}, 'exact_address': {'unique': 4632765, 'duplicate_excess': 652838, 'duplicate_groups': 382891, 'rows_in_duplicate_groups': 1035729}, 'normalized_name': {'unique': 4132933, 'duplicate_excess': 1152670, 'duplicate_groups': 359575, 'rows_in_duplicate_groups': 1512245}, 'normalized_address': {'unique': 4614551, 'duplicate_excess': 671052, 'duplicate_groups': 393894, 'rows_in_duplicate_groups': 1064946}, 'record': {'unique': 5285603, 'duplicate_excess': 0, 'duplicate_groups': 0, 'rows_in_duplicate_groups': 0}}`.

### test_source1

Sampled rows: 34,920. Name length mean/median/p95: 23.78/23.0/36.0; address: 57.23/50.0/105.0.

Name token mean: 3.52; address token mean: 9.35. Non-ASCII name/address: 2.354%/4.260%. Digit-bearing name/address: 1.166%/95.854%.

Script counts in sample: `{'address:Latin': 34920, 'name:Latin': 34920}`. Accent rates in sample: name 2.403%, address 4.098%.

Duplicate metrics: `{'exact_name': {'unique': 1238867, 'duplicate_excess': 493677, 'duplicate_groups': 129955, 'rows_in_duplicate_groups': 623632}, 'exact_address': {'unique': 1677483, 'duplicate_excess': 55061, 'duplicate_groups': 31396, 'rows_in_duplicate_groups': 86457}, 'normalized_name': {'unique': 1228140, 'duplicate_excess': 504404, 'duplicate_groups': 132124, 'rows_in_duplicate_groups': 636528}, 'normalized_address': {'unique': 1673947, 'duplicate_excess': 58597, 'duplicate_groups': 34220, 'rows_in_duplicate_groups': 92817}, 'record': {'unique': 1732544, 'duplicate_excess': 0, 'duplicate_groups': 0, 'rows_in_duplicate_groups': 0}}`.

### test_source2

Sampled rows: 98,116. Name length mean/median/p95: 25.74/25.0/42.0; address: 50.50/43.0/99.0.

Name token mean: 4.75; address token mean: 8.82. Non-ASCII name/address: 18.991%/14.746%. Digit-bearing name/address: 3.888%/92.674%.

Script counts in sample: `{'address:Latin': 84242, 'address:Mixed': 11225, 'address:NoLetters': 2649, 'name:Devanagari': 5829, 'name:Latin': 87170, 'name:Mixed': 421, 'name:NoLetters': 3, 'name:OtherLetter': 4693}`. Accent rates in sample: name 18.707%, address 11.144%.

Duplicate metrics: `{'exact_name': {'unique': 4311041, 'duplicate_excess': 576232, 'duplicate_groups': 223103, 'rows_in_duplicate_groups': 799335}, 'exact_address': {'unique': 4224784, 'duplicate_excess': 662489, 'duplicate_groups': 444699, 'rows_in_duplicate_groups': 1107188}, 'normalized_name': {'unique': 3744675, 'duplicate_excess': 1142598, 'duplicate_groups': 288523, 'rows_in_duplicate_groups': 1431121}, 'normalized_address': {'unique': 4145360, 'duplicate_excess': 741913, 'duplicate_groups': 503779, 'rows_in_duplicate_groups': 1245692}, 'record': {'unique': 4887273, 'duplicate_excess': 0, 'duplicate_groups': 0, 'rows_in_duplicate_groups': 0}}`.

### test_source3

Sampled rows: 101,231. Name length mean/median/p95: 25.70/25.0/42.0; address: 48.71/43.0/94.0.

Name token mean: 4.30; address token mean: 8.46. Non-ASCII name/address: 14.511%/14.348%. Digit-bearing name/address: 3.984%/92.494%.

Script counts in sample: `{'address:Devanagari': 1, 'address:Latin': 87410, 'address:Mixed': 11141, 'address:NoLetters': 2679, 'name:Devanagari': 3146, 'name:Latin': 94783, 'name:Mixed': 808, 'name:NoLetters': 4, 'name:OtherLetter': 2490}`. Accent rates in sample: name 14.288%, address 10.881%.

Duplicate metrics: `{'exact_name': {'unique': 4521929, 'duplicate_excess': 560387, 'duplicate_groups': 238156, 'rows_in_duplicate_groups': 798543}, 'exact_address': {'unique': 4456436, 'duplicate_excess': 625880, 'duplicate_groups': 404757, 'rows_in_duplicate_groups': 1030637}, 'normalized_name': {'unique': 4028232, 'duplicate_excess': 1054084, 'duplicate_groups': 316873, 'rows_in_duplicate_groups': 1370957}, 'normalized_address': {'unique': 4414644, 'duplicate_excess': 667672, 'duplicate_groups': 432087, 'rows_in_duplicate_groups': 1099759}, 'record': {'unique': 5082316, 'duplicate_excess': 0, 'duplicate_groups': 0, 'rows_in_duplicate_groups': 0}}`.

## Ground-truth structure

`{'rows': 2206821, 'positive_edges': 7638365, 'singletons': 123247, 'singleton_rate': 0.05584820880352326, 'cardinality': {'4+': 1058364, '3': 530841, '2': 375212, '0': 123247, '1': 119157}, 'per_source_edges': {'S1-S2': 3693619, 'S1-S3': 3944746}, 'matched_in_both_sources': 1776047, 'unknown_source1_references': 0, 'unknown_target_references': 0, 'duplicate_ids_within_lists': 0, 'source1_ids_as_targets': 0, 'conflicting_target_ownership': 0, 'by_country': {'India': {'s1_entities': 883188, 'positive_edges': 3059843}, 'US': {'s1_entities': 1323633, 'positive_edges': 4578522}}, 'by_missingness': {'complete': {'s1_entities': 2206821, 'positive_edges': 7638365}}}`

## Positive-pair analysis

`{'rows': 7638365, 'group_counts': {'S2|US': 2213074, 'S3|US': 2365448, 'S2|India': 1480545, 'S3|India': 1579298}, 'group_rates': {'S2|US': {'name_raw_exact': 0.06102823493475591, 'address_raw_exact': 2.2593008638662784e-06, 'name_normalized_exact': 0.25811563463309406, 'address_normalized_exact': 0.13296663374112208, 'digit_conflict': 0.10678721091115796}, 'S3|US': {'name_raw_exact': 0.05754047436257318, 'address_raw_exact': 0.044258846527169486, 'name_normalized_exact': 0.2584533669731907, 'address_normalized_exact': 0.044258846527169486, 'digit_conflict': 0.08888971560566962}, 'S2|India': {'name_raw_exact': 0.026717864029799837, 'address_raw_exact': 0.0, 'name_normalized_exact': 0.14912076296228755, 'address_normalized_exact': 0.11256868247841166, 'digit_conflict': 0.0348304171774583}, 'S3|India': {'name_raw_exact': 0.027475498607609204, 'address_raw_exact': 0.04139624060816895, 'name_normalized_exact': 0.16806581151878872, 'address_normalized_exact': 0.041631788300877985, 'digit_conflict': 0.03881218110831521}}, 'feature_summary_by_source': {'rows': 7638365, 'group_field': 'target_source', 'group_counts': {'S2': 3693619, 'S3': 3944746}, 'group_means': {'S2': {'country_agreement': 1.0, 'name_raw_exact': 0.0472753145356898, 'address_raw_exact': 1.3536859107558198e-06, 'name_normalized_exact': 0.2144262848983612, 'address_normalized_exact': 0.12479034789457169, 'name_char_similarity': 0.7877299154982876, 'address_char_similarity': 0.7909783919675036, 'name_token_jaccard': 0.6331765542107359, 'address_token_jaccard': 0.6773065680462325, 'digit_overlap': 0.6904163309064071, 'digit_conflict': 0.0779441517925915, 's1_name_missing': 0.0, 'target_name_missing': 0.0, 's1_address_missing': 0.0, 'target_address_missing': 0.044721450696457866, 'name_length_ratio': 0.8711805988340325, 'address_length_ratio': 0.8436769220490816}, 'S3': {'country_agreement': 1.0, 'name_raw_exact': 0.04550381697579515, 'address_raw_exact': 0.04311278850400001, 'name_normalized_exact': 0.2222662751923698, 'address_normalized_exact': 0.043207091153650956, 'name_char_similarity': 0.8034860653443282, 'address_char_similarity': 0.7229672204091876, 'name_token_jaccard': 0.649861154084417, 'address_token_jaccard': 0.5202354194914263, 'digit_overlap': 0.7127339520893555, 'digit_conflict': 0.06884093424519601, 's1_name_missing': 0.0, 'target_name_missing': 0.0, 's1_address_missing': 0.0, 'target_address_missing': 0.043560219086349286, 'name_length_ratio': 0.8483918954458285, 'address_length_ratio': 0.7939177832439014}}, 'columns': ['source1_entity_id', 'matched_entity_id', 'target_source', 'country', 'country_agreement', 'name_raw_exact', 'address_raw_exact', 'name_normalized_exact', 'address_normalized_exact', 'name_char_similarity', 'address_char_similarity', 'name_token_jaccard', 'address_token_jaccard', 'digit_overlap', 'digit_conflict', 's1_name_missing', 'target_name_missing', 's1_address_missing', 'target_address_missing', 'name_length_ratio', 'address_length_ratio']}, 'feature_summary_by_country': {'rows': 7638365, 'group_field': 'country', 'group_counts': {'US': 4578522, 'India': 3059843}, 'group_means': {'US': {'country_agreement': 1.0, 'name_raw_exact': 0.05922631801266872, 'address_raw_exact': 0.022866986333144188, 'name_normalized_exact': 0.2582901206983389, 'address_normalized_exact': 0.08713663492279823, 'name_char_similarity': 0.8584898636793491, 'address_char_similarity': 0.7656607784937465, 'name_token_jaccard': 0.7040401284506476, 'address_token_jaccard': 0.555243909164275, 'digit_overlap': 0.6947176517620646, 'digit_conflict': 0.09754064739669265, 's1_name_missing': 0.0, 'target_name_missing': 0.0, 's1_address_missing': 0.0, 'target_address_missing': 0.047380137083539185, 'name_length_ratio': 0.8616666363749876, 'address_length_ratio': 0.8370691893057498}, 'India': {'country_agreement': 1.0, 'name_raw_exact': 0.027108907221710395, 'address_raw_exact': 0.021366128915764632, 'name_normalized_exact': 0.1588990023344335, 'address_normalized_exact': 0.07595553105175658, 'name_char_similarity': 0.7021627898695275, 'address_char_similarity': 0.7411818675435751, 'name_token_jaccard': 0.5486513269220264, 'address_token_jaccard': 0.6574561326129641, 'digit_overlap': 0.7127520045747169, 'digit_conflict': 0.03688555262475885, 's1_name_missing': 0.0, 'target_name_missing': 0.0, 's1_address_missing': 0.0, 'target_address_missing': 0.03924613125575397, 'name_length_ratio': 0.8560374166152409, 'address_length_ratio': 0.789414819702748}}, 'columns': ['source1_entity_id', 'matched_entity_id', 'target_source', 'country', 'country_agreement', 'name_raw_exact', 'address_raw_exact', 'name_normalized_exact', 'address_normalized_exact', 'name_char_similarity', 'address_char_similarity', 'name_token_jaccard', 'address_token_jaccard', 'digit_overlap', 'digit_conflict', 's1_name_missing', 'target_name_missing', 's1_address_missing', 'target_address_missing', 'name_length_ratio', 'address_length_ratio']}}`

## Negative-pair comparison samples

`{'counts': {'random_negative': 500, 'similar_name_negative': 500, 'similar_address_negative': 500, 'same_normalized_name_negative': 500, 'digit_conflict_negative': 500}, 'feature_summary': {'rows': 2500, 'group_field': 'sample_type', 'group_counts': {'random_negative': 500, 'similar_name_negative': 500, 'similar_address_negative': 500, 'same_normalized_name_negative': 500, 'digit_conflict_negative': 500}, 'group_means': {'random_negative': {'country_agreement': 0.516, 'name_raw_exact': 0.0, 'address_raw_exact': 0.0, 'name_normalized_exact': 0.0, 'address_normalized_exact': 0.0, 'name_char_similarity': 0.3196873343232101, 'address_char_similarity': 0.31642309725631074, 'name_token_jaccard': 0.016303174603174604, 'address_token_jaccard': 0.009592384994287847, 'digit_overlap': 0.004583333333333333, 'digit_conflict': 0.878, 's1_name_missing': 0.0, 'target_name_missing': 0.0, 's1_address_missing': 0.0, 'target_address_missing': 0.032, 'name_length_ratio': 0.6912353854429969, 'address_length_ratio': 0.6346441152939902}, 'similar_name_negative': {'country_agreement': 0.984, 'name_raw_exact': 0.0, 'address_raw_exact': 0.0, 'name_normalized_exact': 0.0, 'address_normalized_exact': 0.0, 'name_char_similarity': 0.7169405958293925, 'address_char_similarity': 0.34950525906632346, 'name_token_jaccard': 0.34968809523809524, 'address_token_jaccard': 0.022230600174574733, 'digit_overlap': 0.009702380952380952, 'digit_conflict': 0.856, 's1_name_missing': 0.0, 'target_name_missing': 0.0, 's1_address_missing': 0.0, 'target_address_missing': 0.04, 'name_length_ratio': 0.8143095277108388, 'address_length_ratio': 0.7717653765565634}, 'similar_address_negative': {'country_agreement': 1.0, 'name_raw_exact': 0.0, 'address_raw_exact': 0.0, 'name_normalized_exact': 0.0, 'address_normalized_exact': 0.0, 'name_char_similarity': 0.3115631173882292, 'address_char_similarity': 0.7016879622921894, 'name_token_jaccard': 0.0415111111111111, 'address_token_jaccard': 0.3460458458050388, 'digit_overlap': 0.1514079365079365, 'digit_conflict': 0.68, 's1_name_missing': 0.0, 'target_name_missing': 0.0, 's1_address_missing': 0.0, 'target_address_missing': 0.0, 'name_length_ratio': 0.6959620131457052, 'address_length_ratio': 0.8431562129060981}, 'same_normalized_name_negative': {'country_agreement': 0.998, 'name_raw_exact': 0.42, 'address_raw_exact': 0.0, 'name_normalized_exact': 1.0, 'address_normalized_exact': 0.0, 'name_char_similarity': 1.0, 'address_char_similarity': 0.34280106770229923, 'name_token_jaccard': 1.0, 'address_token_jaccard': 0.014804189524683136, 'digit_overlap': 0.0032523809523809523, 'digit_conflict': 0.882, 's1_name_missing': 0.0, 'target_name_missing': 0.0, 's1_address_missing': 0.0, 'target_address_missing': 0.028, 'name_length_ratio': 0.9742690226700932, 'address_length_ratio': 0.7852006128066593}, 'digit_conflict_negative': {'country_agreement': 0.986, 'name_raw_exact': 0.002, 'address_raw_exact': 0.0, 'name_normalized_exact': 0.002, 'address_normalized_exact': 0.0, 'name_char_similarity': 0.6725323344541883, 'address_char_similarity': 0.3605427334822587, 'name_token_jaccard': 0.2965175324675325, 'address_token_jaccard': 0.016799508387638922, 'digit_overlap': 0.0, 'digit_conflict': 1.0, 's1_name_missing': 0.0, 'target_name_missing': 0.0, 's1_address_missing': 0.0, 'target_address_missing': 0.0, 'name_length_ratio': 0.7932970925835584, 'address_length_ratio': 0.8172246669120102}}, 'columns': ['source1_entity_id', 'matched_entity_id', 'target_source', 'country', 'country_agreement', 'name_raw_exact', 'address_raw_exact', 'name_normalized_exact', 'address_normalized_exact', 'name_char_similarity', 'address_char_similarity', 'name_token_jaccard', 'address_token_jaccard', 'digit_overlap', 'digit_conflict', 's1_name_missing', 'target_name_missing', 's1_address_missing', 'target_address_missing', 'name_length_ratio', 'address_length_ratio', 'sample_type']}}`. Samples are deterministic verified nonmatches and contain identifiers plus comparison features, not raw business text.

## Train/test shift

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
