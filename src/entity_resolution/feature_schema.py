"""Stable, label-free CatBoost pair-feature contract."""

from __future__ import annotations

import pyarrow as pa

SCHEMA_VERSION = 1
IDENTIFIER_COLUMNS = ("s1_id", "target_id", "target_source", "run_id")
CATEGORICAL_FEATURES = ("target_source", "strongest_channel")

_FLOATS = (
    "retrieval_score",
    "fusion_score",
    "reciprocal_rank",
    "support_best_score",
    "support_mean_score",
    "support_min_score",
    "support_max_score",
    "support_mean_rank",
    "name_char_similarity",
    "name_accent_char_similarity",
    "name_token_jaccard",
    "name_token_containment_s1",
    "name_token_containment_target",
    "name_token_set_similarity",
    "name_token_sort_similarity",
    "name_prefix_similarity",
    "name_suffix_similarity",
    "name_char_3gram_jaccard",
    "name_token_count_ratio",
    "name_char_length_ratio",
    "address_char_similarity",
    "address_token_jaccard",
    "address_token_containment_s1",
    "address_token_containment_target",
    "address_number_overlap",
    "address_token_count_ratio",
    "address_char_length_ratio",
    "combined_string_similarity",
    "retrieval_name_interaction",
    "retrieval_address_interaction",
)
_INT16 = (
    "channel_rank",
    "channel_support_count",
    "exact_support_count",
    "support_min_rank",
    "source_support_count",
    "s2_support_count",
    "s3_support_count",
    "name_support_count",
    "address_support_count",
    "exact_support_family_count",
    "fuzzy_support_count",
    "name_token_count_diff",
    "name_char_length_diff",
    "address_token_count_diff",
    "address_char_length_diff",
)
_INT8 = (
    "exact_match",
    "has_source_support",
    "has_s2_support",
    "has_s3_support",
    "has_name_support",
    "has_address_support",
    "has_exact_support",
    "has_fuzzy_support",
    "independent_channel_agreement",
    "name_s1_missing",
    "name_target_missing",
    "name_s1_null_like",
    "name_target_null_like",
    "name_raw_exact",
    "name_normalized_exact",
    "name_casefold_exact",
    "name_accent_exact",
    "name_punctuation_agreement",
    "name_legal_suffix_agreement",
    "name_initials_agreement",
    "name_acronym_agreement",
    "name_digit_agreement",
    "address_s1_missing",
    "address_target_missing",
    "address_s1_null_like",
    "address_target_null_like",
    "address_raw_exact",
    "address_normalized_exact",
    "address_accent_exact",
    "address_number_exact",
    "address_house_number_agreement",
    "address_postal_agreement",
    "both_names_usable",
    "both_addresses_usable",
    "exact_name_and_address",
    "strong_name_weak_address",
    "strong_address_weak_name",
    "exact_name_conflicting_address_digits",
    "country_agreement",
    "country_both_present",
)

NUMERIC_FEATURES = (*_FLOATS, *_INT16, *_INT8)
FEATURE_DTYPES = {
    **dict.fromkeys(_FLOATS, "float32"),
    **dict.fromkeys(_INT16, "int16"),
    **dict.fromkeys(_INT8, "int8"),
    "strongest_channel": "string",
    "target_source": "string",
}
OUTPUT_SCHEMA = pa.schema(
    [(name, pa.string()) for name in IDENTIFIER_COLUMNS]
    + [("strongest_channel", pa.string())]
    + [(name, pa.float32()) for name in _FLOATS]
    + [(name, pa.int16()) for name in _INT16]
    + [(name, pa.int8()) for name in _INT8]
)


def feature_names() -> tuple[str, ...]:
    """Ordered model inputs, with categorical columns included once."""
    return (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)
