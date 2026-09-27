from entity_resolution.views import (
    comparison_views,
    digit_sequence,
    legal_suffix_stripped_name,
    normalize_field,
)


def test_comparison_views_preserve_raw_and_address_digits() -> None:
    raw = "  Café—North\t12-B  "

    views = comparison_views(raw)

    assert views.raw_unicode == raw
    assert views.unicode_lower == "  café—north\t12-b  "
    assert views.whitespace_normalized == "café—north 12-b"
    assert views.punctuation_normalized == "café north 12 b"
    assert views.accent_folded == "cafe north 12 b"
    assert views.alphanumeric_compact == "cafenorth12b"
    assert views.word_token_sequence == ("cafe", "north", "12", "b")
    assert views.sorted_token_sequence == ("12", "b", "cafe", "north")
    assert digit_sequence(raw) == ("12",)


def test_legal_suffix_stripping_is_terminal_and_conservative() -> None:
    assert legal_suffix_stripped_name("Acme, Inc.") == "acme"
    assert legal_suffix_stripped_name("Limited Edition") == "limited edition"
    assert legal_suffix_stripped_name("The LLC Cafe") == "the llc cafe"


def test_candidate_normalization_marks_null_like_and_separates_controls() -> None:
    assert normalize_field(" N/A ").is_null_like
    assert normalize_field(" N/A ").unicode_preserving == ""
    assert normalize_field("").is_missing
    controlled = normalize_field("Alpha\x00Beta")
    assert controlled.unicode_preserving == "alpha beta"
    assert controlled.compact_alphanumeric == "alphabeta"
