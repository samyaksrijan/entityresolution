"""Non-destructive text comparison views for entity resolution."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_ALNUM_RE = re.compile(r"[^\w]", flags=re.UNICODE)
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_DIGIT_RE = re.compile(r"\d+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+")
NULL_LIKE_VALUES = frozenset({"null", "none", "nan", "n/a", "na", "nil", "<null>", "missing"})

# Deliberately conservative: only terminal, standalone corporate designators.
_LEGAL_SUFFIX_RE = re.compile(
    r"(?:\s|,)+(?:inc(?:orporated)?|corp(?:oration)?|co(?:mpany)?|llc|ltd|limited|"
    r"llp|plc|pvt(?:\s+ltd)?|private\s+limited)\.?$",
    flags=re.IGNORECASE,
)


def raw_unicode(value: object) -> str:
    """Return the raw value as text; missing values become an empty string."""
    return "" if value is None else str(value)


def unicode_lower(value: object) -> str:
    """Apply Unicode NFKC normalization and case folding."""
    separated = _CONTROL_RE.sub(" ", raw_unicode(value))
    return unicodedata.normalize("NFKC", separated).casefold()


def whitespace_normalized(value: object) -> str:
    """Normalize Unicode/case and collapse surrounding/internal whitespace."""
    return _SPACE_RE.sub(" ", unicode_lower(value)).strip()


def punctuation_normalized(value: object) -> str:
    """Replace punctuation with spaces while preserving letters and digits."""
    return _SPACE_RE.sub(" ", _PUNCT_RE.sub(" ", whitespace_normalized(value))).strip()


def accent_folded(value: object) -> str:
    """Remove combining marks after Unicode decomposition."""
    normalized = unicodedata.normalize("NFKD", punctuation_normalized(value))
    return "".join(char for char in normalized if not unicodedata.combining(char))


def alphanumeric_compact(value: object) -> str:
    """Return accent-folded letters/digits with separators removed."""
    return _ALNUM_RE.sub("", accent_folded(value))


def word_tokens(value: object) -> tuple[str, ...]:
    """Return the normalized word-token sequence."""
    return tuple(_WORD_RE.findall(accent_folded(value)))


def sorted_tokens(value: object) -> tuple[str, ...]:
    """Return normalized word tokens in deterministic sorted order."""
    return tuple(sorted(word_tokens(value)))


def digit_sequence(value: object) -> tuple[str, ...]:
    """Return digit runs in their original order; digits are never discarded."""
    return tuple(_DIGIT_RE.findall(unicode_lower(value)))


def legal_suffix_stripped_name(value: object) -> str:
    """Strip one unambiguous terminal legal suffix from a normalized name."""
    return _LEGAL_SUFFIX_RE.sub("", punctuation_normalized(value)).strip(" ,.")


@dataclass(frozen=True)
class ComparisonViews:
    """All supported views of one raw text value."""

    raw_unicode: str
    unicode_lower: str
    whitespace_normalized: str
    punctuation_normalized: str
    accent_folded: str
    alphanumeric_compact: str
    word_token_sequence: tuple[str, ...]
    sorted_token_sequence: tuple[str, ...]
    digit_sequence: tuple[str, ...]
    legal_suffix_stripped: str


@dataclass(frozen=True)
class NormalizedField:
    """Candidate-generation views with explicit missing/null-like state."""

    raw: str
    unicode_preserving: str
    accent_folded: str
    compact_alphanumeric: str
    tokens: tuple[str, ...]
    digit_tokens: tuple[str, ...]
    is_missing: bool
    is_null_like: bool


def normalize_field(value: object) -> NormalizedField:
    """Normalize one field without turning null-like literals into blocking keys."""
    raw = raw_unicode(value)
    stripped = _SPACE_RE.sub(" ", _CONTROL_RE.sub(" ", raw)).strip()
    null_like = stripped.casefold() in NULL_LIKE_VALUES
    missing = not stripped or null_like
    if missing:
        return NormalizedField(
            raw=raw,
            unicode_preserving="",
            accent_folded="",
            compact_alphanumeric="",
            tokens=(),
            digit_tokens=(),
            is_missing=not stripped,
            is_null_like=null_like,
        )
    unicode_view = punctuation_normalized(stripped)
    folded = accent_folded(stripped)
    return NormalizedField(
        raw=raw,
        unicode_preserving=unicode_view,
        accent_folded=folded,
        compact_alphanumeric=alphanumeric_compact(stripped),
        tokens=word_tokens(stripped),
        digit_tokens=digit_sequence(stripped),
        is_missing=False,
        is_null_like=False,
    )


def comparison_views(value: object, *, is_name: bool = False) -> ComparisonViews:
    """Build immutable comparison views without mutating the source value."""
    raw = raw_unicode(value)
    return ComparisonViews(
        raw_unicode=raw,
        unicode_lower=unicode_lower(raw),
        whitespace_normalized=whitespace_normalized(raw),
        punctuation_normalized=punctuation_normalized(raw),
        accent_folded=accent_folded(raw),
        alphanumeric_compact=alphanumeric_compact(raw),
        word_token_sequence=word_tokens(raw),
        sorted_token_sequence=sorted_tokens(raw),
        digit_sequence=digit_sequence(raw),
        legal_suffix_stripped=(
            legal_suffix_stripped_name(raw) if is_name else punctuation_normalized(raw)
        ),
    )
