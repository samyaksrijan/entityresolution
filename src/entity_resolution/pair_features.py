"""Deterministic pair features from deduplicated candidate rows and source text."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from entity_resolution.feature_schema import FEATURE_DTYPES, OUTPUT_SCHEMA
from entity_resolution.views import comparison_views, normalize_field

_POSTAL_RE = re.compile(r"(?<!\d)\d{5}(?:[- ]?\d{4})?\s*$")
_normalize = lru_cache(maxsize=20000)(normalize_field)
_compare = lru_cache(maxsize=20000)(comparison_views)


@dataclass(frozen=True)
class SourceRecord:
    source: str
    name: str
    address: str
    country: str


def _support(row: object, has_json: bool) -> list[dict[str, object]]:
    if not has_json:
        return [
            {
                "channel": row.channel,
                "score": row.retrieval_score,
                "rank": row.channel_rank,
                "exact": row.exact_match,
            }
        ]
    try:
        evidence = json.loads(row.support_json)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"malformed support_json for {row.s1_id}/{row.target_id}") from exc
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("support_json must be a nonempty list")
    channels: set[str] = set()
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"channel", "score", "rank", "exact"}:
            raise ValueError("invalid support_json observation")
        channel, score, rank, exact = (item[k] for k in ("channel", "score", "rank", "exact"))
        if (
            not isinstance(channel, str)
            or not channel
            or channel in channels
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 1
            or rank > 32767
            or not isinstance(exact, bool)
        ):
            raise ValueError("invalid support_json observation")
        channels.add(channel)
    if len(evidence) > 32767:
        raise ValueError("too many support_json observations")
    if row.channel not in channels or bool(row.exact_match) != any(x["exact"] for x in evidence):
        raise ValueError("support_json disagrees with primary candidate")
    primary = next(item for item in evidence if item["channel"] == row.channel)
    if (
        primary["rank"] != row.channel_rank
        or primary["exact"] != bool(row.exact_match)
        or not math.isclose(float(primary["score"]), float(row.retrieval_score), abs_tol=1e-6)
    ):
        raise ValueError("support_json primary score or rank disagrees with candidate")
    return evidence


def _ratio(a: str, b: str) -> float:
    return fuzz.ratio(a, b) / 100 if a and b else 0.0


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _containment(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a) if a and b else 0.0


def _length_ratio(a: int, b: int) -> float:
    return min(a, b) / max(a, b) if max(a, b) else 0.0


def _ngrams(value: str) -> set[str]:
    return {value[i : i + 3] for i in range(len(value) - 2)}


def _terminal_postal(value: str) -> str:
    """Extract a conservative US-style terminal postal token from a street address."""
    match = _POSTAL_RE.search(value)
    return (
        match.group().strip() if match and any(c.isalpha() for c in value[: match.start()]) else ""
    )


def _field_features(left: str, right: str, prefix: str) -> dict[str, float | int]:
    a, b = _normalize(left), _normalize(right)
    av, bv = (
        _compare(left, is_name=prefix == "name"),
        _compare(right, is_name=prefix == "name"),
    )
    usable = not (a.is_missing or a.is_null_like or b.is_missing or b.is_null_like)
    at, bt = set(a.tokens), set(b.tokens)
    values: dict[str, float | int] = {
        f"{prefix}_s1_missing": int(a.is_missing),
        f"{prefix}_target_missing": int(b.is_missing),
        f"{prefix}_s1_null_like": int(a.is_null_like),
        f"{prefix}_target_null_like": int(b.is_null_like),
        f"{prefix}_raw_exact": int(usable and left == right),
        f"{prefix}_normalized_exact": int(usable and a.unicode_preserving == b.unicode_preserving),
        f"{prefix}_accent_exact": int(usable and a.accent_folded == b.accent_folded),
        f"{prefix}_char_similarity": _ratio(a.unicode_preserving, b.unicode_preserving),
        f"{prefix}_token_jaccard": _jaccard(at, bt),
        f"{prefix}_token_containment_s1": _containment(at, bt),
        f"{prefix}_token_containment_target": _containment(bt, at),
        f"{prefix}_token_count_diff": min(abs(len(a.tokens) - len(b.tokens)), 32767),
        f"{prefix}_char_length_diff": min(abs(len(a.raw) - len(b.raw)), 32767),
        f"{prefix}_token_count_ratio": _length_ratio(len(a.tokens), len(b.tokens)),
        f"{prefix}_char_length_ratio": _length_ratio(len(a.raw), len(b.raw)),
    }
    if prefix == "name":
        ai, bi = "".join(t[0] for t in a.tokens), "".join(t[0] for t in b.tokens)
        values.update(
            {
                "name_casefold_exact": int(usable and av.unicode_lower == bv.unicode_lower),
                "name_punctuation_agreement": int(
                    usable and av.punctuation_normalized == bv.punctuation_normalized
                ),
                "name_legal_suffix_agreement": int(
                    usable
                    and bool(av.legal_suffix_stripped)
                    and av.legal_suffix_stripped == bv.legal_suffix_stripped
                ),
                "name_accent_char_similarity": _ratio(a.accent_folded, b.accent_folded),
                "name_token_set_similarity": fuzz.token_set_ratio(a.accent_folded, b.accent_folded)
                / 100
                if usable
                else 0.0,
                "name_token_sort_similarity": fuzz.token_sort_ratio(
                    a.accent_folded, b.accent_folded
                )
                / 100
                if usable
                else 0.0,
                "name_prefix_similarity": _ratio(a.accent_folded[:8], b.accent_folded[:8]),
                "name_suffix_similarity": _ratio(a.accent_folded[-8:], b.accent_folded[-8:]),
                "name_initials_agreement": int(usable and bool(ai) and ai == bi),
                "name_acronym_agreement": int(
                    usable
                    and bool(ai)
                    and (ai == b.compact_alphanumeric or bi == a.compact_alphanumeric)
                ),
                "name_digit_agreement": int(
                    usable and bool(a.digit_tokens) and a.digit_tokens == b.digit_tokens
                ),
                "name_char_3gram_jaccard": _jaccard(
                    _ngrams(a.accent_folded), _ngrams(b.accent_folded)
                ),
            }
        )
    else:
        digits_a, digits_b = a.digit_tokens, b.digit_tokens
        postal_a, postal_b = _terminal_postal(left), _terminal_postal(right)
        values.update(
            {
                "address_number_exact": int(usable and bool(digits_a) and digits_a == digits_b),
                "address_number_overlap": _jaccard(set(digits_a), set(digits_b)),
                "address_house_number_agreement": int(
                    usable and bool(digits_a) and bool(digits_b) and digits_a[0] == digits_b[0]
                ),
                "address_postal_agreement": int(
                    usable and bool(postal_a) and bool(postal_b) and postal_a == postal_b
                ),
            }
        )
    return values


def _retrieval_features(row: object, has_json: bool, has_fusion: bool) -> dict[str, object]:
    support = _support(row, has_json)
    scores = [float(item["score"]) for item in support]
    ranks = [int(item["rank"]) for item in support]
    channels = [str(item["channel"]) for item in support]
    families = [channel.split(":", 1)[-1].lower() for channel in channels]
    independent = {
        field for family in families for field in ("source", "name", "address") if field in family
    }
    exact_count = sum(bool(item["exact"]) for item in support)
    strongest = min(
        support,
        key=lambda item: (
            -bool(item["exact"]),
            -float(item["score"]),
            int(item["rank"]),
            str(item["channel"]),
        ),
    )
    counts = {
        "source": sum("source" in c for c in families),
        "name": sum("name" in c for c in families),
        "address": sum("address" in c for c in families),
        "exact": sum("exact" in c for c in families),
        "fuzzy": sum("exact" not in c for c in families),
    }
    return {
        "retrieval_score": float(row.retrieval_score),
        "fusion_score": float(row.fusion_score) if has_fusion else float(row.retrieval_score),
        "channel_rank": int(row.channel_rank),
        "reciprocal_rank": 1.0 / int(row.channel_rank),
        "exact_match": int(row.exact_match),
        "channel_support_count": len(support),
        "exact_support_count": exact_count,
        "support_best_score": max(scores),
        "support_mean_score": sum(scores) / len(scores),
        "support_min_score": min(scores),
        "support_max_score": max(scores),
        "support_min_rank": min(ranks),
        "support_mean_rank": sum(ranks) / len(ranks),
        "source_support_count": counts["source"],
        "s2_support_count": len(support) if row.target_source == "S2" else 0,
        "s3_support_count": len(support) if row.target_source == "S3" else 0,
        "name_support_count": counts["name"],
        "address_support_count": counts["address"],
        "exact_support_family_count": counts["exact"],
        "fuzzy_support_count": counts["fuzzy"],
        "has_source_support": int(counts["source"] > 0),
        "has_s2_support": int(row.target_source == "S2"),
        "has_s3_support": int(row.target_source == "S3"),
        "has_name_support": int(counts["name"] > 0),
        "has_address_support": int(counts["address"] > 0),
        "has_exact_support": int(exact_count > 0),
        "has_fuzzy_support": int(counts["fuzzy"] > 0),
        "independent_channel_agreement": int(len(independent) > 1),
        "strongest_channel": str(strongest["channel"]),
    }


def compute_pair_features(
    candidates: pd.DataFrame, records: Mapping[str, SourceRecord]
) -> pd.DataFrame:
    """Transform one bounded candidate batch, preserving its input row order."""
    required = {
        "s1_id",
        "target_id",
        "target_source",
        "channel",
        "retrieval_score",
        "channel_rank",
        "exact_match",
        "run_id",
    }
    missing = required - set(candidates)
    if missing:
        raise ValueError(f"candidate columns missing: {sorted(missing)}")
    if candidates.empty:
        return OUTPUT_SCHEMA.empty_table().to_pandas()
    if candidates[["s1_id", "target_id", "target_source"]].duplicated().any():
        raise ValueError("duplicate candidate pair")
    has_json, has_fusion = "support_json" in candidates, "fusion_score" in candidates
    rows: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        if (
            not isinstance(row.s1_id, str)
            or not row.s1_id.startswith("S1-")
            or row.target_source not in {"S2", "S3"}
            or not isinstance(row.target_id, str)
            or not row.target_id.startswith(row.target_source + "-")
            or not isinstance(row.channel, str)
            or not row.channel
            or not isinstance(row.run_id, str)
            or not row.run_id
            or not isinstance(row.exact_match, (bool, np.bool_))
            or isinstance(row.channel_rank, (bool, np.bool_))
            or int(row.channel_rank) != row.channel_rank
            or row.channel_rank < 1
            or row.channel_rank > 32767
            or not math.isfinite(float(row.retrieval_score))
            or (has_fusion and not math.isfinite(float(row.fusion_score)))
        ):
            raise ValueError("invalid candidate identifiers, ownership, or retrieval values")
        query, target = records.get(row.s1_id), records.get(row.target_id)
        if query is None or target is None:
            raise ValueError(
                f"candidate refers to absent source record: {row.s1_id}/{row.target_id}"
            )
        if query.source != "S1" or target.source != row.target_source:
            raise ValueError("target source ownership mismatch")
        name = _field_features(query.name, target.name, "name")
        address = _field_features(query.address, target.address, "address")
        retrieval = _retrieval_features(row, has_json, has_fusion)
        name_usable = not (
            name["name_s1_missing"]
            or name["name_target_missing"]
            or name["name_s1_null_like"]
            or name["name_target_null_like"]
        )
        address_usable = not (
            address["address_s1_missing"]
            or address["address_target_missing"]
            or address["address_s1_null_like"]
            or address["address_target_null_like"]
        )
        ns, ads = float(name["name_char_similarity"]), float(address["address_char_similarity"])
        digits_a, digits_b = (
            _normalize(query.address).digit_tokens,
            _normalize(target.address).digit_tokens,
        )
        country_a, country_b = _normalize(query.country), _normalize(target.country)
        cross = {
            "both_names_usable": int(name_usable),
            "both_addresses_usable": int(address_usable),
            "exact_name_and_address": int(
                name["name_normalized_exact"] and address["address_normalized_exact"]
            ),
            "strong_name_weak_address": int(ns >= 0.85 and (not address_usable or ads < 0.5)),
            "strong_address_weak_name": int(ads >= 0.85 and (not name_usable or ns < 0.5)),
            "exact_name_conflicting_address_digits": int(
                name["name_normalized_exact"]
                and bool(digits_a)
                and bool(digits_b)
                and not (set(digits_a) & set(digits_b))
            ),
            "combined_string_similarity": (ns + ads) / 2
            if name_usable and address_usable
            else max(ns, ads),
            "retrieval_name_interaction": float(row.retrieval_score) * ns,
            "retrieval_address_interaction": float(row.retrieval_score) * ads,
            "country_both_present": int(
                not country_a.is_missing
                and not country_a.is_null_like
                and not country_b.is_missing
                and not country_b.is_null_like
            ),
            "country_agreement": int(
                bool(country_a.unicode_preserving)
                and country_a.unicode_preserving == country_b.unicode_preserving
            ),
        }
        rows.append(
            {
                "s1_id": row.s1_id,
                "target_id": row.target_id,
                "target_source": row.target_source,
                "run_id": row.run_id,
                **retrieval,
                **name,
                **address,
                **cross,
            }
        )
    output = pd.DataFrame.from_records(rows, columns=OUTPUT_SCHEMA.names)
    for field in OUTPUT_SCHEMA:
        if field.name in FEATURE_DTYPES and FEATURE_DTYPES[field.name] != "string":
            output[field.name] = output[field.name].astype(FEATURE_DTYPES[field.name])
    return output
