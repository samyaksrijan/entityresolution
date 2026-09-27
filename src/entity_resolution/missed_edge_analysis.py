"""Structured, content-based audit of positive edges omitted by candidate fusion."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from entity_resolution.config import load_data_paths
from entity_resolution.exact_index import content_tie_key
from entity_resolution.forensic_analysis import _read_chunks
from entity_resolution.views import comparison_views, normalize_field


def _load_truth_target_content(
    truth: dict[str, set[str]], data_paths_config: Path
) -> dict[str, dict[str, Any]]:
    paths = load_data_paths(data_paths_config)
    wanted_by_source = {
        source: {
            target
            for targets in truth.values()
            for target in targets
            if target.startswith(f"{source}-")
        }
        for source in ("S2", "S3")
    }
    result: dict[str, dict[str, Any]] = {}
    for source in ("S2", "S3"):
        wanted = wanted_by_source[source]
        for chunk in _read_chunks(paths[f"train_source{source[-1]}"]):
            subset = chunk[chunk["entity_id"].isin(wanted)]
            for entity_id, name, address, country in subset.itertuples(index=False, name=None):
                normalized_name = normalize_field(name)
                normalized_address = normalize_field(address)
                result[str(entity_id)] = {
                    "name": name,
                    "address": address,
                    "country": str(country),
                    "source": source,
                    "name_normalized": normalized_name.unicode_preserving,
                    "address_normalized": normalized_address.unicode_preserving,
                    "name_missing": normalized_name.is_missing,
                    "address_missing": normalized_address.is_missing,
                    "signature": (
                        source,
                        str(country),
                        normalized_name.unicode_preserving,
                        normalized_address.unicode_preserving,
                    ),
                    "content_key": content_tie_key(name, address, country),
                }
    expected = set().union(*wanted_by_source.values())
    missing = expected - set(result)
    if missing:
        raise ValueError(f"{len(missing)} truth targets were not found in configured source data")
    return result


def _signature_counts(
    signatures: set[tuple[str, str, str, str]], data_paths_config: Path
) -> Counter[tuple[str, str, str, str]]:
    paths = load_data_paths(data_paths_config)
    counts: Counter[tuple[str, str, str, str]] = Counter()
    signatures_by_source = {
        source: {signature for signature in signatures if signature[0] == source}
        for source in ("S2", "S3")
    }
    for source in ("S2", "S3"):
        wanted = signatures_by_source[source]
        if not wanted:
            continue
        for chunk in _read_chunks(paths[f"train_source{source[-1]}"]):
            for _entity_id, name, address, country in chunk.itertuples(index=False, name=None):
                signature = (
                    source,
                    str(country),
                    normalize_field(name).unicode_preserving,
                    normalize_field(address).unicode_preserving,
                )
                if signature in wanted:
                    counts[signature] += 1
    return counts


def _query_content(queries: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        str(entity_id): {"name": name, "address": address, "country": str(country)}
        for entity_id, name, address, country in queries[
            ["entity_id", "business_name", "business_address", "country"]
        ].itertuples(index=False, name=None)
    }


def _primary_category(row: dict[str, Any]) -> str:
    if row["absent_from_every_existing_top80_channel"]:
        for flag, label in (
            ("target_address_missing", "absent_top80_target_address_missing"),
            ("weak_or_missing_name_evidence", "absent_top80_weak_name"),
            ("accent_unicode_normalization_issue", "absent_top80_accent_unicode"),
            ("legal_suffix_or_token_order_variation", "absent_top80_name_variation"),
            ("duplicate_common_normalized_signature", "absent_top80_common_signature"),
        ):
            if row[flag]:
                return label
        return "absent_top80_other"
    if row["multi_match_truncation"]:
        return "multi_match_truncation"
    if row["name_retrieved_below_selected_budget"]:
        return "name_evidence_fusion_truncation"
    if row["address_retrieved_below_selected_budget"]:
        return "address_evidence_fusion_truncation"
    return "other_fusion_truncation"


def audit_missed_edges(
    *,
    selected: pd.DataFrame,
    all_evidence: pd.DataFrame,
    truth: dict[str, set[str]],
    sample: pd.DataFrame,
    queries: pd.DataFrame,
    data_paths_config: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Classify every missed truth edge; category flags intentionally allow overlaps."""
    selected_pairs = set(
        selected[["s1_id", "target_id"]].itertuples(index=False, name=None)
    )
    evidence_groups = {
        (str(s1_id), str(target_id)): group
        for (s1_id, target_id), group in all_evidence.groupby(
            ["s1_id", "target_id"], sort=False, observed=True
        )
    }
    target_content = _load_truth_target_content(truth, data_paths_config)
    query_content = _query_content(queries)
    country_by_s1 = dict(sample[["s1_id", "country"]].itertuples(index=False, name=None))
    rows: list[dict[str, Any]] = []
    for s1_id, targets in truth.items():
        recovered = {target for target in targets if (s1_id, target) in selected_pairs}
        for target_id in sorted(targets):
            if (s1_id, target_id) in selected_pairs:
                continue
            evidence = evidence_groups.get((s1_id, target_id))
            channels = set() if evidence is None else set(evidence["channel"].astype(str))
            target = target_content[target_id]
            query = query_content[s1_id]
            query_name = comparison_views(query["name"], is_name=True)
            target_name = comparison_views(target["name"], is_name=True)
            name_present = bool(
                channels
                & {
                    "name_char_tfidf",
                    "exact_name_unicode_preserving",
                    "exact_name_accent_folded",
                }
            )
            address_present = bool(
                channels
                & {
                    "address_char_tfidf",
                    "exact_address_unicode_preserving",
                    "exact_address_accent_folded",
                }
            )
            row: dict[str, Any] = {
                "s1_id": s1_id,
                "target_id": target_id,
                "target_source": target["source"],
                "country": str(country_by_s1[s1_id]),
                "truth_cardinality": len(targets),
                "retrieved_channels": "|".join(sorted(channels)),
                "target_address_missing": bool(target["address_missing"]),
                "weak_or_missing_name_evidence": bool(
                    normalize_field(query["name"]).is_missing
                    or target["name_missing"]
                    or not name_present
                ),
                "name_retrieved_below_selected_budget": bool(name_present),
                "address_retrieved_below_selected_budget": bool(address_present),
                "absent_from_every_existing_top80_channel": not channels,
                "duplicate_common_normalized_signature": False,
                "accent_unicode_normalization_issue": bool(
                    query_name.punctuation_normalized
                    and query_name.punctuation_normalized != target_name.punctuation_normalized
                    and query_name.accent_folded == target_name.accent_folded
                ),
                "legal_suffix_or_token_order_variation": bool(
                    query_name.punctuation_normalized
                    and query_name.punctuation_normalized != target_name.punctuation_normalized
                    and (
                        query_name.legal_suffix_stripped == target_name.legal_suffix_stripped
                        or query_name.sorted_token_sequence == target_name.sorted_token_sequence
                    )
                ),
                "india_specific_concentration": str(country_by_s1[s1_id]) == "India",
                "s3_specific_concentration": target["source"] == "S3",
                "multi_match_truncation": len(targets) > 1 and bool(recovered),
                "query_name_token_count": len(query_name.word_token_sequence),
                "target_name_token_count": len(target_name.word_token_sequence),
                "query_address_missing": normalize_field(query["address"]).is_missing,
                "redacted_content_key": hashlib_key(
                    query["name"], query["address"], target["name"], target["address"]
                ),
                "_signature": target["signature"],
            }
            if evidence is not None:
                by_channel = evidence.set_index("channel")["channel_rank"].to_dict()
                row["name_char_rank"] = by_channel.get("name_char_tfidf")
                row["address_char_rank"] = by_channel.get("address_char_tfidf")
            else:
                row["name_char_rank"] = None
                row["address_char_rank"] = None
            rows.append(row)
    if not rows:
        columns = [
            "s1_id",
            "target_id",
            "target_source",
            "country",
            "primary_category",
        ]
        return pd.DataFrame(columns=columns), {"missed_edges": 0, "categories": {}}

    signatures = {row["_signature"] for row in rows}
    signature_counts = _signature_counts(signatures, data_paths_config)
    for row in rows:
        row["normalized_signature_frequency"] = signature_counts[row["_signature"]]
        row["duplicate_common_normalized_signature"] = (
            row["normalized_signature_frequency"] > 1
        )
        row["primary_category"] = _primary_category(row)
        del row["_signature"]
    frame = pd.DataFrame(rows)
    category_columns = [
        "target_address_missing",
        "weak_or_missing_name_evidence",
        "name_retrieved_below_selected_budget",
        "address_retrieved_below_selected_budget",
        "absent_from_every_existing_top80_channel",
        "duplicate_common_normalized_signature",
        "accent_unicode_normalization_issue",
        "legal_suffix_or_token_order_variation",
        "india_specific_concentration",
        "s3_specific_concentration",
        "multi_match_truncation",
    ]
    counts = {column: int(frame[column].sum()) for column in category_columns}
    overlaps: dict[str, int] = {}
    for index, left in enumerate(category_columns):
        for right in category_columns[index + 1 :]:
            count = int((frame[left] & frame[right]).sum())
            if count:
                overlaps[f"{left}&{right}"] = count
    examples = []
    for category, group in frame.groupby("primary_category", sort=True):
        row = group.sort_values("redacted_content_key", kind="stable").iloc[0]
        examples.append(
            {
                "primary_category": category,
                "redacted_content_key": row["redacted_content_key"],
                "country": row["country"],
                "target_source": row["target_source"],
                "query_name_token_count": int(row["query_name_token_count"]),
                "target_name_token_count": int(row["target_name_token_count"]),
                "target_address_missing": bool(row["target_address_missing"]),
                "retrieved_channels": row["retrieved_channels"],
            }
        )
    summary = {
        "missed_edges": len(frame),
        "category_counts": counts,
        "category_percentages": {
            key: value / len(frame) for key, value in counts.items()
        },
        "primary_category_counts": frame["primary_category"].value_counts().to_dict(),
        "overlap_counts": overlaps,
        "representative_redacted_content_examples": examples,
    }
    return frame, summary


def hashlib_key(*values: object) -> str:
    """Short content-only key for report examples; entity IDs are never hashed here."""
    import hashlib

    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


TAXONOMY = (
    "missing_or_null_name",
    "missing_address",
    "both_fields_sparse",
    "accent_unicode",
    "legal_suffix",
    "token_reorder",
    "token_segmentation",
    "abbreviation_initials",
    "typographical_corruption",
    "digit_address_conflict",
    "cross_language_like",
    "fusion_truncation",
    "absent_every_channel",
    "other_unknown",
)


def classify_variations(
    query: dict[str, Any], target: dict[str, Any], *, present: bool
) -> dict[str, bool]:
    """Deterministic diagnostic heuristics, never candidate gates or training labels.

    Cross-language-like is a script mismatch flag, not a claim of translation.
    Unknown categories remain explicit rather than treating weak retrieval as a typo.
    """
    import unicodedata

    from rapidfuzz.fuzz import ratio

    qn, tn = normalize_field(query["name"]), normalize_field(target["name"])
    qa, ta = normalize_field(query["address"]), normalize_field(target["address"])
    qv, tv = (
        comparison_views(query["name"], is_name=True),
        comparison_views(target["name"], is_name=True),
    )
    different = bool(
        qn.unicode_preserving
        and tn.unicode_preserving
        and qn.unicode_preserving != tn.unicode_preserving
    )

    def initials(tokens):
        return "".join(t[0] for t in tokens if t)

    def scripts(text):
        return {unicodedata.name(c, "UNKNOWN").split()[0] for c in text if c.isalpha()}

    result = {
        "missing_or_null_name": not qn.unicode_preserving or not tn.unicode_preserving,
        "missing_address": not qa.unicode_preserving or not ta.unicode_preserving,
        "both_fields_sparse": (len(qn.unicode_preserving) < 3 and len(qa.unicode_preserving) < 5)
        or (len(tn.unicode_preserving) < 3 and len(ta.unicode_preserving) < 5),
        "accent_unicode": different and qn.accent_folded == tn.accent_folded,
        "legal_suffix": different
        and bool(qv.legal_suffix_stripped)
        and qv.legal_suffix_stripped == tv.legal_suffix_stripped,
        "token_reorder": different and sorted(qn.tokens) == sorted(tn.tokens),
        "token_segmentation": different and qn.compact_alphanumeric == tn.compact_alphanumeric,
        "abbreviation_initials": different
        and (
            (len(qn.tokens) > 1 and initials(qn.tokens) == tn.compact_alphanumeric)
            or (len(tn.tokens) > 1 and initials(tn.tokens) == qn.compact_alphanumeric)
        ),
        "typographical_corruption": different
        and 70 <= ratio(qn.accent_folded, tn.accent_folded) < 100,
        "digit_address_conflict": bool(qa.digit_tokens and ta.digit_tokens)
        and qa.digit_tokens != ta.digit_tokens,
        "cross_language_like": different
        and bool(scripts(qn.unicode_preserving))
        and scripts(qn.unicode_preserving).isdisjoint(scripts(tn.unicode_preserving)),
        "fusion_truncation": present,
        "absent_every_channel": not present,
    }
    result["other_unknown"] = not any(result[k] for k in TAXONOMY[:11])
    return {key: bool(value) for key, value in result.items()}
