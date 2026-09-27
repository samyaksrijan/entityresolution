"""Schemas and deterministic provenance aggregation for candidate pairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

ALLOWED_TARGET_PREFIXES = ("S2-", "S3-")
CANDIDATE_COLUMNS = (
    "s1_id",
    "target_id",
    "target_source",
    "channel",
    "field",
    "normalization_view",
    "retrieval_score",
    "channel_rank",
    "exact_match",
    "run_id",
)


@dataclass(frozen=True)
class CandidateEvidence:
    """One channel's evidence for one S1-to-target pair."""

    s1_id: str
    target_id: str
    target_source: str
    channel: str
    field: str
    normalization_view: str
    retrieval_score: float
    channel_rank: int
    exact_match: bool
    run_id: str

    def __post_init__(self) -> None:
        if not self.s1_id.startswith("S1-"):
            raise ValueError(f"invalid S1 ID: {self.s1_id!r}")
        if not self.target_id.startswith(ALLOWED_TARGET_PREFIXES):
            raise ValueError(f"invalid target ID: {self.target_id!r}")
        if self.target_source not in {"S2", "S3"}:
            raise ValueError(f"invalid target source: {self.target_source!r}")
        if not self.target_id.startswith(f"{self.target_source}-"):
            raise ValueError("target ID prefix and target_source disagree")
        if self.channel_rank < 1:
            raise ValueError("channel_rank must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evidence_frame(rows: list[CandidateEvidence]) -> pd.DataFrame:
    """Build a typed long-form candidate frame."""
    return pd.DataFrame([row.as_dict() for row in rows], columns=CANDIDATE_COLUMNS)


def aggregate_provenance(frame: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate candidate pairs while retaining channel scores/ranks and evidence flags."""
    missing = set(CANDIDATE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"candidate evidence is missing columns: {sorted(missing)}")
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "s1_id",
                "target_id",
                "target_source",
                "run_id",
                "channels",
                "exact_name",
                "exact_address",
            ]
        )

    keys = ["s1_id", "target_id", "target_source", "run_id"]
    working = frame.copy()
    working["exact_name"] = (working["field"] == "name") & working["exact_match"]
    working["exact_address"] = (working["field"] == "address") & working["exact_match"]
    aggregations: dict[str, str] = {"exact_name": "max", "exact_address": "max"}
    channels = sorted(working["channel"].unique())
    for channel in channels:
        safe = channel.replace("-", "_")
        selected = working["channel"] == channel
        working[f"has_{safe}"] = selected
        working[f"score_{safe}"] = working["retrieval_score"].where(selected)
        working[f"rank_{safe}"] = working["channel_rank"].where(selected)
        aggregations[f"has_{safe}"] = "max"
        aggregations[f"score_{safe}"] = "max"
        aggregations[f"rank_{safe}"] = "min"
    result = working.groupby(keys, sort=True, observed=True, as_index=False).agg(aggregations)
    channel_strings = np.full(len(result), "", dtype=object)
    for channel in channels:
        safe = channel.replace("-", "_")
        separator = np.where(channel_strings == "", "", "|")
        channel_strings = np.where(
            result[f"has_{safe}"].to_numpy(),
            channel_strings + separator + channel,
            channel_strings,
        )
    result["channels"] = channel_strings
    return result


def add_address_digit_agreement(
    frame: pd.DataFrame,
    query_digits: dict[str, tuple[str, ...]],
    target_digits: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    """Add address-derived digit evidence without using digits as a global block."""
    result = frame.copy()
    query_values = [query_digits.get(str(value), ()) for value in result["s1_id"]]
    target_values = [target_digits.get(str(value), ()) for value in result["target_id"]]
    result["query_has_address_digits"] = [bool(value) for value in query_values]
    result["target_has_address_digits"] = [bool(value) for value in target_values]
    result["address_digit_any_agreement"] = [
        bool(set(query) & set(target))
        for query, target in zip(query_values, target_values, strict=True)
    ]
    result["address_digit_exact_agreement"] = [
        bool(query) and query == target
        for query, target in zip(query_values, target_values, strict=True)
    ]
    return result
