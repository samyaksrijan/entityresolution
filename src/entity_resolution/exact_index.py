"""Persisted exact-key indexes using the shared normalization views."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from entity_resolution.candidate_schema import CandidateEvidence
from entity_resolution.views import normalize_field


def normalized_key(value: object, view: str) -> tuple[str, bool, bool]:
    """Return a blocking key plus explicit missing and null-like flags."""
    normalized = normalize_field(value)
    keys = {
        "unicode_preserving": normalized.unicode_preserving,
        "accent_folded": normalized.accent_folded,
        "compact_alphanumeric": normalized.compact_alphanumeric,
    }
    if view not in keys:
        raise ValueError(f"unsupported normalization view: {view!r}")
    return keys[view], normalized.is_missing, normalized.is_null_like


def content_tie_key(name: str, address: str, country: str) -> str:
    """Stable text-derived tie key that contains no entity-ID information."""
    payload = "\x1f".join((name, address, country)).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=12).hexdigest()


@dataclass
class ExactIndex:
    """In-memory exact lookup with a portable Parquet representation."""

    source: str
    field: str
    view: str
    country_partition: bool
    table: pd.DataFrame

    @classmethod
    def build(
        cls,
        targets: pd.DataFrame,
        *,
        source: str,
        field: str,
        view: str,
        country_partition: bool = True,
    ) -> ExactIndex:
        if source not in {"S2", "S3"}:
            raise ValueError("source must be S2 or S3")
        column = "business_name" if field == "name" else "business_address"
        if field not in {"name", "address"}:
            raise ValueError("field must be name or address")
        rows: list[dict[str, object]] = []
        for entity_id, name, address, country in targets[
            ["entity_id", "business_name", "business_address", "country"]
        ].itertuples(index=False, name=None):
            if not entity_id.startswith(f"{source}-"):
                raise ValueError(f"target {entity_id!r} does not belong to {source}")
            value = name if column == "business_name" else address
            key, missing, null_like = normalized_key(value, view)
            if missing or null_like or not key:
                continue
            rows.append(
                {
                    "country": country if country_partition else "*",
                    "normalized_key": key,
                    "target_id": entity_id,
                    "tie_key": content_tie_key(name, address, country),
                }
            )
        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.sort_values(
                ["country", "normalized_key", "tie_key"], kind="stable"
            ).reset_index(drop=True)
        return cls(source, field, view, country_partition, table)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.table.to_parquet(path, index=False)
        metadata = {
            "source": self.source,
            "field": self.field,
            "view": self.view,
            "country_partition": self.country_partition,
            "rows": len(self.table),
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ExactIndex:
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(
            source=metadata["source"],
            field=metadata["field"],
            view=metadata["view"],
            country_partition=metadata["country_partition"],
            table=pd.read_parquet(path),
        )

    def retrieve(
        self,
        queries: pd.DataFrame,
        *,
        top_k: int,
        run_id: str,
    ) -> list[CandidateEvidence]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        lookup: dict[tuple[str, str], list[str]] = defaultdict(list)
        for country, key, target_id in self.table[
            ["country", "normalized_key", "target_id"]
        ].itertuples(index=False, name=None):
            lookup[(country, key)].append(target_id)
        value_column = "business_name" if self.field == "name" else "business_address"
        channel = f"exact_{self.field}_{self.view}"
        output: list[CandidateEvidence] = []
        for s1_id, value, country in queries[
            ["entity_id", value_column, "country"]
        ].itertuples(index=False, name=None):
            key, missing, null_like = normalized_key(value, self.view)
            if missing or null_like or not key:
                continue
            partition = country if self.country_partition else "*"
            for rank, target_id in enumerate(lookup.get((partition, key), ())[:top_k], start=1):
                output.append(
                    CandidateEvidence(
                        s1_id=s1_id,
                        target_id=target_id,
                        target_source=self.source,
                        channel=channel,
                        field=self.field,
                        normalization_view=self.view,
                        retrieval_score=1.0,
                        channel_rank=rank,
                        exact_match=True,
                        run_id=run_id,
                    )
                )
        return output


def query_key_maps(
    queries: pd.DataFrame,
    *,
    field: str,
    view: str,
    country_partition: bool,
) -> dict[tuple[str, str], list[str]]:
    """Build the small query-side map used by full-target streaming exact retrieval."""
    column = "business_name" if field == "name" else "business_address"
    result: dict[tuple[str, str], list[str]] = defaultdict(list)
    for entity_id, value, country in queries[["entity_id", column, "country"]].itertuples(
        index=False, name=None
    ):
        key, missing, null_like = normalized_key(value, view)
        if key and not missing and not null_like:
            result[(country if country_partition else "*", key)].append(entity_id)
    return result
