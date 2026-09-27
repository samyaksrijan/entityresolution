"""Persisted, resumable sparse character TF-IDF top-N retrieval."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn, zip_sp_matmul_topn

from entity_resolution.candidate_schema import CandidateEvidence, evidence_frame
from entity_resolution.exact_index import content_tie_key
from entity_resolution.views import normalize_field


@dataclass(frozen=True)
class SparseRetrievalConfig:
    field: str
    view: str = "unicode_preserving"
    ngram_min: int = 3
    ngram_max: int = 5
    min_df: int = 2
    max_features: int = 50_000
    threshold: float = 0.20
    top_k: int = 80
    country_partition: bool = True
    n_threads: int = 4

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


class BatchManifest:
    """Small atomic resume manifest for a channel run."""

    def __init__(self, path: Path, fingerprint: str) -> None:
        self.path = path
        self.fingerprint = fingerprint

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"fingerprint": self.fingerprint, "completed_batches": 0, "complete": False}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != self.fingerprint:
            raise ValueError("resume manifest configuration fingerprint mismatch")
        return payload

    def save(self, *, completed_batches: int, complete: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fingerprint": self.fingerprint,
            "completed_batches": completed_batches,
            "complete": complete,
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)


def normalized_text(value: object, view: str) -> str:
    normalized = normalize_field(value)
    if normalized.is_missing or normalized.is_null_like:
        return ""
    if view == "unicode_preserving":
        return normalized.unicode_preserving
    if view == "accent_folded":
        return normalized.accent_folded
    if view == "compact_alphanumeric":
        return normalized.compact_alphanumeric
    raise ValueError(f"unsupported normalization view: {view!r}")


def fit_vectorizer(texts: list[str], config: SparseRetrievalConfig) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(config.ngram_min, config.ngram_max),
        min_df=config.min_df,
        max_features=config.max_features,
        sublinear_tf=True,
        dtype=np.float32,
        lowercase=False,
        norm="l2",
    )
    vectorizer.fit(texts)
    return vectorizer


def _merge_topk(
    current: list[tuple[float, str, str]],
    additions: list[tuple[float, str, str]],
    top_k: int,
) -> list[tuple[float, str, str]]:
    """Merge by score then text-only tie key; target IDs are never sort signals."""
    best_by_target: dict[str, tuple[float, str, str]] = {item[2]: item for item in current}
    for item in additions:
        previous = best_by_target.get(item[2])
        if previous is None or item[0] > previous[0]:
            best_by_target[item[2]] = item
    ordered = sorted(best_by_target.values(), key=lambda item: (-item[0], item[1]))
    return ordered[:top_k]


class SparseTopNRetriever:
    """Fit once and retrieve batches with heap-bounded sparse multiplication."""

    def __init__(self, config: SparseRetrievalConfig, vectorizer: TfidfVectorizer) -> None:
        self.config = config
        self.vectorizer = vectorizer

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"config": self.config, "vectorizer": self.vectorizer}, path)

    @classmethod
    def load(cls, path: Path) -> SparseTopNRetriever:
        payload = joblib.load(path)
        return cls(payload["config"], payload["vectorizer"])

    def retrieve(
        self,
        queries: pd.DataFrame,
        target_batch_factory: Callable[[], Iterable[pd.DataFrame]],
        *,
        source: str,
        run_id: str,
        artifact_path: Path | None = None,
        checkpoint_every: int = 25,
        query_batch_size: int = 1_000,
    ) -> tuple[pd.DataFrame, dict[str, float | int]]:
        if query_batch_size < 1:
            raise ValueError("query_batch_size must be positive")
        field_column = "business_name" if self.config.field == "name" else "business_address"
        query_texts = [normalized_text(value, self.config.view) for value in queries[field_column]]
        country_indices = {
            country: np.flatnonzero(queries["country"].to_numpy() == country)
            for country in sorted(queries["country"].unique())
        }
        query_batches = {
            country: [
                (
                    start,
                    self.vectorizer.transform(
                        [query_texts[index] for index in indices[start : start + query_batch_size]]
                    ).astype(np.float32),
                )
                for start in range(0, len(indices), query_batch_size)
            ]
            for country, indices in country_indices.items()
        }
        best_matrices: dict[str, list[sparse.csr_matrix]] = {
            country: [
                sparse.csr_matrix((matrix.shape[0], 0), dtype=np.float32)
                for _start, matrix in batches
            ]
            for country, batches in query_batches.items()
        }
        manifest: BatchManifest | None = None
        checkpoint_path: Path | None = None
        completed_batches = 0
        if artifact_path is not None:
            manifest = BatchManifest(
                artifact_path.with_suffix(".manifest.json"), self.config.fingerprint()
            )
            state = manifest.load()
            if state["complete"] and artifact_path.exists():
                return pd.read_parquet(artifact_path), {
                    "runtime_seconds": 0.0,
                    "peak_rss_mb": psutil.Process().memory_info().rss / 2**20,
                    "resumed_complete_artifact": 1,
                }
            completed_batches = int(state["completed_batches"])
            checkpoint_path = artifact_path.with_suffix(".checkpoint.joblib")
            if completed_batches and checkpoint_path.exists():
                best_matrices = joblib.load(checkpoint_path)

        started = time.perf_counter()
        peak_mb = psutil.Process().memory_info().rss / 2**20
        batches_seen = 0
        for batch_number, target_batch in enumerate(target_batch_factory()):
            if batch_number < completed_batches:
                continue
            batches_seen = batch_number + 1
            target_texts = [
                normalized_text(value, self.config.view) for value in target_batch[field_column]
            ]
            for country, query_positions in country_indices.items():
                if self.config.country_partition:
                    target_positions = np.flatnonzero(
                        target_batch["country"].to_numpy() == country
                    )
                else:
                    target_positions = np.arange(len(target_batch))
                if not len(target_positions) or not len(query_positions):
                    continue
                target_matrix = self.vectorizer.transform(
                    [target_texts[index] for index in target_positions]
                ).astype(np.float32)
                for query_batch_number, (_start, query_matrix) in enumerate(
                    query_batches[country]
                ):
                    similarities = sp_matmul_topn(
                        query_matrix,
                        target_matrix.T,
                        top_n=self.config.top_k,
                        threshold=self.config.threshold,
                        sort=True,
                        n_threads=self.config.n_threads,
                    ).tocsr()
                    best_matrices[country][query_batch_number] = zip_sp_matmul_topn(
                        self.config.top_k,
                        [best_matrices[country][query_batch_number], similarities],
                    ).tocsr()
            peak_mb = max(peak_mb, psutil.Process().memory_info().rss / 2**20)
            if manifest is not None and checkpoint_path is not None and (
                batches_seen % checkpoint_every == 0
            ):
                joblib.dump(best_matrices, checkpoint_path)
                manifest.save(completed_batches=batches_seen)

        selected_indices = {
            country: {int(index) for matrix in matrices for index in matrix.indices}
            for country, matrices in best_matrices.items()
        }
        target_lookup: dict[str, dict[int, tuple[str, str]]] = {
            country: {} for country in country_indices
        }
        country_offsets: Counter[str] = Counter()
        for target_batch in target_batch_factory():
            for entity_id, name, address, country in target_batch[
                ["entity_id", "business_name", "business_address", "country"]
            ].itertuples(index=False, name=None):
                local_index = country_offsets[country]
                if local_index in selected_indices.get(country, set()):
                    target_lookup[country][local_index] = (
                        entity_id,
                        content_tie_key(name, address, country),
                    )
                country_offsets[country] += 1

        rows: list[CandidateEvidence] = []
        channel = f"{self.config.field}_char_tfidf"
        for country, query_positions in country_indices.items():
            for (batch_start, _query_matrix), matrix in zip(
                query_batches[country], best_matrices[country], strict=True
            ):
                for local_query in range(matrix.shape[0]):
                    start, end = matrix.indptr[local_query : local_query + 2]
                    matches = []
                    for score, target_index in zip(
                        matrix.data[start:end], matrix.indices[start:end], strict=True
                    ):
                        target_id, tie_key = target_lookup[country][int(target_index)]
                        matches.append((float(score), tie_key, target_id))
                    matches.sort(key=lambda item: (-item[0], item[1]))
                    global_query = query_positions[batch_start + local_query]
                    s1_id = str(queries.iloc[int(global_query)]["entity_id"])
                    for rank, (score, _tie, target_id) in enumerate(matches, start=1):
                        rows.append(
                            CandidateEvidence(
                                s1_id=s1_id,
                                target_id=target_id,
                                target_source=source,
                                channel=channel,
                                field=self.config.field,
                                normalization_view=self.config.view,
                                retrieval_score=score,
                                channel_rank=rank,
                                exact_match=False,
                                run_id=run_id,
                            )
                        )
        result = evidence_frame(rows)
        if artifact_path is not None:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_parquet(artifact_path, index=False)
            if checkpoint_path is not None:
                checkpoint_path.unlink(missing_ok=True)
            if manifest is not None:
                manifest.save(completed_batches=batches_seen, complete=True)
        return result, {
            "runtime_seconds": time.perf_counter() - started,
            "peak_rss_mb": peak_mb,
            "target_batches": batches_seen,
            "candidate_rows": len(result),
        }
