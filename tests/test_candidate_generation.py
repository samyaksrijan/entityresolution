from pathlib import Path

import pandas as pd
import pytest

from entity_resolution.candidate_evaluation import (
    evaluate_candidates,
    oracle_macro_f05,
    per_s1_fbeta,
    union_topk,
)
from entity_resolution.candidate_schema import (
    CandidateEvidence,
    add_address_digit_agreement,
    aggregate_provenance,
    evidence_frame,
)
from entity_resolution.exact_index import ExactIndex
from entity_resolution.sparse_retrieval import (
    BatchManifest,
    SparseRetrievalConfig,
    SparseTopNRetriever,
    _merge_topk,
    fit_vectorizer,
)


def _sources() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("S2-a", "Café North", "1 Main St", "US"),
            ("S2-b", "Cafe North", "2 Main St", "US"),
            ("S2-c", "null", "", "US"),
        ],
        columns=["entity_id", "business_name", "business_address", "country"],
    )


def _queries() -> pd.DataFrame:
    return pd.DataFrame(
        [("S1-a", "CAFÉ NORTH", "1 Main Street", "US"), ("S1-b", "null", "", "US")],
        columns=["entity_id", "business_name", "business_address", "country"],
    )


def test_exact_index_separates_sources_and_handles_unicode_views(tmp_path: Path) -> None:
    unicode_index = ExactIndex.build(
        _sources(), source="S2", field="name", view="unicode_preserving"
    )
    folded_index = ExactIndex.build(
        _sources(), source="S2", field="name", view="accent_folded"
    )
    path = tmp_path / "index.parquet"
    folded_index.save(path)

    unicode_rows = unicode_index.retrieve(_queries(), top_k=5, run_id="test")
    folded_rows = ExactIndex.load(path).retrieve(_queries(), top_k=5, run_id="test")

    assert {row.target_id for row in unicode_rows} == {"S2-a"}
    assert {row.target_id for row in folded_rows} == {"S2-a", "S2-b"}
    assert all(row.target_source == "S2" for row in folded_rows)
    assert all(row.s1_id != "S1-b" for row in folded_rows)


def test_exact_index_missing_address_and_topk_deterministic() -> None:
    index = ExactIndex.build(
        _sources(), source="S2", field="address", view="unicode_preserving"
    )
    first = index.retrieve(_queries(), top_k=1, run_id="test")
    second = index.retrieve(_queries(), top_k=1, run_id="test")
    assert [row.target_id for row in first] == [row.target_id for row in second]
    assert len(first) == 0  # "street" is not an exact match to "st"


def test_candidate_prefix_validation_and_no_numeric_id_feature() -> None:
    row = CandidateEvidence(
        "S1-999", "S3-123", "S3", "name_char_tfidf", "name",
        "unicode_preserving", 0.8, 1, False, "run"
    )
    assert set(row.as_dict()) == {
        "s1_id", "target_id", "target_source", "channel", "field",
        "normalization_view", "retrieval_score", "channel_rank", "exact_match", "run_id",
    }
    with pytest.raises(ValueError, match="invalid target ID"):
        CandidateEvidence("S1-a", "S1-b", "S2", "x", "name", "x", 1.0, 1, True, "run")


def test_union_deduplication_and_provenance_aggregation() -> None:
    rows = [
        CandidateEvidence(
            "S1-a", "S2-x", "S2", "exact_name", "name",
            "unicode_preserving", 1.0, 1, True, "r"
        ),
        CandidateEvidence(
            "S1-a", "S2-x", "S2", "name_char_tfidf", "name",
            "unicode_preserving", 0.9, 2, False, "r"
        ),
    ]
    frame = evidence_frame(rows)
    union = union_topk([frame], {"exact_name", "name_char_tfidf"}, 5)
    aggregated = aggregate_provenance(frame)
    assert len(union) == 1
    assert aggregated.loc[0, "channels"] == "exact_name|name_char_tfidf"
    assert aggregated.loc[0, "exact_name"]


def test_address_digit_agreement_is_pair_evidence_only() -> None:
    pairs = pd.DataFrame(
        [("S1-a", "S2-x"), ("S1-b", "S3-y")], columns=["s1_id", "target_id"]
    )
    enriched = add_address_digit_agreement(
        pairs,
        {"S1-a": ("12", "4"), "S1-b": ()},
        {"S2-x": ("12", "5"), "S3-y": ()},
    )
    assert enriched.loc[0, "address_digit_any_agreement"]
    assert not enriched.loc[0, "address_digit_exact_agreement"]
    assert not enriched.loc[1, "address_digit_any_agreement"]
    assert not enriched.loc[1, "address_digit_exact_agreement"]


def test_recall_coverage_and_oracle_macro_f05() -> None:
    truth = {"S1-a": {"S2-x", "S3-y"}, "S1-b": set()}
    candidates = {"S1-a": {"S2-x"}, "S1-b": {"S2-z"}}
    assert oracle_macro_f05(candidates, truth) == pytest.approx((1.25 * 0.5 / 0.75 + 1) / 2)
    assert per_s1_fbeta(set(), set()) == 1.0
    frame = pd.DataFrame([{"s1_id": "S1-a", "target_id": "S2-x"}])
    metadata = pd.DataFrame(
        [
            ("S1-a", "US", "2", "both_sources", "present"),
            ("S1-b", "India", "0", "no_match", "present"),
        ],
        columns=["s1_id", "country", "cardinality_bucket", "truth_composition", "address_state"],
    )
    metrics = evaluate_candidates(frame, truth, metadata)
    assert metrics["positive_edge_recall"] == 0.5
    assert metrics["all_truth_recovered_rate"] == 0.5
    assert metrics["at_least_one_missed_truth_rate"] == 0.5


def test_text_tie_merge_and_resume_manifest(tmp_path: Path) -> None:
    merged = _merge_topk(
        [(0.8, "b", "S2-x")],
        [(0.8, "a", "S2-y"), (0.7, "c", "S2-z")],
        2,
    )
    assert [item[2] for item in merged] == ["S2-y", "S2-x"]
    path = tmp_path / "resume.json"
    manifest = BatchManifest(path, "abc")
    manifest.save(completed_batches=3)
    assert manifest.load()["completed_batches"] == 3
    with pytest.raises(ValueError, match="fingerprint"):
        BatchManifest(path, "different").load()


def test_sparse_retrieval_batches_queries_enforces_topk_and_resumes(tmp_path: Path) -> None:
    targets = _sources().iloc[:2].copy()
    queries = _queries().iloc[:1].copy()
    config = SparseRetrievalConfig(
        field="name", ngram_min=2, ngram_max=3, min_df=1, max_features=100, top_k=1,
        threshold=0.0,
    )
    vectorizer = fit_vectorizer(targets["business_name"].tolist(), config)
    retriever = SparseTopNRetriever(config, vectorizer)
    output = tmp_path / "candidates.parquet"

    result, first_measurement = retriever.retrieve(
        queries,
        lambda: [targets.iloc[:1], targets.iloc[1:]],
        source="S2",
        run_id="test",
        artifact_path=output,
        checkpoint_every=1,
        query_batch_size=1,
    )
    assert len(result) == 1
    assert result["channel_rank"].max() == 1
    assert first_measurement["target_batches"] == 2

    def forbidden_factory() -> list[pd.DataFrame]:
        raise AssertionError("completed artifact should resume without scanning targets")

    resumed, resumed_measurement = retriever.retrieve(
        queries,
        forbidden_factory,
        source="S2",
        run_id="test",
        artifact_path=output,
        query_batch_size=1,
    )
    pd.testing.assert_frame_equal(result, resumed)
    assert resumed_measurement["resumed_complete_artifact"] == 1
