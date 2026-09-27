import json

import pandas as pd
import pytest
from test_production_candidates import fixture_run, generate  # noqa: F401

from entity_resolution import production_candidates as pc
from entity_resolution.candidate_fusion import query_quality
from entity_resolution.missed_edge_analysis import TAXONOMY, classify_variations
from entity_resolution.retrieval_rescue import (
    _metrics,
    rescue_queries,
    retrieve_missing_address,
    taxonomy,
)


def test_missing_address_view_reuses_index_and_disabled_is_noop(fixture_run):  # noqa: F811
    generate(fixture_run)
    config, paths, output = fixture_run
    retriever = pc.CachedRetriever(output / "index", config)
    query = pd.DataFrame([("S1-a", "Cafe Alpha", "", "US")], columns=pc.SOURCE_COLUMNS)
    base = retriever.retrieve(query, "r")
    disabled, stats = retrieve_missing_address(retriever, query, base, "r", {"enabled": False})
    assert disabled.empty and stats["activated_queries"] == 0
    extra, stats = retrieve_missing_address(
        retriever, query, base, "r", {"enabled": True, "top_k": 1, "threshold": 0.2}
    )
    assert stats["activated_queries"] == 1
    assert set(extra.target_id) == {"S3-a"}  # same-name S2 has an address and is excluded
    assert set(extra.channel) == {"missing_address_name_char"}
    assert not extra.exact_match.any()
    assert extra.channel_rank.max() == 1
    query["business_name"] = "null"
    empty, stats = retrieve_missing_address(
        retriever, query, base, "r", {"enabled": True, "top_k": 1, "threshold": 0.2}
    )
    assert empty.empty and stats["activated_queries"] == 0


def test_gate_strong_weak_and_empty_evidence():
    columns = pc.EVIDENCE_COLUMNS
    evidence = pd.DataFrame(
        [
            ("S1-a", "S2-a", "S2", "name_char_tfidf", 0.9, 1, False, "r", "a"),
            ("S1-a", "S2-b", "S2", "name_char_tfidf", 0.5, 2, False, "r", "b"),
            ("S1-a", "S2-a", "S2", "address_char_tfidf", 0.9, 1, False, "r", "a"),
            ("S1-a", "S2-b", "S2", "address_char_tfidf", 0.4, 2, False, "r", "b"),
            ("S1-a", "S2-a", "S2", "exact_name_unicode_preserving", 1.0, 1, True, "r", "a"),
        ],
        columns=columns,
    )
    assert not query_quality(evidence, "Alpha Company", "123 Main Street")["weak"]
    assert query_quality(evidence, "Alpha Company", "N/A")["weak"]
    assert query_quality(evidence.iloc[:0], "Alpha Company", "123 Main Street")["weak"]
    queries = pd.DataFrame(
        [
            ("S1-a", "Alpha Company", "123 Main Street", "US"),
            ("S1-b", "Beta Company", "", "US"),
            ("S1-c", "null", "123 Road", "US"),
        ],
        columns=pc.SOURCE_COLUMNS,
    )
    assert list(rescue_queries(queries, evidence).entity_id) == ["S1-b"]


@pytest.mark.parametrize(
    "query,target,category",
    [
        (("null", "12 Main"), ("Alpha", "12 Main"), "missing_or_null_name"),
        (("Alpha", "N/A"), ("Alpha", "12 Main"), "missing_address"),
        (("", ""), ("Alpha", "12 Main"), "both_fields_sparse"),
        (("Café", "1 Road"), ("Cafe", "1 Road"), "accent_unicode"),
        (("Alpha LLC", "1 Road"), ("Alpha", "1 Road"), "legal_suffix"),
        (("Alpha Beta", "1 Road"), ("Beta Alpha", "1 Road"), "token_reorder"),
        (("Alpha Beta", "1 Road"), ("AlphaBeta", "1 Road"), "token_segmentation"),
        (("Alpha Beta", "1 Road"), ("AB", "1 Road"), "abbreviation_initials"),
        (("Alphaa Company", "1 Road"), ("Alpha Company", "1 Road"), "typographical_corruption"),
        (("Alpha", "12 Main"), ("Alpha", "13 Main"), "digit_address_conflict"),
        (("Москва", "1 Road"), ("Moskva", "1 Road"), "cross_language_like"),
        (("Alpha", "1 Road"), ("Zebra", "1 Road"), "other_unknown"),
    ],
)
def test_taxonomy_synthetic_examples(query, target, category):
    result = classify_variations(
        dict(zip(("name", "address"), query, strict=True)),
        dict(zip(("name", "address"), target, strict=True)),
        present=True,
    )
    assert result[category]
    assert result["fusion_truncation"] and not result["absent_every_channel"]
    assert set(result) == set(TAXONOMY)


def test_evaluation_zero_partial_full_and_official_metric():
    truth = {"S1-a": {"S2-a", "S3-a"}, "S1-b": {"S2-b"}, "S1-c": set()}
    frame = pd.DataFrame([("S1-a", "S2-a", "S2")], columns=["s1_id", "target_id", "target_source"])
    result = _metrics(frame, truth, set(), 0.1)
    assert result["full_owner_queries"] == 1
    assert result["partial_owner_queries"] == 1
    assert result["zero_owner_queries"] == 1
    assert result["recalled_edges"] == 1
    assert result["positive_edges"] == 3
    assert result["source_recall"]["S2"]["recall"] == 0.5
    assert result["incremental_edges_recovered"] == 1
    assert result["average_candidates"] == pytest.approx(1 / 3)
    assert result["oracle_macro_f0_5"] == pytest.approx((5 / 6 + 1) / 3)
    json.dumps(result)


def test_taxonomy_recovery_counts_are_actual_pairs():
    missed = pd.DataFrame(
        [("S1-a", "S2-a", True)],
        columns=["s1_id", "target_id", "absent_from_every_existing_top80_channel"],
    )
    query = pd.DataFrame([("S1-a", "Alpha", "1 Road", "US")], columns=pc.SOURCE_COLUMNS)
    target = pd.DataFrame([("S2-a", "Alpha", "", "US")], columns=pc.SOURCE_COLUMNS)
    report = taxonomy(missed, query, target, {"base": set(), "rescue": {("S1-a", "S2-a")}})
    assert report["categories"]["missing_address"]["recovered_by_configuration"] == {
        "base": 0,
        "rescue": 1,
    }
    assert report["layers"] == {"retrieval_channel": 1}
    assert "Alpha" not in json.dumps(report)


def test_rebudget_rejects_corrupt_cached_rescue(tmp_path):
    import yaml

    from entity_resolution.retrieval_rescue import rebudget_cached

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "rescue_evidence.parquet").write_bytes(b"corrupt")
    report = tmp_path / "evaluation.json"
    report.write_text(json.dumps({"context": {"files": {}}, "rescue_evidence_sha256": "expected"}))
    config = tmp_path / "rescue.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "cache": str(cache),
                "output": str(report),
                "data_paths_config": str(pc.DEFAULT_CONFIG.with_name("data_paths.yaml")),
            }
        )
    )
    with pytest.raises(ValueError, match="corrupt cached rescue"):
        rebudget_cached(config, 400)


def test_grouped_measurements_use_fold_mean_and_source_official_metric():
    from entity_resolution.retrieval_rescue import grouped_measurements

    truth = {"S1-a": {"S2-a"}, "S1-b": {"S3-b"}, "S1-c": {"S2-c"}}
    frame = pd.DataFrame([("S1-a", "S2-a", "S2")], columns=["s1_id", "target_id", "target_source"])
    result = grouped_measurements(frame, truth, set(), {"S1-a": 0, "S1-b": 1, "S1-c": 1}, [1], 0)
    assert result["fold_summary"]["mean_oracle"] == 0.5
    assert result["all"]["oracle_macro_f0_5"] == pytest.approx(1 / 3)
    assert result["all"]["source_recall"]["S2"]["oracle_macro_f0_5"] == pytest.approx(2 / 3)
    assert result["fold_1"]["zero_owner_queries"] == 2


def test_incremental_cost_counts_negative_candidates_and_lost_positives():
    truth = {"S1-a": {"S2-a", "S3-a"}}
    frame = pd.DataFrame(
        [("S1-a", "S2-a", "S2"), ("S1-a", "S2-negative", "S2"), ("S1-a", "S3-new-negative", "S3")],
        columns=["s1_id", "target_id", "target_source"],
    )
    baseline = {("S1-a", "S3-a"), ("S1-a", "S2-negative")}
    result = _metrics(frame, truth, baseline, 0)
    assert result["incremental_candidates_added"] == 2
    assert result["net_candidate_change"] == 1
    assert result["incremental_edges_recovered"] == 1
    assert result["edges_lost_vs_baseline"] == 1
    assert result["source_recall"]["S2"]["candidate_rows"] == 2


def test_streaming_oracle_matches_authoritative_in_memory_metrics(tmp_path):
    from entity_resolution.retrieval_rescue import grouped_measurements, measure_candidate_shards

    truth = {"S1-a": {"S2-a", "S3-a"}, "S1-b": {"S3-b"}, "S1-c": set()}
    frame = pd.DataFrame(
        [
            ("S1-a", "S2-a", "S2"),
            ("S1-a", "S2-negative", "S2"),
            ("S1-b", "S2-b", "S2"),
            ("S1-b", "S3-b", "S3"),
        ],
        columns=["s1_id", "target_id", "target_source"],
    )
    baseline = {("S1-a", "S3-a"), ("S1-a", "S2-negative")}
    folds = {"S1-a": 0, "S1-b": 1, "S1-c": 1}
    paths = []
    for i, (_, part) in enumerate(frame.groupby("s1_id")):
        path = tmp_path / f"part-{i}.parquet"
        part.to_parquet(path, index=False)
        paths.append(path)
    expected = grouped_measurements(frame, truth, baseline, folds, [1], 0)
    actual, positives = measure_candidate_shards(paths, truth, baseline, folds, [1], 0)
    assert positives == {("S1-a", "S2-a"), ("S1-b", "S3-b")}
    for name in expected:
        expected[name].pop("peak_rss_mb", None)
        actual[name].pop("peak_rss_mb", None)
        assert actual[name] == expected[name]
    with pytest.raises(ValueError, match="multiple candidate shards"):
        measure_candidate_shards([*paths, paths[0]], truth, baseline, folds, [1], 0)
