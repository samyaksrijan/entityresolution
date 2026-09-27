from entity_resolution.forensic_analysis import _candidate_summary, _pair_features


def test_candidate_summary_counts_singletons_and_all_match_coverage() -> None:
    candidates = {"S1-a": {"S2-x", "S2-z"}, "S1-b": set()}
    truth = {"S1-a": {"S2-x", "S2-y"}, "S1-b": set()}

    summary = _candidate_summary(candidates, truth)

    assert summary["positive_edge_recall"] == 0.5
    assert summary["all_true_matches_per_s1_coverage"] == 0.5
    assert summary["average_candidates"] == 1.0
    assert summary["maximum_candidates"] == 2


def test_pair_features_measure_digit_conflict_without_id_components() -> None:
    features = _pair_features(
        (
            "S1-999",
            "S2-123",
            "S2",
            "Café North LLC",
            "12 Main Street",
            "US",
            "Cafe North",
            "98 Main St.",
            "US",
        )
    )

    assert features["name_normalized_exact"] is False
    assert features["digit_conflict"] is True
    assert features["digit_overlap"] == 0.0
    assert "source1_numeric_component" not in features
