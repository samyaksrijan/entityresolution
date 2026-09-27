import json
from pathlib import Path

import pandas as pd
import pytest

from entity_resolution.candidate_fusion import (
    EvidenceValidationError,
    FusionSpec,
    aggregate_evidence,
    select_candidates,
    validate_and_load_evidence,
)
from entity_resolution.candidate_schema import CandidateEvidence, evidence_frame
from entity_resolution.missed_edge_analysis import audit_missed_edges


def _queries() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("S1-a", "Cafe Alpha", "12 Main St", "US"),
            ("S1-b", "Beta LLC", "", "India"),
        ],
        columns=["entity_id", "business_name", "business_address", "country"],
    )


def _aggregated() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("S1-a", "S2-z", "S2", "b", True, False, 2, 0.01, 0.02),
            ("S1-a", "S2-a", "S2", "c", False, False, 2, 0.005, 0.005),
            ("S1-a", "S3-y", "S3", "same", False, False, 1, 0.02, 0.00),
            ("S1-a", "S3-x", "S3", "same", False, False, 1, 0.02, 0.00),
            ("S1-b", "S2-q", "S2", "q", False, False, 1, 0.01, 0.00),
        ],
        columns=[
            "s1_id",
            "target_id",
            "target_source",
            "target_content_key",
            "exact_name",
            "exact_address",
            "supporting_channels",
            "rrf_name",
            "rrf_address",
        ],
    ).assign(
        rrf_total=lambda frame: frame["rrf_name"] + frame["rrf_address"],
        rrf_exact_name=lambda frame: frame["rrf_name"].where(frame["exact_name"], 0.0),
        rrf_exact_address=lambda frame: frame["rrf_address"].where(
            frame["exact_address"], 0.0
        ),
    )


def test_fusion_preserves_exact_and_expands_content_ties_without_id_order() -> None:
    aggregated = _aggregated()
    first = select_candidates(aggregated, FusionSpec("rrf", global_budget=2))
    shuffled = select_candidates(
        aggregated.sample(frac=1, random_state=7), FusionSpec("rrf", global_budget=2)
    )
    first_pairs = set(first[["s1_id", "target_id"]].itertuples(index=False, name=None))
    shuffled_pairs = set(shuffled[["s1_id", "target_id"]].itertuples(index=False, name=None))
    assert first_pairs == shuffled_pairs
    assert ("S1-a", "S2-z") in first_pairs  # qualifying exact is retained
    # Both identical evidence/content ties cross the nominal boundary together.
    tied = {pair for pair in first_pairs if pair[1] in {"S3-x", "S3-y"}}
    assert tied == {("S1-a", "S3-x"), ("S1-a", "S3-y")}


def test_aggregate_keeps_channel_scores_separate_and_builds_rank_features() -> None:
    evidence = evidence_frame(
        [
            CandidateEvidence(
                "S1-a",
                "S2-x",
                "S2",
                "exact_name_accent_folded",
                "name",
                "accent_folded",
                1.0,
                1,
                True,
                "v1",
            ),
            CandidateEvidence(
                "S1-a",
                "S2-x",
                "S2",
                "name_char_tfidf",
                "name",
                "unicode_preserving",
                0.73,
                4,
                False,
                "v1",
            ),
        ]
    )
    target = pd.DataFrame(
        [
            {
                "target_id": "S2-x",
                "target_content_key": "content",
                "target_name_missing": False,
                "target_address_missing": False,
                "target_address_digits": ("12",),
            }
        ]
    )
    result = aggregate_evidence(evidence, _queries().iloc[:1], target, rrf_k=60)
    assert result.loc[0, "exact_name_accent_folded"]
    assert result.loc[0, "score_name_char_tfidf"] == pytest.approx(0.73)
    assert result.loc[0, "normalized_rank_name_char_tfidf"] == pytest.approx(1 - 3 / 79)
    assert result.loc[0, "supporting_channels"] == 2
    assert result.loc[0, "address_digit_exact_agreement"]


def test_validator_fails_loudly_when_required_artifacts_are_missing(tmp_path: Path) -> None:
    sample = tmp_path / "sample.parquet"
    pd.DataFrame({"s1_id": ["S1-a"]}).to_parquet(sample, index=False)
    config = tmp_path / "v1.yaml"
    config.write_text(
        "\n".join(
            [
                "run_id: v1",
                "sparse_top_k: 80",
                "normalization_view: unicode_preserving",
                "sparse_threshold: 0.2",
                "country_partition: true",
                "tfidf:",
                "  ngram_range: [3, 5]",
                "  min_df: 2",
                "  max_features: 50000",
            ]
        )
    )
    with pytest.raises(EvidenceValidationError, match="required evidence artifact is missing"):
        validate_and_load_evidence(
            tmp_path / "artifacts", sample_path=sample, v1_config_path=config
        )
    with pytest.raises(EvidenceValidationError, match="sample fingerprint mismatch"):
        validate_and_load_evidence(
            tmp_path / "artifacts",
            sample_path=sample,
            v1_config_path=config,
            expected_fingerprints={sample.name: "stale"},
        )


def test_missed_edge_audit_classifies_absent_and_fusion_truncated_edges(
    tmp_path: Path,
) -> None:
    train = tmp_path / "dataset" / "train"
    test = tmp_path / "dataset" / "test"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    (train / "train_source1.tsv").write_text(
        header + "S1-a\tCafe Alpha LLC\t12 Main St\tUS\n"
    )
    (train / "train_source2.tsv").write_text(
        header
        + "S2-x\tCafe Alpha\t12 Main St\tUS\n"
        + "S2-y\tCafe Alpha\t\tUS\n"
    )
    (train / "train_source3.tsv").write_text(header + "S3-z\tOther\t8 Side St\tUS\n")
    (train / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-a\tS2-x,S2-y\n"
    )
    for source in (1, 2, 3):
        (test / f"test_source{source}.tsv").write_text(header)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    paths = config_dir / "data_paths.yaml"
    paths.write_text(
        "\n".join(
            [
                "train_source1: dataset/train/train_source1.tsv",
                "train_source2: dataset/train/train_source2.tsv",
                "train_source3: dataset/train/train_source3.tsv",
                "train_ground_truth: dataset/train/train_ground_truth.tsv",
                "test_source1: dataset/test/test_source1.tsv",
                "test_source2: dataset/test/test_source2.tsv",
                "test_source3: dataset/test/test_source3.tsv",
            ]
        )
    )
    evidence = pd.DataFrame(
        [
            {
                "s1_id": "S1-a",
                "target_id": "S2-x",
                "channel": "name_char_tfidf",
                "channel_rank": 20,
            }
        ]
    )
    selected = pd.DataFrame(columns=["s1_id", "target_id"])
    sample = pd.DataFrame([{"s1_id": "S1-a", "country": "US"}])
    frame, summary = audit_missed_edges(
        selected=selected,
        all_evidence=evidence,
        truth={"S1-a": {"S2-x", "S2-y"}},
        sample=sample,
        queries=_queries().iloc[:1],
        data_paths_config=paths,
    )
    by_target = frame.set_index("target_id")
    assert by_target.loc["S2-x", "name_retrieved_below_selected_budget"]
    assert by_target.loc["S2-y", "absent_from_every_existing_top80_channel"]
    assert by_target.loc["S2-y", "target_address_missing"]
    assert summary["missed_edges"] == 2
    json.dumps(summary)  # summary is JSON serializable
