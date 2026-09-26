from pathlib import Path

import pandas as pd
import pytest

from entity_resolution.config import DataPaths
from entity_resolution.io import (
    DataValidationError,
    load_source,
    parse_ground_truth,
    validate_ground_truth_file,
    validate_source_file,
)


def _paths(tmp_path: Path, **overrides: Path) -> DataPaths:
    values = {
        "train_source1": tmp_path / "train_source1.tsv",
        "train_source2": tmp_path / "train_source2.tsv",
        "train_source3": tmp_path / "train_source3.tsv",
        "train_ground_truth": tmp_path / "train_ground_truth.tsv",
        "test_source1": tmp_path / "test_source1.tsv",
        "test_source2": tmp_path / "test_source2.tsv",
        "test_source3": tmp_path / "test_source3.tsv",
    }
    values.update(overrides)
    return DataPaths(repository_root=tmp_path, values=values)


def test_load_source_preserves_text_and_empty_values(tmp_path: Path) -> None:
    source = tmp_path / "train_source1.tsv"
    source.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-0001\t  A&B, Inc.  \t\tFrance\n",
        encoding="utf-8",
    )

    frame = load_source("train_source1", _paths(tmp_path))

    assert frame.loc[0, "business_name"] == "  A&B, Inc.  "
    assert frame.loc[0, "business_address"] == ""
    assert all(isinstance(dtype, pd.StringDtype) for dtype in frame.dtypes)


def test_load_source_rejects_comma_header(tmp_path: Path) -> None:
    source = tmp_path / "train_source1.tsv"
    source.write_text("entity_id,business_name,business_address,country\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="Unexpected header"):
        load_source("train_source1", _paths(tmp_path))


def test_load_source_rejects_duplicate_and_wrong_prefix(tmp_path: Path) -> None:
    source = tmp_path / "train_source2.tsv"
    source.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tA\tX\tUS\nS1-1\tB\tY\tUS\n",
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="duplicate entity IDs"):
        load_source("train_source2", _paths(tmp_path))


def test_load_source_rejects_wrong_source_prefix(tmp_path: Path) -> None:
    source = tmp_path / "train_source2.tsv"
    source.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\nS3-1\tA\tX\tUS\n",
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="required prefix 'S2-'"):
        load_source("train_source2", _paths(tmp_path))


def test_parse_ground_truth_returns_canonical_positive_edges(tmp_path: Path) -> None:
    truth = tmp_path / "train_ground_truth.tsv"
    truth.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-2,S3-3\n"
        "S1-2\t\n",
        encoding="utf-8",
    )

    edges = parse_ground_truth(_paths(tmp_path))

    assert edges.to_dict("records") == [
        {"source1_entity_id": "S1-1", "matched_entity_id": "S2-2"},
        {"source1_entity_id": "S1-1", "matched_entity_id": "S3-3"},
    ]


def test_parse_ground_truth_rejects_duplicate_targets(tmp_path: Path) -> None:
    truth = tmp_path / "train_ground_truth.tsv"
    truth.write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-2,S2-2\n", encoding="utf-8"
    )

    with pytest.raises(DataValidationError, match="duplicate IDs"):
        parse_ground_truth(_paths(tmp_path))


def test_parse_ground_truth_rejects_invalid_target_source(tmp_path: Path) -> None:
    truth = tmp_path / "train_ground_truth.tsv"
    truth.write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS1-2\n", encoding="utf-8"
    )

    with pytest.raises(DataValidationError, match="non-S2/S3"):
        parse_ground_truth(_paths(tmp_path))


def test_streaming_validators_report_rows_and_edges(tmp_path: Path) -> None:
    source = tmp_path / "test_source3.tsv"
    source.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\nS3-1\tA\tX\tUS\n",
        encoding="utf-8",
    )
    truth = tmp_path / "train_ground_truth.tsv"
    truth.write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\t\n",
        encoding="utf-8",
    )

    source_summary = validate_source_file("test_source3", source)
    truth_summary = validate_ground_truth_file(truth)

    assert source_summary.rows == 1
    assert truth_summary.rows == 2
    assert truth_summary.positive_edges == 2
