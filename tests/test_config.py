from pathlib import Path

import pytest

from entity_resolution.config import ConfigurationError, load_data_paths


def test_load_data_paths_resolves_relative_to_repository(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "paths.yaml").write_text(
        "\n".join(
            [
                "train_source1: raw/train_source1.tsv",
                "train_source2: raw/train_source2.tsv",
                "train_source3: raw/train_source3.tsv",
                "train_ground_truth: raw/train_ground_truth.tsv",
                "test_source1: raw/test_source1.tsv",
                "test_source2: raw/test_source2.tsv",
                "test_source3: raw/test_source3.tsv",
            ]
        ),
        encoding="utf-8",
    )

    paths = load_data_paths(config_dir / "paths.yaml")

    assert paths["train_source1"] == (tmp_path / "raw/train_source1.tsv").resolve()


def test_load_data_paths_requires_all_datasets(tmp_path: Path) -> None:
    path = tmp_path / "paths.yaml"
    path.write_text("train_source1: raw/file.tsv\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Missing string path"):
        load_data_paths(path)

