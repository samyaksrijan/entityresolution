"""Configuration-driven resolution of organizer data paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DATASET_KEYS = (
    "train_source1",
    "train_source2",
    "train_source3",
    "train_ground_truth",
    "test_source1",
    "test_source2",
    "test_source3",
)
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "data_paths.yaml"


class ConfigurationError(ValueError):
    """Raised when the data-path configuration is missing or invalid."""


@dataclass(frozen=True)
class DataPaths:
    """Resolved repository-root-relative paths from the YAML configuration."""

    repository_root: Path
    values: dict[str, Path]

    def __getitem__(self, key: str) -> Path:
        try:
            return self.values[key]
        except KeyError as exc:
            raise ConfigurationError(f"Unknown data path key: {key!r}") from exc


def load_data_paths(config_path: str | Path = DEFAULT_CONFIG) -> DataPaths:
    """Load and resolve configured paths without embedding organizer directory names."""
    path = Path(config_path).resolve()
    if not path.is_file():
        raise ConfigurationError(f"Data-path configuration does not exist: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"Expected a YAML mapping in {path}")

    missing = [key for key in DATASET_KEYS if not isinstance(raw.get(key), str)]
    if missing:
        raise ConfigurationError(f"Missing string path(s) in {path}: {', '.join(missing)}")

    repository_root = path.parent.parent
    resolved = {
        str(key): (repository_root / value).resolve()
        for key, value in raw.items()
        if isinstance(value, str)
    }
    return DataPaths(repository_root=repository_root, values=resolved)

