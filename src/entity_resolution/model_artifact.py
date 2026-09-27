"""Versioned native models and JSON metadata, atomically published and verified."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path

from catboost import CatBoostClassifier

from entity_resolution.decode import Decoder
from entity_resolution.training_schema import atomic_json, contract, sha256

ARTIFACT_VERSION = 1
REQUIRED = {
    "parameters",
    "seed",
    "fold_strategy",
    "fold_count",
    "sampling_policy",
    "decoder",
    "input_fingerprints",
    "candidate_diagnostics",
    "oof_summary",
    "identity",
    "versions",
}


def versions():
    return {
        name: importlib.metadata.version(name)
        for name in ("catboost", "numpy", "pandas", "pyarrow", "scikit-learn")
    }


def save_model(model, path):
    temporary = Path(str(path) + ".tmp")
    model.save_model(str(temporary), format="cbm")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def seal_artifact(work, output, metadata):
    if output.exists():
        raise ValueError(f"artifact exists: {output}")
    metadata = {
        **metadata,
        **contract(),
        "artifact_version": ARTIFACT_VERSION,
        "complete": True,
        "versions": versions(),
    }
    metadata["files"] = {
        str(p.relative_to(work)): sha256(p)
        for p in sorted(work.rglob("*"))
        if p.is_file() and p.name != "metadata.json"
    }
    atomic_json(work / "metadata.json", metadata)
    load_artifact(work)
    os.rename(work, output)
    return metadata


def load_artifact(path):
    path = Path(path)
    try:
        metadata = json.loads((path / "metadata.json").read_text())
        if not metadata.get("complete") or metadata.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("incomplete or incompatible artifact")
        if not metadata.keys() >= REQUIRED:
            raise ValueError("incomplete artifact metadata")
        for key, value in contract().items():
            if metadata.get(key) != value:
                raise ValueError(f"artifact feature order/schema mismatch: {key}")
        files = metadata["files"]
        if "model.cbm" not in files:
            raise ValueError("incomplete artifact: missing final model")
        for name, digest in files.items():
            file = path / name
            if (
                file.resolve().parent != path.resolve()
                and path.resolve() not in file.resolve().parents
            ):
                raise ValueError("invalid artifact file path")
            if not file.is_file() or sha256(file) != digest:
                raise ValueError(f"corrupt artifact file: {name}")
        Decoder(**metadata["decoder"])
        model = CatBoostClassifier()
        model.load_model(str(path / "model.cbm"), format="cbm")
        if model.feature_names_ != metadata["feature_names"]:
            raise ValueError("native model feature order mismatch")
        expected_cats = [
            metadata["feature_names"].index(c) for c in metadata["categorical_features"]
        ]
        if model.get_cat_feature_indices() != expected_cats:
            raise ValueError("native model categorical feature mismatch")
        return model, metadata
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"incomplete or corrupt artifact: {path}: {exc}") from exc
