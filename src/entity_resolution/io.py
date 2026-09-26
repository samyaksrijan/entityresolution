"""Safe, reusable ingestion and structural validation for organizer TSV files."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from entity_resolution.config import DATASET_KEYS, DataPaths, load_data_paths

SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")


class DataValidationError(ValueError):
    """Raised when an input violates the competition data contract."""


@dataclass(frozen=True)
class ValidationSummary:
    """Minimal structural result; deliberately excludes profiling statistics."""

    key: str
    path: Path
    rows: int
    positive_edges: int | None = None


def _read_header(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise DataValidationError(f"Input file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            row = next(csv.reader(handle, delimiter="\t"), None)
    except UnicodeDecodeError as exc:
        raise DataValidationError(f"Input is not valid UTF-8: {path}") from exc
    if row is None:
        raise DataValidationError(f"Input file is empty: {path}")
    return tuple(row)


def _validate_header(path: Path, required: tuple[str, ...]) -> None:
    actual = _read_header(path)
    if actual != required:
        raise DataValidationError(
            f"Unexpected header in {path}: got {list(actual)!r}; expected {list(required)!r}. "
            "Confirm this is the correct tab-separated organizer file."
        )


def _read_tsv(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    _validate_header(path, columns)
    try:
        return pd.read_csv(
            path,
            sep="\t",
            dtype={column: "string" for column in columns},
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8",
        )
    except (OSError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"Could not parse tab-separated input {path}: {exc}") from exc


def _validate_ids(series: pd.Series, *, label: str, prefix: str) -> None:
    if bool(series.eq("").any()):
        raise DataValidationError(f"{label} contains an empty entity ID")
    duplicates = series[series.duplicated(keep=False)]
    if not duplicates.empty:
        examples = ", ".join(duplicates.drop_duplicates().head(5).tolist())
        raise DataValidationError(f"{label} contains duplicate entity IDs, e.g. {examples}")
    wrong = series[~series.str.startswith(prefix, na=False)]
    if not wrong.empty:
        examples = ", ".join(wrong.head(5).tolist())
        raise DataValidationError(
            f"{label} contains IDs without required prefix {prefix!r}, e.g. {examples}"
        )


def load_source(key: str, paths: DataPaths | None = None) -> pd.DataFrame:
    """Load and validate one source TSV while preserving every textual value."""
    if not key.endswith(("source1", "source2", "source3")):
        raise DataValidationError(f"Not a source dataset key: {key!r}")
    configured = paths or load_data_paths()
    frame = _read_tsv(configured[key], SOURCE_COLUMNS)
    source_number = key[-1]
    _validate_ids(frame["entity_id"], label=key, prefix=f"S{source_number}-")
    return frame


def parse_ground_truth(paths: DataPaths | None = None) -> pd.DataFrame:
    """Return canonical positive edges as source1_entity_id/matched_entity_id rows."""
    configured = paths or load_data_paths()
    key = "train_ground_truth"
    frame = _read_tsv(configured[key], GROUND_TRUTH_COLUMNS)
    _validate_ids(frame["source1_entity_id"], label=key, prefix="S1-")

    edges: list[tuple[str, str]] = []
    for row_number, (source1_id, raw_matches) in enumerate(
        frame.itertuples(index=False, name=None), start=2
    ):
        if raw_matches == "":
            continue
        matches = raw_matches.split(",")
        if any(match == "" for match in matches):
            raise DataValidationError(
                f"{key} line {row_number} has an empty item in matched_entity_ids"
            )
        if len(matches) != len(set(matches)):
            raise DataValidationError(
                f"{key} line {row_number} has duplicate IDs in matched_entity_ids"
            )
        wrong = [match for match in matches if not match.startswith(("S2-", "S3-"))]
        if wrong:
            raise DataValidationError(
                f"{key} line {row_number} has non-S2/S3 matched IDs: {wrong[:5]!r}"
            )
        edges.extend((source1_id, match) for match in matches)
    return pd.DataFrame(edges, columns=["source1_entity_id", "matched_entity_id"], dtype="string")


def _iter_rows(path: Path, required: tuple[str, ...]) -> Iterator[tuple[int, list[str]]]:
    _validate_header(path, required)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(required):
                raise DataValidationError(
                    f"{path} line {line_number} has {len(row)} fields; expected {len(required)}"
                )
            yield line_number, row


def validate_source_file(key: str, path: Path) -> ValidationSummary:
    """Stream structural checks over a source file without retaining text columns."""
    expected_prefix = f"S{key[-1]}-"
    seen: set[str] = set()
    rows = 0
    for line_number, row in _iter_rows(path, SOURCE_COLUMNS):
        entity_id = row[0]
        if not entity_id.startswith(expected_prefix):
            raise DataValidationError(
                f"{key} line {line_number}: {entity_id!r} lacks prefix {expected_prefix!r}"
            )
        if entity_id in seen:
            raise DataValidationError(f"{key} line {line_number}: duplicate ID {entity_id!r}")
        seen.add(entity_id)
        rows += 1
    return ValidationSummary(key=key, path=path, rows=rows)


def validate_ground_truth_file(path: Path) -> ValidationSummary:
    """Stream checks over ground truth and count only rows and positive edges."""
    seen: set[str] = set()
    rows = 0
    positive_edges = 0
    for line_number, row in _iter_rows(path, GROUND_TRUTH_COLUMNS):
        source1_id, raw_matches = row
        if not source1_id.startswith("S1-"):
            raise DataValidationError(
                f"train_ground_truth line {line_number}: {source1_id!r} lacks prefix 'S1-'"
            )
        if source1_id in seen:
            raise DataValidationError(
                f"train_ground_truth line {line_number}: duplicate S1 ID {source1_id!r}"
            )
        seen.add(source1_id)
        matches = [] if raw_matches == "" else raw_matches.split(",")
        if len(matches) != len(set(matches)):
            raise DataValidationError(
                f"train_ground_truth line {line_number}: duplicate matched ID"
            )
        wrong = [match for match in matches if not match.startswith(("S2-", "S3-"))]
        if wrong:
            raise DataValidationError(
                f"train_ground_truth line {line_number}: invalid match IDs {wrong[:5]!r}"
            )
        positive_edges += len(matches)
        rows += 1
    return ValidationSummary(
        key="train_ground_truth", path=path, rows=rows, positive_edges=positive_edges
    )


def validate_all(paths: DataPaths | None = None) -> list[ValidationSummary]:
    """Stream-ingest and structurally validate all seven configured dataset files."""
    configured = paths or load_data_paths()
    summaries = []
    for key in DATASET_KEYS:
        if key == "train_ground_truth":
            summaries.append(validate_ground_truth_file(configured[key]))
        else:
            summaries.append(validate_source_file(key, configured[key]))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-all", action="store_true", help="validate all configured inputs"
    )
    args = parser.parse_args()
    if not args.validate_all:
        parser.error("choose --validate-all")
    for summary in validate_all():
        edge_text = (
            "" if summary.positive_edges is None else f", positive_edges={summary.positive_edges}"
        )
        print(f"PASS {summary.key}: rows={summary.rows}{edge_text}")


if __name__ == "__main__":
    main()
