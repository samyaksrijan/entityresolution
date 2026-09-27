"""Synthetic contract tests for the production pair-feature lane."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from entity_resolution.feature_materialization import materialize
from entity_resolution.feature_schema import (
    CATEGORICAL_FEATURES,
    FEATURE_DTYPES,
    NUMERIC_FEATURES,
    OUTPUT_SCHEMA,
    feature_names,
)
from entity_resolution.pair_features import SourceRecord, compute_pair_features


def _records() -> dict[str, SourceRecord]:
    return {
        "S1-a": SourceRecord("S1", "Café North LLC", "12 Main St 90210", "US"),
        "S1-b": SourceRecord("S1", "N/A", "", ""),
        "S2-a": SourceRecord("S2", "Cafe North", "12 Main Street 90210", "US"),
        "S2-b": SourceRecord("S2", "North Cafe", "99 Main Street 10001", "US"),
        "S3-a": SourceRecord("S3", "CN", "", ""),
    }


def _candidate(s1: str = "S1-a", target: str = "S2-a", **updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "s1_id": s1,
        "target_id": target,
        "target_source": target[:2],
        "channel": "name_char_tfidf",
        "retrieval_score": 0.8,
        "channel_rank": 2,
        "exact_match": False,
        "run_id": "fixture",
        "support_json": json.dumps(
            [
                {"channel": "name_char_tfidf", "score": 0.8, "rank": 2, "exact": False},
                {"channel": "address_char_tfidf", "score": 0.7, "rank": 3, "exact": False},
            ]
        ),
        "fusion_score": 0.3,
    }
    result.update(updates)
    return result


def _paths(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    paths = {}
    records = _records()
    for mode in ("train", "test"):
        for i in (1, 2, 3):
            file = data_dir / f"{mode}_source{i}.tsv"
            rows = [(key, value) for key, value in records.items() if key.startswith(f"S{i}-")]
            lines = ["entity_id\tbusiness_name\tbusiness_address\tcountry"]
            lines.extend(f"{key}\t{v.name}\t{v.address}\t{v.country}" for key, v in rows)
            file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            paths[f"{mode}_source{i}"] = f"raw/{file.name}"
    paths["train_ground_truth"] = "raw/absent_truth.tsv"
    config = config_dir / "data_paths.yaml"
    # Config loader resolves paths from its parent directory's parent.
    config.write_text("\n".join(f"{key}: {value}" for key, value in paths.items()))
    return config


def _read_output(output: Path) -> pd.DataFrame:
    return pd.concat(
        (pd.read_parquet(path) for path in sorted(output.glob("part-*.parquet"))),
        ignore_index=True,
    )


def test_actual_schema_support_and_stable_dtypes() -> None:
    frame = compute_pair_features(pd.DataFrame([_candidate()]), _records())
    row = frame.iloc[0]
    assert list(frame) == OUTPUT_SCHEMA.names
    assert feature_names() == (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)
    assert set(feature_names()) == set(FEATURE_DTYPES)
    assert all(str(frame[name].dtype) == FEATURE_DTYPES[name] for name in NUMERIC_FEATURES)
    assert row.channel_support_count == 2
    assert row.s2_support_count == 2
    assert row.s3_support_count == 0
    assert row.independent_channel_agreement == 1
    assert row.support_mean_rank == pytest.approx(2.5)
    assert row.strongest_channel == "name_char_tfidf"
    assert row.name_accent_exact == 0  # suffix remains
    assert row.name_legal_suffix_agreement == 0  # Café/Cafe differs before accent fold
    assert row.address_house_number_agreement == 1
    assert row.address_postal_agreement == 1
    assert row.country_agreement == 1


def test_fallback_and_text_variants() -> None:
    fallback = {k: v for k, v in _candidate().items() if k not in {"support_json", "fusion_score"}}
    records = _records()
    records["S2-a"] = SourceRecord("S2", "CAFÉ NORTH", "12 Main St 90210", "US")
    row = compute_pair_features(pd.DataFrame([fallback]), records).iloc[0]
    assert row.channel_support_count == 1
    assert row.fusion_score == pytest.approx(0.8)
    assert row.name_casefold_exact == 0  # legal suffix on S1
    assert row.name_legal_suffix_agreement == 1
    assert row.address_normalized_exact == 1
    records["S2-a"] = SourceRecord("S2", "Cafe North LLC", "12 Main St 90210", "US")
    row = compute_pair_features(pd.DataFrame([fallback]), records).iloc[0]
    assert row.name_accent_exact == 1
    assert row.name_raw_exact == 0


@pytest.mark.parametrize("dropped", ["support_json", "fusion_score"])
def test_independent_optional_candidate_columns(dropped: str) -> None:
    candidate = _candidate()
    del candidate[dropped]
    row = compute_pair_features(pd.DataFrame([candidate]), _records()).iloc[0]
    assert row.channel_support_count == (1 if dropped == "support_json" else 2)
    assert row.fusion_score == pytest.approx(0.8 if dropped == "fusion_score" else 0.3)


def test_exact_support_and_punctuation() -> None:
    records = _records()
    records["S2-a"] = SourceRecord("S2", "Café—North LLC", "12 Main St 90210", "US")
    support = json.dumps(
        [
            {"channel": "exact_name_unicode", "score": 1.0, "rank": 1, "exact": True},
            {"channel": "exact_address_unicode", "score": 1.0, "rank": 1, "exact": True},
        ]
    )
    row = compute_pair_features(
        pd.DataFrame(
            [
                _candidate(
                    channel="exact_name_unicode",
                    retrieval_score=1.0,
                    channel_rank=1,
                    exact_match=True,
                    support_json=support,
                )
            ]
        ),
        records,
    ).iloc[0]
    assert row.exact_support_count == 2
    assert row.exact_support_family_count == 2
    assert row.name_punctuation_agreement == 1
    assert row.name_raw_exact == 0
    assert row.exact_name_and_address == 1
    assert row.independent_channel_agreement == 1


def test_token_order_acronym_nulls_and_digit_conflict() -> None:
    rows = [_candidate(target="S2-b"), _candidate(s1="S1-b", target="S3-a")]
    result = compute_pair_features(pd.DataFrame(rows), _records())
    assert result.loc[0, "name_token_jaccard"] == pytest.approx(2 / 3)
    assert result.loc[0, "name_token_sort_similarity"] > result.loc[0, "name_char_similarity"]
    assert result.loc[0, "exact_name_conflicting_address_digits"] == 0
    assert result.loc[1, "name_s1_null_like"] == 1
    assert result.loc[1, "both_names_usable"] == 0
    assert result.loc[1, "both_addresses_usable"] == 0
    assert result.loc[1, "name_char_similarity"] == 0


def test_exact_name_address_digit_conflict_and_acronym() -> None:
    records = _records()
    records["S2-b"] = SourceRecord("S2", "Café North LLC", "99 Broadway 10001", "US")
    row = compute_pair_features(pd.DataFrame([_candidate(target="S2-b")]), records).iloc[0]
    assert row.name_normalized_exact == 1
    assert row.address_number_overlap == 0
    assert row.exact_name_conflicting_address_digits == 1
    records["S1-a"] = SourceRecord("S1", "Cafe North", "", "US")
    records["S3-a"] = SourceRecord("S3", "CN", "", "US")
    acronym = compute_pair_features(pd.DataFrame([_candidate(target="S3-a")]), records).iloc[0]
    assert acronym.name_acronym_agreement == 1


@pytest.mark.parametrize("support", ["{", "[]", "{}", '[{"channel":"x"}]'])
def test_malformed_support_fails(support: str) -> None:
    with pytest.raises(ValueError, match="support_json"):
        compute_pair_features(pd.DataFrame([_candidate(support_json=support)]), _records())


def test_inconsistent_support_fails() -> None:
    with pytest.raises(ValueError, match="primary score or rank"):
        compute_pair_features(pd.DataFrame([_candidate(channel_rank=4)]), _records())


def test_ownership_and_absent_records_fail() -> None:
    with pytest.raises(ValueError, match="ownership"):
        compute_pair_features(pd.DataFrame([_candidate(target_source="S3")]), _records())
    with pytest.raises(ValueError, match="absent"):
        compute_pair_features(pd.DataFrame([_candidate(target="S2-missing")]), _records())


def test_materialization_batch_invariance_resume_and_no_label_access(tmp_path: Path) -> None:
    config = _paths(tmp_path)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pd.DataFrame(
        [_candidate(), _candidate(target="S2-b"), _candidate(s1="S1-b", target="S3-a")]
    ).to_parquet(input_dir / "part-000000.parquet", index=False)
    common = dict(mode="train", candidates=input_dir, data_paths_config=config)
    first, second = tmp_path / "first", tmp_path / "second"
    state = materialize(**common, output=first, batch_size=1, shard_size=1)
    materialize(**common, output=second, batch_size=2, shard_size=3)
    assert state["complete"] and state["rows"] == 3
    assert not (first / "records.sqlite").exists()
    pd.testing.assert_frame_equal(_read_output(first), _read_output(second))
    assert pq.read_table(first / "part-000000.parquet").schema.equals(OUTPUT_SCHEMA)
    assert not any("business_name" in name for name in _read_output(first))
    assert materialize(**common, output=first, batch_size=1, shard_size=1, resume=True) == state
    with pytest.raises(ValueError, match="incompatible"):
        materialize(**common, output=first, batch_size=2, shard_size=2, resume=True)


@pytest.mark.parametrize(
    "dropped", [("support_json", "fusion_score"), ("support_json",), ("fusion_score",)]
)
def test_materialization_optional_columns(tmp_path: Path, dropped: tuple[str, ...]) -> None:
    config = _paths(tmp_path)
    candidate = _candidate()
    for column in dropped:
        del candidate[column]
    input_file = tmp_path / "candidates.parquet"
    pd.DataFrame([candidate]).to_parquet(input_file, index=False)
    output = tmp_path / "output"
    assert (
        materialize(mode="test", candidates=input_file, output=output, data_paths_config=config)[
            "rows"
        ]
        == 1
    )
    assert _read_output(output).iloc[0].channel_support_count == (
        1 if "support_json" in dropped else 2
    )


def test_corruption_atomic_resume_and_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _paths(tmp_path)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pd.DataFrame([_candidate(), _candidate(target="S2-b")]).to_parquet(
        input_dir / "part-000000.parquet", index=False
    )
    output = tmp_path / "output"
    from entity_resolution import feature_materialization as fm

    original = fm._write_shard
    calls = 0

    def interrupt(table: object, path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        original(table, path)

    monkeypatch.setattr(fm, "_write_shard", interrupt)
    kwargs = dict(
        mode="test",
        candidates=input_dir,
        output=output,
        data_paths_config=config,
        batch_size=1,
        shard_size=1,
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        materialize(**kwargs)
    assert not list(output.glob("*.tmp"))
    monkeypatch.setattr(fm, "_write_shard", original)
    assert materialize(**kwargs, resume=True)["rows"] == 2
    (output / "part-000000.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        materialize(**kwargs, resume=True)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    state = materialize(
        mode="test",
        candidates=empty_dir,
        output=tmp_path / "empty_output",
        data_paths_config=config,
    )
    assert state["complete"] and state["rows"] == 0


def test_atomic_shard_write_cleans_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from entity_resolution import feature_materialization as fm

    table = fm.pa.Table.from_pandas(
        compute_pair_features(pd.DataFrame([_candidate()]), _records()),
        schema=OUTPUT_SCHEMA,
        preserve_index=False,
    )
    original = fm.pq.write_table

    def fail_after_write(*args: object, **kwargs: object) -> None:
        original(*args, **kwargs)
        raise OSError("interrupted write")

    monkeypatch.setattr(fm.pq, "write_table", fail_after_write)
    final = tmp_path / "part-000000.parquet"
    with pytest.raises(OSError, match="interrupted write"):
        fm._write_shard(table, final)
    assert not final.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_duplicate_and_corrupt_input_rejected(tmp_path: Path) -> None:
    config = _paths(tmp_path)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pd.DataFrame([_candidate(), _candidate()]).to_parquet(
        input_dir / "part-000000.parquet", index=False
    )
    with pytest.raises(ValueError, match="duplicate"):
        materialize(
            mode="train",
            candidates=input_dir,
            output=tmp_path / "output",
            data_paths_config=config,
            batch_size=1,
            shard_size=1,
        )
    (input_dir / "part-000000.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        materialize(
            mode="train", candidates=input_dir, output=tmp_path / "other", data_paths_config=config
        )


def test_incompatible_manifest_run_id_and_raw_output_guard(tmp_path: Path) -> None:
    config = _paths(tmp_path)
    input_file = tmp_path / "candidates.parquet"
    pd.DataFrame([_candidate()]).to_parquet(input_file, index=False)
    kwargs = dict(mode="train", candidates=input_file, data_paths_config=config)
    with pytest.raises(ValueError, match="organizer raw data"):
        materialize(**kwargs, output=tmp_path / "raw" / "features")
    output = tmp_path / "features"
    materialize(**kwargs, output=output)
    manifest = output / "manifest.json"
    state = json.loads(manifest.read_text())
    state["schema"][0][0] = "wrong_id"
    manifest.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="incompatible"):
        materialize(**kwargs, output=output, resume=True)
    pd.DataFrame([_candidate(), _candidate(target="S2-b", run_id="other")]).to_parquet(
        input_file, index=False
    )
    with pytest.raises(ValueError, match="run ID"):
        materialize(**kwargs, output=tmp_path / "mixed", batch_size=1, shard_size=1)


def test_incomplete_candidate_manifest_fails(tmp_path: Path) -> None:
    config = _paths(tmp_path)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pd.DataFrame([_candidate()]).to_parquet(input_dir / "part-000000.parquet", index=False)
    (input_dir / "manifest.json").write_text(json.dumps({"complete": False}))
    with pytest.raises(ValueError, match="incomplete"):
        materialize(
            mode="test",
            candidates=input_dir,
            output=tmp_path / "features",
            data_paths_config=config,
        )
