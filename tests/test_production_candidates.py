import copy
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

from entity_resolution import production_candidates as pc
from entity_resolution.candidate_fusion import fuse_budgeted


@pytest.fixture
def fixture_run(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    paths = {}
    header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    records = {
        1: "S1-a\tCafe Alpha\t12 Main St\tUS\nS1-b\tnull\t\tUS\nS1-c\tBeta\t9 Road\tUS\n",
        2: "S2-a\tCafé Alpha\t12 Main St\tUS\nS2-b\tBeta\t9 Road\tUS\n",
        3: "S3-a\tCafe Alpha\t\tUS\nS3-b\tGamma\t9 Road\tUS\n",
    }
    for mode in ("train", "test"):
        for source in (1, 2, 3):
            file = raw / f"{mode}{source}.tsv"
            file.write_text(header + records[source])
            paths[f"{mode}_source{source}"] = str(file)
    # Deliberately nonexistent: generation must never try to open labels.
    paths["train_ground_truth"] = str(raw / "FORBIDDEN_LABELS")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    data_paths = config_dir / "data_paths.yaml"
    data_paths.write_text(yaml.safe_dump(paths))
    config = pc.load_profile(pc.DEFAULT_CONFIG, "safe_k40")
    config["vectorizer_root"] = None
    config.update(batch_size=2, shard_size=1, index_block_size=2, fit_sample_size=10, threads=1)
    config["sparse"].update(min_df=1, ngram_min=2, ngram_max=3, top_k=2)
    return config, data_paths, tmp_path / "out"


def generate(fixture_run, **kwargs):
    config, paths, output = fixture_run
    return pc.generate(
        config,
        mode=kwargs.pop("mode", "test"),
        run_id="test",
        output=output,
        data_paths_config=paths,
        **kwargs,
    )


def read_output(output):
    return (
        pd.concat(
            [pd.read_parquet(p) for p in sorted(output.glob("part-*.parquet"))], ignore_index=True
        )
        .sort_values(["s1_id", "target_source", "target_id"])
        .reset_index(drop=True)
    )


def test_schema_exact_and_test_label_isolation(fixture_run):
    state = generate(fixture_run)
    output = fixture_run[2]
    frame = read_output(output)
    assert state["complete"] and state["unique_query_count"] == 3
    assert state["input_fingerprints"].keys() == {"test_source1", "test_source2", "test_source3"}
    assert pq.read_schema(output / "part-000000.parquet").equals(pc.SCHEMA, check_metadata=False)
    assert frame.retrieval_score.dtype == "float32"
    assert frame.channel_rank.dtype == "int16"
    assert frame.exact_match.dtype == "bool"
    assert not frame.duplicated(["s1_id", "target_id", "target_source"]).any()
    assert set(frame[frame.s1_id == "S1-a"].target_id) >= {"S2-a", "S3-a"}
    assert "S1-b" not in set(frame.s1_id)
    before = {p.name: pc._sha256(p) for p in output.glob("part-*.parquet")}
    assert generate(fixture_run, resume=True)["complete"]
    assert before == {p.name: pc._sha256(p) for p in output.glob("part-*.parquet")}


def test_interruption_resume_and_atomic_failure(fixture_run, monkeypatch):
    original = pc.atomic_shard

    def interrupted(frame, path, run_id):
        if path.name == "part-000001.parquet":
            path.with_suffix(".parquet.tmp").write_bytes(b"interrupted")
            raise RuntimeError("interrupted")
        original(frame, path, run_id)

    monkeypatch.setattr(pc, "atomic_shard", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        generate(fixture_run)
    output = fixture_run[2]
    state = json.loads((output / "manifest.json").read_text())
    assert len(state["completed_shards"]) == 1 and not state["complete"]
    assert not (output / "part-000001.parquet").exists()
    digest = pc._sha256(output / "part-000000.parquet")
    monkeypatch.setattr(pc, "atomic_shard", original)
    assert generate(fixture_run, resume=True)["complete"]
    assert digest == pc._sha256(output / "part-000000.parquet")
    bad = read_output(output)
    bad["target_source"] = "S1"
    with pytest.raises(ValueError, match="source ownership"):
        pc.atomic_shard(bad, output / "bad.parquet", "test")
    assert not (output / "bad.parquet").exists()


@pytest.mark.parametrize("mutation", ["config", "run_id", "schema", "source", "shard", "index"])
def test_incompatible_and_corrupt_resume(fixture_run, mutation):
    generate(fixture_run)
    config, paths, output = fixture_run
    if mutation == "config":
        config["policy"]["top_k"] = 20
    elif mutation in {"run_id", "schema"}:
        path = output / "manifest.json"
        state = json.loads(path.read_text())
        state["run_id" if mutation == "run_id" else "schema_version"] = "wrong"
        path.write_text(json.dumps(state))
    elif mutation == "source":
        path = Path(yaml.safe_load(paths.read_text())["test_source2"])
        path.write_text(path.read_text() + "S2-z\tZ\t1 X\tUS\n")
    elif mutation == "shard":
        (output / "part-000000.parquet").write_bytes(b"corrupt")
    else:
        # An incomplete run must revalidate its cache before retrieving more shards.
        path = output / "manifest.json"
        state = json.loads(path.read_text())
        state["complete"] = False
        path.write_text(json.dumps(state))
        (output / "index/records.sqlite").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="incompatible|corrupt"):
        generate(fixture_run, resume=True)


def test_batch_boundary_invariance_and_train_mode(fixture_run):
    generate(fixture_run)
    first = read_output(fixture_run[2])
    config, paths, output = fixture_run
    config = copy.deepcopy(config)
    config.update(batch_size=3, shard_size=3)
    other = (config, paths, output.with_name("other"))
    generate(other, mode="train")
    pd.testing.assert_frame_equal(first, read_output(other[2]))


@pytest.mark.parametrize("bad", ["S3-wrong", "S2-a"])
def test_identifier_integrity_across_batches(fixture_run, bad):
    _, paths, _ = fixture_run
    path = Path(yaml.safe_load(paths.read_text())["test_source2"])
    path.write_text(path.read_text() + f"{bad}\tBeta\t9 Road\tUS\n")
    with pytest.raises(ValueError, match="ownership|duplicate"):
        generate(fixture_run)


def test_empty_inputs_and_configuration_hash(fixture_run):
    config, paths, output = fixture_run
    empty = pd.DataFrame(columns=pc.SOURCE_COLUMNS)
    assert pc.CachedRetriever(output / "unbuilt", config).retrieve(empty, "test").empty
    for key, value in yaml.safe_load(paths.read_text()).items():
        if key.startswith("test_source"):
            Path(value).write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
    assert generate(fixture_run)["candidate_rows"] == 0
    assert pc.configuration_hash({"a": 1, "b": 2}) == pc.configuration_hash({"b": 2, "a": 1})
    assert pc.configuration_hash({"a": 1}) != pc.configuration_hash({"a": 2})
    config["labels"] = "forbidden"
    with pytest.raises(ValueError, match="labels"):
        generate(fixture_run)


def evidence():
    return pd.DataFrame(
        [
            ("S1-a", "S2-a", "S2", "name", 0.8, 1, False, "r", "a"),
            ("S1-a", "S2-b", "S2", "name", 0.7, 2, False, "r", "b"),
            ("S1-a", "S3-a", "S3", "address", 0.1, 9, False, "r", "c"),
            ("S1-a", "S3-x", "S3", "exact_name", 1.0, 1, True, "r", "x"),
        ],
        columns=pc.EVIDENCE_COLUMNS,
    )


def queries():
    return pd.DataFrame([("S1-a", "Alpha", "12 Main St", "US")], columns=pc.SOURCE_COLUMNS)


def test_fusion_source_quotas_spillover_dedup_and_topk():
    rows = evidence()
    selected, _ = fuse_budgeted(
        pd.concat([rows, rows]),
        queries(),
        {"top_k": 3, "source_quotas": {"S2": 1, "S3": 2}},
        run_id="r",
    )
    assert set(selected.target_id) == {"S2-a", "S3-a", "S3-x"}
    assert all(len(json.loads(v)) == 1 for v in selected.support_json)
    selected, _ = fuse_budgeted(rows, queries(), {"top_k": 1}, run_id="r")
    assert list(selected.target_id) == ["S3-x"]
    selected, _ = fuse_budgeted(
        rows[rows.target_source == "S2"],
        queries(),
        {"top_k": 2, "source_quotas": {"S3": 2}},
        run_id="r",
    )
    assert len(selected) == 2


def test_stable_ties_adaptive_and_exact_overflow():
    rows = evidence()
    rows.loc[1, ["channel_rank", "retrieval_score", "target_content_key"]] = [1, 0.8, "a"]
    policy = {"top_k": 2}
    first, _ = fuse_budgeted(rows, queries(), policy, run_id="r")
    second, _ = fuse_budgeted(rows.sample(frac=1), queries(), policy, run_id="r")
    pd.testing.assert_frame_equal(first, second)
    assert set(first.target_id) == {"S2-a", "S2-b", "S3-x"}  # boundary ties expand
    q = queries()
    q["business_address"] = "N/A"
    selected, gating = fuse_budgeted(
        rows, q, {"top_k": 1, "expanded_k": 4, "adaptive": True}, run_id="r"
    )
    assert len(selected) == 4 and gating["adaptive_queries"] == 1
    rows["exact_match"] = True
    selected, _ = fuse_budgeted(rows, queries(), {"top_k": 1}, run_id="r")
    assert len(selected) == 4


def test_submission_uses_authoritative_writer_and_decoder(fixture_run):
    generate(fixture_run)
    destination = pc.export_fallback(
        fixture_run[2], fixture_run[2].parent / "submission", "fallback"
    )
    matches = pd.read_csv(destination / "matching_results.tsv", sep="\t", keep_default_na=False)
    assert set(matches.source1_entity_id) == {"S1-a", "S1-b", "S1-c"}
    assert matches.set_index("source1_entity_id").loc["S1-a", "matched_entity_ids"] == "S2-a"
    assert matches.set_index("source1_entity_id").loc["S1-b", "matched_entity_ids"] == ""
    candidates = pd.read_csv(destination / "candidate_pairs.tsv", sep="\t", keep_default_na=False)
    assert len(candidates) == 3


def test_sparse_ties_invariant_to_target_block_and_thread_count(fixture_run):
    config, paths, output = fixture_run
    mapping = yaml.safe_load(paths.read_text())
    target = Path(mapping["test_source2"])
    target.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        + "".join(f"S2-{i}\tAlpha Cafe\t{i} Road\tUS\n" for i in range(8))
    )
    config["sparse"]["top_k"] = 1
    config["index_block_size"] = 2
    generate(fixture_run)
    first = read_output(output)
    other_config = copy.deepcopy(config)
    other_config.update(index_block_size=5, threads=2)
    other = (other_config, paths, output.with_name("other"))
    generate(other)
    pd.testing.assert_frame_equal(first, read_output(other[2]))


def test_raw_output_guard(fixture_run):
    config, paths, _ = fixture_run
    raw = Path(yaml.safe_load(paths.read_text())["test_source1"]).parent
    with pytest.raises(ValueError, match="raw data root"):
        pc.generate(config, mode="test", run_id="x", output=raw / "output", data_paths_config=paths)


def test_post_rename_interruption_replays_uncommitted_shard(fixture_run, monkeypatch):
    original = pc._atomic_json

    def interrupted(path, state):
        if path.name == "manifest.json" and len(state["completed_shards"]) == 1:
            raise RuntimeError("post rename")
        original(path, state)

    monkeypatch.setattr(pc, "_atomic_json", interrupted)
    with pytest.raises(RuntimeError, match="post rename"):
        generate(fixture_run)
    output = fixture_run[2]
    digest = pc._sha256(output / "part-000000.parquet")
    monkeypatch.setattr(pc, "_atomic_json", original)
    generate(fixture_run, resume=True)
    assert pc._sha256(output / "part-000000.parquet") == digest
    (output / "part-999999.parquet").write_bytes(b"orphan")
    with pytest.raises(ValueError, match="unmanifested"):
        generate(fixture_run, resume=True)


def test_channel_quota_and_weight_are_rank_based():
    rows = evidence()
    selected, _ = fuse_budgeted(
        rows, queries(), {"top_k": 2, "channel_quotas": {"address": 1}}, run_id="r"
    )
    assert set(selected.target_id) == {"S3-x", "S3-a"}
    selected, _ = fuse_budgeted(
        rows, queries(), {"top_k": 2, "channel_weights": {"S3:address": 2}}, run_id="r"
    )
    assert set(selected.target_id) == {"S3-x", "S3-a"}


def test_emergency_and_rescue_production_profiles(fixture_run):
    config, paths, output = fixture_run
    config["policy"] = pc.load_profile(pc.DEFAULT_CONFIG, "emergency_k20")["policy"]
    assert generate(fixture_run)["complete"]
    config = copy.deepcopy(config)
    config["policy"] = pc.load_profile(pc.DEFAULT_CONFIG, "stretch_adaptive")["policy"]
    config["rescue"] = {"enabled": True, "top_k": 2, "threshold": 0.2, "gate": "weak_query"}
    result = generate((config, paths, output.with_name("stretch")))
    assert result["complete"]
    assert all("activated_queries" in shard["rescue"] for shard in result["completed_shards"])


def test_country_partition_and_rescue_mask_indices(fixture_run):
    config, paths, output = fixture_run
    mapping = yaml.safe_load(paths.read_text())
    query = Path(mapping["test_source1"])
    query.write_text(query.read_text() + "S1-fr\tCafe Alpha\t\tFrance\n")
    target = Path(mapping["test_source2"])
    target.write_text(target.read_text() + "S2-fr\tCafe Alpha\t\tFrance\n")
    config.update(batch_size=4, shard_size=4)
    config["rescue"] = {"enabled": True, "top_k": 2, "threshold": 0.2}
    generate(fixture_run)
    frame = read_output(output)
    assert set(frame[frame.s1_id == "S1-fr"].target_id) == {"S2-fr"}
    assert "S2-fr" not in set(frame[frame.s1_id != "S1-fr"].target_id)


def test_reuses_fitted_vectorizers_and_rejects_changed_assets(fixture_run, monkeypatch):
    config, paths, output = fixture_run
    assets = output.parent / "vectorizers"
    assets.mkdir()
    mapping = yaml.safe_load(paths.read_text())
    for source in ("S2", "S3"):
        frame = pd.read_csv(mapping[f"test_source{source[-1]}"], sep="\t", keep_default_na=False)
        for field in ("name", "address"):
            spec = pc.SparseRetrievalConfig(field=field, **config["sparse"])
            vocabulary = pc.fit_vectorizer(frame["business_" + field].tolist(), spec)
            pc.SparseTopNRetriever(spec, vocabulary).save(
                assets / f"{source}_{field}_vectorizer.joblib"
            )
    config["vectorizer_root"] = str(assets)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("reused vocabulary must not refit")

    monkeypatch.setattr(pc, "fit_vectorizer", forbidden_fit)
    state = generate(fixture_run)
    assert state["complete"]
    assert len([key for key in state["input_fingerprints"] if key.startswith("vectorizer_")]) == 4
    (assets / "S2_name_vectorizer.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="incompatible"):
        generate(fixture_run, resume=True)


def test_union_retains_complementary_low_rank_channels_and_provenance():
    query = pd.DataFrame([("S1-a", "Alpha", "Road", "US")], columns=pc.SOURCE_COLUMNS)
    evidence = pd.DataFrame(
        [
            ("S1-a", "S2-a", "S2", "name_char_tfidf", 0.9, 1, False, "r", "a"),
            ("S1-a", "S2-b", "S2", "name_char_tfidf", 0.3, 512, False, "r", "b"),
            ("S1-a", "S3-c", "S3", "missing_address_name_char", 0.2, 512, False, "r", "c"),
            ("S1-a", "S3-c", "S3", "exact_name_unicode_preserving", 1.0, 1, True, "r", "c"),
        ],
        columns=pc.EVIDENCE_COLUMNS,
    )
    policy = {"top_k": 1, "preserve_channel_union": True}
    first, _ = fuse_budgeted(evidence, query, policy, run_id="union")
    second, _ = fuse_budgeted(
        evidence.sample(frac=1, random_state=42), query, policy, run_id="union"
    )
    pd.testing.assert_frame_equal(first, second)
    assert set(first.target_id) == {"S2-a", "S2-b", "S3-c"}
    support = json.loads(first.set_index("target_id").loc["S3-c", "support_json"])
    assert {v["channel"]: v["rank"] for v in support} == {
        "exact_name_unicode_preserving": 1,
        "missing_address_name_char": 512,
    }
    assert first.set_index("target_id").loc["S3-c", "exact_match"]


def test_named_profiles_and_expanded_name_budget(fixture_run):
    for profile in ("emergency_k20", "safe_k40", "competitive_adaptive", "max_score"):
        config = pc.load_profile(pc.DEFAULT_CONFIG, profile)
        assert config["policy"]["top_k"] > 0
    assert pc.load_profile(pc.DEFAULT_CONFIG, "max_score")["policy"]["preserve_channel_union"]
    assert pc.load_profile(pc.DEFAULT_CONFIG, "competitive_adaptive")["rescue"]["enabled"]
    generate(fixture_run)
    config, _, output = fixture_run
    retriever = pc.CachedRetriever(
        output / "index", {**config, "name_top_k": 1, "address_top_k": 2}
    )
    query = pd.DataFrame([("S1-a", "Cafe Alpha", "12 Main St", "US")], columns=pc.SOURCE_COLUMNS)
    frame = pd.DataFrame(retriever.sparse_rows(query, "r"), columns=pc.EVIDENCE_COLUMNS)
    assert frame[frame.channel.eq("name_char_tfidf")].channel_rank.max() <= 1
    assert frame[frame.channel.eq("address_char_tfidf")].channel_rank.max() <= 2


def test_max_score_fixture_writes_union_and_can_resume(fixture_run):
    config, paths, output = fixture_run
    maximum = pc.load_profile(pc.DEFAULT_CONFIG, "max_score")
    config["policy"] = maximum["policy"]
    config["rescue"] = maximum["rescue"]
    config["name_top_k"] = 4
    state = generate(fixture_run)
    assert state["complete"]
    assert read_output(output).support_json.map(json.loads).map(len).max() > 1
    assert generate(fixture_run, resume=True)["candidate_rows"] == state["candidate_rows"]


def test_enabling_rescue_preserves_max_score_profile(monkeypatch):
    import sys

    seen = {}

    def capture(config, **kwargs):
        seen.update(config)
        return {"run_id": "r", "complete": True, "candidate_rows": 0, "mean": 0}

    monkeypatch.setattr(pc, "generate", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        ["production_candidates", "--mode", "test", "--profile", "max_score", "--enable-rescue"],
    )
    pc.main()
    expected = pc.load_profile(pc.DEFAULT_CONFIG, "max_score")["rescue"]
    assert seen["rescue"] == expected


def test_bounded_warm_cache_reuses_blocks_without_changing_candidates(fixture_run, monkeypatch):
    generate(fixture_run)
    config, _, output = fixture_run
    query = pd.DataFrame([("S1-a", "Cafe Alpha", "12 Main St", "US")], columns=pc.SOURCE_COLUMNS)
    cold = pc.CachedRetriever(output / "index", config)
    expected = cold.sparse_rows(query, "r")
    warm = pc.CachedRetriever(output / "index", {**config, "memory_cache_mb": 1})
    loads = []
    original = pc.sparse.load_npz

    def tracked(path):
        loads.append(path)
        return original(path)

    monkeypatch.setattr(pc.sparse, "load_npz", tracked)
    assert warm.sparse_rows(query, "r") == expected
    first_loads = len(loads)
    assert first_loads > 0
    assert warm.sparse_rows(query, "r") == expected
    assert len(loads) == first_loads
    assert 0 < warm._cached_bytes <= 2**20
    # A block larger than the configured byte allowance is used without caching.
    tiny = pc.CachedRetriever(output / "index", config)
    tiny._cache_limit = 1
    assert tiny.sparse_rows(query, "r") == expected
    assert not tiny._block_cache and tiny._cached_bytes == 0


def test_warm_cache_evicts_within_byte_budget(fixture_run):
    generate(fixture_run)
    config, _, output = fixture_run
    query = pd.DataFrame([("S1-a", "Cafe Alpha", "12 Main St", "US")], columns=pc.SOURCE_COLUMNS)
    reference = pc.CachedRetriever(output / "index", {**config, "memory_cache_mb": 1})
    expected = reference.sparse_rows(query, "r")
    largest_block = max(entry[2] for entry in reference._block_cache.values())
    assert reference._cached_bytes > largest_block
    constrained = pc.CachedRetriever(output / "index", config)
    constrained._cache_limit = largest_block
    assert constrained.sparse_rows(query, "r") == expected
    assert 0 < constrained._cached_bytes <= largest_block
    assert len(constrained._block_cache) < len(reference._block_cache)
    assert constrained.sparse_rows(query, "r") == expected
    assert constrained._cached_bytes <= largest_block


def test_fast_cli_exports_complete_submission_without_any_sparse_work(fixture_run, monkeypatch):
    import sqlite3
    import sys

    _, paths, output = fixture_run

    def forbidden(*args, **kwargs):
        pytest.fail("fast_submission constructed or invoked sparse retrieval")

    monkeypatch.setattr(pc.CachedRetriever, "__init__", forbidden)
    monkeypatch.setattr(pc.CachedRetriever, "build", forbidden)
    monkeypatch.setattr(pc.CachedRetriever, "retrieve", forbidden)
    monkeypatch.setattr(pc.CachedRetriever, "sparse_rows", forbidden)
    monkeypatch.setattr(pc.SparseTopNRetriever, "__init__", forbidden)
    monkeypatch.setattr(pc.SparseTopNRetriever, "load", forbidden)
    monkeypatch.setattr(pc, "fit_vectorizer", forbidden)
    monkeypatch.setattr(pc, "sp_matmul_topn", forbidden)
    monkeypatch.setattr(pc.sparse, "load_npz", forbidden)
    monkeypatch.setattr(pc.sparse, "save_npz", forbidden)
    monkeypatch.chdir(output.parent)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "production_candidates",
            "--mode",
            "test",
            "--profile",
            "fast_submission",
            "--data-paths",
            str(paths),
            "--output",
            str(output),
            "--vectorizer-root",
            str(output.parent / "FORBIDDEN_VECTORIZERS"),
        ],
    )
    pc.main()
    state = json.loads((output / "manifest.json").read_text())
    assert state["complete"] and state["unique_query_count"] == 3
    assert set(state["input_fingerprints"]) == {"test_source1", "test_source2", "test_source3"}
    assert {p.name for p in (output / "index").iterdir()} == {"records.sqlite", "index.json"}
    with sqlite3.connect(output / "index/records.sqlite") as db:
        indexes = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        assert {"idx_name", "idx_address"} <= indexes
        assert not {"idx_folded_name", "idx_folded_address", "idx_order"} & indexes
    frame = read_output(output)
    assert frame.exact_match.all()
    assert set(frame[frame.s1_id.eq("S1-a")].target_id) == {"S2-a", "S3-a"}
    both = frame[frame.target_id.eq("S2-b")].iloc[0]
    assert {v["channel"] for v in json.loads(both.support_json)} == {
        "exact_name_unicode_preserving",
        "exact_address_unicode_preserving",
    }
    destination = output.parent / "output/test_fast_submission"
    matches = pd.read_csv(destination / "matching_results.tsv", sep="\t", keep_default_na=False)
    candidates = pd.read_csv(destination / "candidate_pairs.tsv", sep="\t", keep_default_na=False)
    assert list(matches.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert list(candidates.columns) == ["source1_entity_id", "candidate_entity_ids"]
    assert matches.source1_entity_id.tolist() == ["S1-a", "S1-b", "S1-c"]
    assert candidates.source1_entity_id.tolist() == matches.source1_entity_id.tolist()
    assert matches.matched_entity_ids.tolist() == ["", "", "S2-b"]
    assert candidates.candidate_entity_ids.tolist() == ["S2-a,S3-a", "", "S2-b,S3-b"]


@pytest.mark.parametrize("mode", ["train", "test"])
def test_fast_profile_resume_and_batch_invariance(fixture_run, monkeypatch, mode):
    _, paths, output = fixture_run
    config = pc.load_profile(pc.DEFAULT_CONFIG, "fast_submission")
    config.update(batch_size=2, shard_size=1)
    run = (config, paths, output)
    original = pc.atomic_shard

    def interrupted(frame, path, run_id):
        if path.name == "part-000001.parquet":
            raise RuntimeError("interrupted")
        original(frame, path, run_id)

    monkeypatch.setattr(pc, "atomic_shard", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        generate(run, mode=mode)
    first_hash = pc._sha256(output / "part-000000.parquet")
    monkeypatch.setattr(pc, "atomic_shard", original)
    assert generate(run, mode=mode, resume=True)["complete"]
    assert pc._sha256(output / "part-000000.parquet") == first_hash
    other = ({**config, "batch_size": 3, "shard_size": 3}, paths, output.with_name("other"))
    generate(other, mode=mode)
    pd.testing.assert_frame_equal(read_output(output), read_output(other[2]))
    state_path = output / "manifest.json"
    state = json.loads(state_path.read_text())
    state["complete"] = False
    state_path.write_text(json.dumps(state))
    (output / "index/records.sqlite").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt exact index"):
        generate(run, mode=mode, resume=True)


def test_fast_exact_fusion_matches_existing_fusion_and_keeps_overflow():
    query = pd.DataFrame([("S1-a", "Alpha", "Road", "US")], columns=pc.SOURCE_COLUMNS)
    evidence = pd.DataFrame(
        [
            ("S1-a", "S2-a", "S2", "exact_name_unicode_preserving", 1.0, 1, True, "r", "a"),
            ("S1-a", "S2-a", "S2", "exact_address_unicode_preserving", 1.0, 1, True, "r", "a"),
            ("S1-a", "S3-b", "S3", "exact_address_unicode_preserving", 1.0, 1, True, "r", "b"),
        ],
        columns=pc.EVIDENCE_COLUMNS,
    )
    policy = {"top_k": 1}
    expected, _ = fuse_budgeted(evidence, query, policy, run_id="r")
    actual, _ = pc._fuse_exact(pd.concat([evidence, evidence]), query, policy, "r")
    pd.testing.assert_frame_equal(expected, actual)
    assert len(actual) == 2


def test_fast_rejects_rescue_and_preserves_existing_profile_depth(fixture_run):
    _, paths, output = fixture_run
    config = pc.load_profile(pc.DEFAULT_CONFIG, "fast_submission")
    config["rescue"] = {"enabled": True, "top_k": 20, "threshold": 0.2}
    with pytest.raises(ValueError, match="exact-only backend does not accept rescue"):
        generate((config, paths, output))
    for profile, budget in (("emergency_k20", 20), ("safe_k40", 40), ("stretch_adaptive", 40)):
        old = pc.load_profile(pc.DEFAULT_CONFIG, profile)
        assert old.get("retrieval_backend", "sparse") == "sparse"
        assert old["policy"]["top_k"] == budget
        for field in ("name", "address"):
            assert old.get(f"{field}_top_k", old["sparse"]["top_k"]) == 80


def test_batched_export_rejects_ambiguous_owners_across_query_shards(fixture_run, monkeypatch):
    from entity_resolution.decode import Decoder

    _, paths, output = fixture_run
    mapping = yaml.safe_load(paths.read_text())
    source = Path(mapping["test_source1"])
    source.write_text(source.read_text() + "S1-duplicate\tBeta\t9 Road\tUS\n")
    config = pc.load_profile(pc.DEFAULT_CONFIG, "fast_submission")
    config.update(batch_size=2, shard_size=1)
    generate((config, paths, output))
    calls = []
    original = Decoder.decode

    def observe(self, frame, ids):
        calls.append(len(frame))
        return original(self, frame, ids)

    monkeypatch.setattr(Decoder, "decode", observe)
    destination = pc.export_fallback(output, output.parent / "submission", "fast")
    matches = pd.read_csv(destination / "matching_results.tsv", sep="\t", keep_default_na=False)
    assert matches.matched_entity_ids.eq("").all()
    assert calls == [2]
