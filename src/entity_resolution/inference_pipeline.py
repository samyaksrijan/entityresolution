"""Bounded feature scoring and validated submission without training-data access."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from entity_resolution.catboost_ranker import predict
from entity_resolution.config import DEFAULT_CONFIG, load_data_paths
from entity_resolution.decode import Decoder
from entity_resolution.model_artifact import load_artifact
from entity_resolution.submission import write_submission
from entity_resolution.training_pipeline import reduced_scores
from entity_resolution.training_schema import (
    atomic_json,
    feature_inputs,
    frames,
    open_store,
    protect_output,
    sha256,
    source_ids,
    stage_features,
)


class DiskCandidates(Mapping):
    """The submission API's mapping contract, with only one S1 list in memory."""

    def __init__(self, db, ids):
        self.db, self.ids = db, ids

    def __iter__(self):
        return iter(sorted(self.ids))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, key):
        if key not in self.ids:
            raise KeyError(key)
        return [
            row[0]
            for row in self.db.execute(
                "SELECT target FROM pairs WHERE s1=? ORDER BY target", (key,)
            )
        ]


def infer(
    features,
    model_path,
    output,
    run_id,
    data_paths_config=DEFAULT_CONFIG,
    *,
    batch_size=2048,
    max_decode_rows=500000,
):
    features, model_path, output = Path(features), Path(model_path), Path(output)
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run-id must be a single directory name")
    if min(batch_size, max_decode_rows) < 1:
        raise ValueError("batch and decode limits must be positive")
    paths = load_data_paths(data_paths_config)
    protect_output(output, paths, [features, model_path])
    destination = output / run_id
    if destination.exists():
        raise ValueError(f"submission run already exists: {destination}")
    model, metadata = load_artifact(model_path)
    files, fingerprints = feature_inputs(features, "test")
    ids = source_ids(paths, "test")  # Only test TSVs; never read training data or labels.
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{run_id}-", dir=output) as temporary:
        work = Path(temporary)
        db = open_store(work / "scores.sqlite")
        try:
            stage_features(db, files, ids, batch_size)
            scored_dir = work / "scored"
            scored_dir.mkdir()
            shards = []
            for number, frame in enumerate(frames(db, batch_size=batch_size)):
                scores = predict(model, frame)
                scored = frame[["s1_id", "target_id", "target_source"]].copy()
                scored["score"] = scores
                name = f"part-{number:06d}.parquet"
                path = scored_dir / name
                tmp = path.with_suffix(".tmp")
                pq.write_table(
                    pa.Table.from_pandas(scored, preserve_index=False), tmp, compression="zstd"
                )
                os.replace(tmp, path)
                shards.append({"file": name, "rows": len(scored), "sha256": sha256(path)})
                db.executemany(
                    "UPDATE pairs SET score=? WHERE s1=? AND target=? AND source=?",
                    [
                        (float(p), s, t, source)
                        for (s, t, source), p in zip(
                            scored.iloc[:, :3].itertuples(index=False, name=None),
                            scores,
                            strict=True,
                        )
                    ],
                )
                db.commit()
            atomic_json(
                scored_dir / "manifest.json",
                {"complete": True, "shards": shards, "inputs": fingerprints},
            )
            reduced = reduced_scores(db, metadata["decoder"], max_decode_rows)
            matches = Decoder(**metadata["decoder"]).decode(reduced, ids["S1"])
            candidates = DiskCandidates(db, ids["S1"])
            run = write_submission(
                ids["S1"], ids["S2"] | ids["S3"], candidates, matches, work, run_id
            )
            os.rename(scored_dir, run / "scored")
            atomic_json(
                run / "manifest.json",
                {
                    "complete": True,
                    "model_metadata_sha256": sha256(model_path / "metadata.json"),
                    "inputs": fingerprints,
                    "decoder": metadata["decoder"],
                    "s1_count": len(ids["S1"]),
                    "pair_count": sum(map(len, matches.values())),
                    "submission_files": {p.name: sha256(p) for p in sorted(run.glob("*.tsv"))},
                },
            )
            if destination.exists():
                raise ValueError("submission run already exists")
            os.rename(run, destination)
        finally:
            db.close()
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-paths", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-decode-rows", type=int, default=500000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        path = infer(
            args.features,
            args.model,
            args.output,
            args.run_id,
            args.data_paths,
            batch_size=args.batch_size,
            max_decode_rows=args.max_decode_rows,
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"ranker inference failed: {exc}\n")
    print(json.dumps({"submission": str(path), "complete": True}))


if __name__ == "__main__":
    main()
