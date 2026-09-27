"""Per-entity hard negatives, selected on disk independently of shard boundaries."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from entity_resolution.feature_schema import OUTPUT_SCHEMA


def tie_hash(seed, *parts):
    return hashlib.sha256(json.dumps([seed, *parts]).encode()).hexdigest()


def hardness(row):
    exact = max(row[k] for k in ("exact_match", "name_normalized_exact", "exact_name_and_address"))
    evidence = max(
        row[k]
        for k in (
            "retrieval_score",
            "fusion_score",
            "name_char_similarity",
            "combined_string_similarity",
        )
    )
    return exact, evidence


def sample(db, allowed_s1, policy, seed, table="sampled"):
    if table not in {"sampled", "inner_train", "inner_eval"}:
        raise ValueError("invalid sampling table")
    cap = int(policy["per_s1_negative_cap"])
    minimum = int(policy["minimum_negatives"])
    ratio = int(policy["negatives_per_positive"])
    maximum = policy.get("max_sampled_rows")
    if cap < 2 or minimum < 2 or ratio < 1 or (maximum is not None and maximum < 1):
        raise ValueError("sampling requires cap/minimum >=2 and positive ratio/row limit")
    db.execute(f"DROP TABLE IF EXISTS {table}")
    db.execute(
        f"CREATE TEMP TABLE {table}(s1 TEXT,target TEXT,source TEXT,PRIMARY KEY(s1,target,source))"
    )
    positives = negatives = exact_kept = 0
    by_source = {"S2": 0, "S3": 0}
    # Keep only up to cap negative payloads per entity; all positives remain on disk.
    for s1 in sorted(allowed_s1):
        pos = []
        hard = {"S2": [], "S3": []}
        for target, source, payload, label in db.execute(
            "SELECT target,source,payload,label FROM pairs WHERE s1=?", (s1,)
        ):
            if label:
                pos.append((s1, target, source))
            else:
                exact, evidence = hardness(json.loads(payload))
                hard[source].append((-exact, -evidence, tie_hash(seed, s1, target), target, source))
                if len(hard[source]) > cap * 2:
                    hard[source] = sorted(hard[source])[:cap]
        quota = min(cap, max(minimum, ratio * len(pos)))
        pools = {source: sorted(values)[:cap] for source, values in hard.items()}
        chosen = [values.pop(0) for values in pools.values() if values]
        remaining = sorted(pools["S2"] + pools["S3"])
        chosen += remaining[: max(0, quota - len(chosen))]
        keys = pos + [(s1, item[3], item[4]) for item in chosen]
        # Never silently remove positives or no-match/source allocations under a global cap.
        if maximum is not None and positives + negatives + len(keys) > maximum:
            raise ValueError(
                "max_sampled_rows exceeded; raise limit or reduce negative policy; "
                "no positives were discarded"
            )
        db.executemany(f"INSERT INTO {table} VALUES(?,?,?)", keys)
        positives += len(pos)
        negatives += len(chosen)
        exact_kept += sum(item[0] < 0 for item in chosen)
        for item in chosen:
            by_source[item[4]] += 1
    db.commit()
    return {
        "positives": positives,
        "negatives": negatives,
        "rows": positives + negatives,
        "exact_looking_negatives": exact_kept,
        "negative_sources": by_source,
    }


def sampled_frame(db, table="sampled"):
    if table not in {"sampled", "inner_train", "inner_eval"}:
        raise ValueError("invalid sampling table")
    rows = db.execute(
        f"SELECT p.payload,p.label FROM pairs p JOIN {table} s "
        "ON p.s1=s.s1 AND p.target=s.target AND p.source=s.source "
        "ORDER BY p.s1,p.target,p.source"
    )
    payloads, labels = [], []
    for payload, label in rows:
        row = json.loads(payload)
        payloads.append([row[name] for name in OUTPUT_SCHEMA.names])
        labels.append(label)
    return pd.DataFrame(payloads, columns=OUTPUT_SCHEMA.names), labels
