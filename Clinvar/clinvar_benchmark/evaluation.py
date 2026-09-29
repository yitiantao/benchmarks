"""Common prediction table and binary-ranking evaluation, independent of model code."""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
from pathlib import Path
import tempfile
import zlib

from .data import iter_variants


FIELDS = ("model", "version", "task", "variant_key", "chrom", "pos", "ref", "alt",
          "label", "stars", "score_name", "raw_score", "score", "status", "error",
          "extra_scores_json")


def _open(path: Path, mode: str):
    return gzip.open(path, mode, newline="") if str(path).endswith(".gz") else open(path, mode, newline="")


def write_predictions(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique staging names are essential when duplicate jobs target the same
    # output: a fixed '.tmp' path lets their gzip streams corrupt each other.
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.fchmod(descriptor, 0o644)
    os.close(descriptor)
    temporary = Path(temporary_name)
    opener = gzip.open if str(path).endswith(".gz") else open
    try:
        with opener(temporary, "wt", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_predictions(paths):
    for path in paths:
        with _open(path, "rt") as handle:
            reader = csv.DictReader(handle)
            missing = set(FIELDS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} missing columns {sorted(missing)}")
            yield from reader


def read_recoverable_predictions(path: Path):
    """Read the intact prefix of a damaged gzip/CSV shard for --resume."""
    rows = []
    try:
        with _open(path, "rt") as handle:
            reader = csv.DictReader(handle)
            missing = set(FIELDS) - set(reader.fieldnames or ())
            if missing:
                return rows, f"missing columns: {sorted(missing)}"
            for row in reader:
                rows.append(row)
    except (OSError, EOFError, UnicodeError, csv.Error, zlib.error) as error:
        return rows, f"{type(error).__name__}: {error}"
    return rows, None


def ranking_metrics(pairs):
    """AUROC and AP with deterministic, tie-aware grouped thresholds."""
    pairs = [(float(s), int(y)) for s, y in pairs if math.isfinite(float(s))]
    positive = sum(y for _, y in pairs)
    negative = len(pairs) - positive
    if not positive or not negative:
        return {"auroc": None, "auprc": None, "n_benign": negative, "n_pathogenic": positive}
    pairs.sort(reverse=True)
    tp = fp = 0
    area_roc = area_pr = 0.0
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        new_tp = new_fp = 0
        while index < len(pairs) and pairs[index][0] == score:
            new_tp += pairs[index][1]
            new_fp += 1 - pairs[index][1]
            index += 1
        area_roc += (new_fp / negative) * ((tp + new_tp / 2) / positive)
        tp += new_tp
        fp += new_fp
        area_pr += (new_tp / positive) * (tp / (tp + fp))
    return {"auroc": area_roc, "auprc": area_pr,
            "n_benign": negative, "n_pathogenic": positive}


def evaluate(paths, dataset: Path, version: str, task: str, model: str,
             limit: int, n_shards: int):
    expected = {variant.key: variant for variant in iter_variants(dataset, task, limit)}
    records = {}
    score_name = None
    for row in read_predictions(paths):
        if (row["version"], row["task"], row["model"]) != (version, task, model):
            raise ValueError(f"Mixed run metadata at {row['variant_key']}")
        key = row["variant_key"]
        if key in records:
            raise ValueError(f"Duplicate scored variant {key}")
        if key not in expected or int(row["label"]) != expected[key].label:
            raise ValueError(f"Unexpected variant or label: {key}")
        if score_name is not None and row["score_name"] != score_name:
            raise ValueError("Mixed score types")
        score_name = row["score_name"]
        records[key] = row
    missing = set(expected) - set(records)
    if missing:
        raise ValueError(f"Predictions incomplete: {len(missing)} missing, e.g. {next(iter(missing))}")
    scored = [(float(row["score"]), int(row["label"])) for row in records.values()
              if row["status"] == "ok" and row["score"]]
    expected_pathogenic = sum(v.label for v in expected.values())
    metrics = ranking_metrics(scored)
    counts = {}
    for row in records.values():
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {
        "schema_version": 1, "model": model, "version": version, "task": task,
        "score_name": score_name, "n_shards": n_shards, "n_expected": len(expected),
        "n_expected_benign": len(expected) - expected_pathogenic,
        "n_expected_pathogenic": expected_pathogenic,
        "n_scored": len(scored), "coverage": len(scored) / len(expected) if expected else None,
        "status_counts": counts, **metrics,
        "note": "AUROC/AUPRC use scored variants only; compare models on their common keys for strict paired analysis.",
    }


def write_metrics(path: Path, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.fchmod(descriptor, 0o644)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
