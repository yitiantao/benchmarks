"""CLI: task-specific scoring on one GPU or a deterministic multi-GPU shard."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

from .data import FASTA, Reference, dataset_path, iter_variants, list_tasks
from .evaluation import (evaluate, ranking_metrics, read_predictions,
                         read_recoverable_predictions, write_metrics, write_predictions)
from .models import make_model


def task_slug(task: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", task.lower()).strip("_") or "symbol"
    return f"{slug}_{hashlib.sha256(task.encode()).hexdigest()[:8]}"


def output_folder(base: Path, version: str, task: str, model: str, limit: int) -> Path:
    suffix = "all_rows" if limit == 0 else f"first_{limit}"
    return base / version / task_slug(task) / model / suffix


def shard_path(folder: Path, index: int, count: int) -> Path:
    return folder / f"shard-{index:03d}-of-{count:03d}.csv.gz"


def _shard(key: str, count: int) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % count


def _progress_line(prefix: str, done: int, total: int, counts: dict,
                   scored_labels: dict, started: float) -> str:
    fraction = done / total if total else 1.0
    filled = int(fraction * 20)
    bar = "#" * filled + "-" * (20 - filled)
    elapsed = max(time.monotonic() - started, 0.001)
    if done:
        remaining = int(elapsed * (total - done) / done)
        eta = f"{remaining // 3600:02d}:{remaining // 60 % 60:02d}:{remaining % 60:02d}"
    else:
        eta = "--:--:--" if total else "00:00:00"
    return (f"{prefix} [{bar}] {done}/{total} ({fraction:.0%}) ETA={eta} "
            f"ok={counts['ok']} B={scored_labels['Benign']} "
            f"P={scored_labels['Pathogenic']} skip={counts['no_score'] + counts['error']}")


def score(args, dataset: Path, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    output = shard_path(folder, args.shard_index, args.num_shards)
    lock_path = output.with_name(output.name + ".lock")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is scoring this shard: {output}") from error
        return _score_locked(args, dataset, output)


def _valid_recovered_row(row: dict, variant, model, args) -> bool:
    if row.get(None) is not None:
        return False
    if any(value is None for value in row.values()):
        return False
    if (row.get("variant_key"), row.get("model"), row.get("version"), row.get("task"),
        row.get("score_name"), row.get("status")) != (
        variant.key, model.name, args.version, args.task, model.default_score, "ok"):
        return False
    if (row.get("chrom"), row.get("pos"), row.get("ref"), row.get("alt"),
        row.get("label")) != (variant.chrom, str(variant.pos), variant.ref,
                             variant.alt, str(variant.label)):
        return False
    try:
        raw, oriented = float(row["raw_score"]), float(row["score"])
        json.loads(row["extra_scores_json"] or "{}")
        return (math.isfinite(raw) and math.isfinite(oriented) and
                math.isclose(oriented, model.direction * raw, rel_tol=1e-6, abs_tol=1e-6))
    except (ValueError, TypeError, json.JSONDecodeError):
        return False


def _score_locked(args, dataset: Path, output: Path):
    reference = Reference(args.fasta)
    model = make_model(args.model, reference, device=args.device)
    if output.exists() and not args.force and not args.resume:
        raise FileExistsError(f"{output} exists; pass --resume to salvage/continue or --force to replace it")
    prefix = f"[{model.name} shard {args.shard_index + 1}/{args.num_shards}]"
    print(f"{prefix} selecting task variants...", flush=True)
    selected = [v for v in iter_variants(dataset, args.task, args.limit)
                if _shard(v.key, args.num_shards) == args.shard_index]
    selected_pathogenic = sum(v.label for v in selected)
    selected_benign = len(selected) - selected_pathogenic
    print(f"{prefix} selected {len(selected)} variants "
          f"(Benign={selected_benign}, Pathogenic={selected_pathogenic})", flush=True)
    recovered = {}
    read_error = None
    rejected = 0
    if args.resume and output.exists():
        possible_rows, read_error = read_recoverable_predictions(output)
        expected = {variant.key: variant for variant in selected}
        for row in possible_rows:
            variant = expected.get(row.get("variant_key"))
            if variant is not None and _valid_recovered_row(row, variant, model, args):
                recovered[variant.key] = row
            else:
                rejected += 1
        print(f"{prefix} recovered {len(recovered)}/{len(selected)} valid scored rows; "
              f"rejected={rejected}; read_error={read_error or 'none'}", flush=True)
        if len(recovered) == len(selected) and not read_error and not rejected:
            print(f"{prefix} complete shard already exists; no inference needed", flush=True)
            return
        backup = output.with_name(output.name + f".backup-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}")
        shutil.copy2(output, backup)
        print(f"{prefix} preserved original shard at {backup}", flush=True)
    remaining = len(selected) - len(recovered)
    if remaining:
        print(f"{prefix} loading model...", flush=True)
        model.load()  # fail early, rather than recording a model-load failure per variant
        print(f"{prefix} model ready; scoring...", flush=True)
    counts = {"ok": 0, "no_score": 0, "error": 0}
    scored_labels = {"Benign": 0, "Pathogenic": 0}
    def rows():
        if args.num_shards == 1:
            from tqdm.auto import tqdm
            progress = tqdm(total=remaining,
                            desc=f"{model.name} {args.task} new",
                            leave=True, mininterval=1.0, unit="var", dynamic_ncols=True)
        else:
            progress = None
        started = last_update = time.monotonic()
        interval = max(1, math.ceil(remaining / 20))
        new_done = 0
        if args.num_shards > 1:
            print(_progress_line(prefix, 0, remaining, counts, scored_labels, started), flush=True)
        for variant in selected:
            scores = {}
            error = ""
            old = recovered.get(variant.key)
            if old is not None:
                status = "ok"
                raw, oriented = old["raw_score"], old["score"]
                scores = json.loads(old["extra_scores_json"]) if old["extra_scores_json"] else {}
            else:
                try:
                    scores = model.score(variant)
                    raw = scores.get(model.default_score)
                    if raw is None or not math.isfinite(float(raw)):
                        status = "no_score"
                        raw = ""
                        oriented = ""
                    else:
                        status = "ok"
                        oriented = model.direction * float(raw)
                except Exception as exc:
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"[:500]
                    raw = oriented = ""
            counts[status] += 1
            if status == "ok":
                scored_labels["Pathogenic" if variant.label else "Benign"] += 1
            if old is None:
                new_done += 1
            if progress is not None:
                if old is None:
                    progress.update(1)
                progress.set_postfix_str(
                    f"ok={counts['ok']} B={scored_labels['Benign']} "
                    f"P={scored_labels['Pathogenic']} skip={counts['no_score'] + counts['error']}",
                    refresh=False)
            elif old is None:
                now = time.monotonic()
                if new_done == 1 or new_done == remaining or new_done % interval == 0 or now - last_update >= 30:
                    print(_progress_line(prefix, new_done, remaining, counts,
                                         scored_labels, started), flush=True)
                    last_update = now
            yield dict(model=model.name, version=args.version, task=args.task,
                       variant_key=variant.key, chrom=variant.chrom, pos=variant.pos,
                       ref=variant.ref, alt=variant.alt, label=variant.label,
                       stars=variant.stars, score_name=model.default_score,
                       raw_score=raw, score=oriented, status=status, error=error,
                       extra_scores_json=json.dumps(scores, sort_keys=True))
        if progress is not None:
            progress.close()
    write_predictions(output, rows())
    print(f"{prefix} finished: selected={len(selected)} "
          f"(Benign={selected_benign}, Pathogenic={selected_pathogenic}); "
          f"reused={len(recovered)}; newly_attempted={remaining}; "
          f"scored={counts['ok']} (Benign={scored_labels['Benign']}, "
          f"Pathogenic={scored_labels['Pathogenic']}); "
          f"no_score={counts['no_score']} error={counts['error']}", flush=True)
    print(output, flush=True)
    if counts["error"] and not counts["ok"]:
        raise RuntimeError(f"Every scored variant failed; see {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("score", "evaluate", "compare", "tasks"))
    parser.add_argument("--model", choices=("borzoi", "ntv3", "alphagenome"), default="ntv3")
    parser.add_argument("--models", default="borzoi,ntv3,alphagenome",
                        help="comma-separated models for paired compare")
    parser.add_argument("--version", choices=("2025-03", "2026-02", "2026-09-14"), default="2026-02")
    parser.add_argument("--task", default="all", help="ClinVar group name, e.g. missense, splice, 5'UTR")
    parser.add_argument("--limit", type=int, default=0, help="first N task variants; 0 means all")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fasta", type=Path, default=FASTA)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--force", action="store_true")
    restart.add_argument("--resume", action="store_true",
                         help="salvage valid rows from an existing incomplete/corrupt shard")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid --limit or shard arguments")
    dataset = dataset_path(args.version)
    if args.stage == "tasks":
        print("\n".join(list_tasks(dataset)))
        return
    if args.task not in list_tasks(dataset):
        parser.error(f"unknown task {args.task!r}; use stage 'tasks' to list available tasks")
    folder = output_folder(args.output_dir, args.version, args.task, args.model, args.limit)
    if args.stage == "score":
        score(args, dataset, folder)
    elif args.stage == "evaluate":
        paths = [shard_path(folder, i, args.num_shards) for i in range(args.num_shards)]
        absent = [p for p in paths if not p.is_file()]
        if absent:
            raise FileNotFoundError(f"Missing {len(absent)} shards, first: {absent[0]}")
        metrics = evaluate(paths, dataset, args.version, args.task,
                           {"borzoi": "borzoi_replicate_0", "ntv3": "ntv3_100m_post",
                            "alphagenome": "alphagenome_all_folds"}[args.model],
                           args.limit, args.num_shards)
        path = folder / "metrics.json"
        write_metrics(path, metrics)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(path)
    else:
        names = [name.strip() for name in args.models.split(",") if name.strip()]
        if not names or len(names) != len(set(names)) or set(names) - {"borzoi", "ntv3", "alphagenome"}:
            parser.error("--models must be a nonempty, unique subset of borzoi,ntv3,alphagenome")
        expected = {v.key: v.label for v in iter_variants(dataset, args.task, args.limit)}
        by_model = {}
        for name in names:
            model_folder = output_folder(args.output_dir, args.version, args.task, name, args.limit)
            paths = [shard_path(model_folder, i, args.num_shards) for i in range(args.num_shards)]
            if any(not path.is_file() for path in paths):
                raise FileNotFoundError(f"Missing prediction shard in {model_folder}")
            rows = {}
            seen = set()
            for row in read_predictions(paths):
                key = row["variant_key"]
                expected_model = {"borzoi": "borzoi_replicate_0", "ntv3": "ntv3_100m_post",
                                  "alphagenome": "alphagenome_all_folds"}[name]
                if (row["version"], row["task"], row["model"]) != (args.version, args.task, expected_model):
                    raise ValueError(f"Mixed run metadata in {name}: {key}")
                if key not in expected or int(row["label"]) != expected[key]:
                    raise ValueError(f"Unexpected key or label in {name}: {key}")
                if key in seen:
                    raise ValueError(f"Duplicate key in {name}: {key}")
                seen.add(key)
                if row["status"] == "ok" and row["score"] and math.isfinite(float(row["score"])):
                    rows[key] = float(row["score"])
            if seen != set(expected):
                raise ValueError(f"Incomplete predictions in {name}: {len(set(expected) - seen)} missing")
            by_model[name] = rows
        common = set(expected).intersection(*(set(rows) for rows in by_model.values()))
        report = {"schema_version": 1, "version": args.version, "task": args.task,
                  "models": names, "n_expected": len(expected), "n_common": len(common),
                  "metrics": {name: ranking_metrics((rows[key], expected[key]) for key in common)
                              for name, rows in by_model.items()}}
        path = args.output_dir / args.version / task_slug(args.task) / (
            "all_rows" if args.limit == 0 else f"first_{args.limit}") / "paired_comparison.json"
        write_metrics(path, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(path)


if __name__ == "__main__":
    main()
