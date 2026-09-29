#!/usr/bin/env python3
"""Run an NTv3 checkpoint on any or all unified QTL tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .data import GTExEQTLDataModule, MatchedQTLDataModule
from .evaluation import GTExEQTLEvaluator, MatchedQTLEvaluator
from .models.ntv3 import NTv3Model, NTv3ModelAdapter
from .runner import BenchmarkRunner


TASKS = ("eqtl", "sqtl", "paqtl", "ipaqtl")


def _tasks(value: str, parser: argparse.ArgumentParser) -> tuple[str, ...]:
    tasks = TASKS if value == "all" else tuple(
        item.strip() for item in value.split(",") if item.strip()
    )
    unknown = set(tasks) - set(TASKS)
    if unknown:
        parser.error(f"unknown tasks: {sorted(unknown)}")
    if not tasks:
        parser.error("--tasks cannot be empty")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="eqtl")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--model-name")
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--device", help="Override config device, e.g. cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ntv3_qtl"))
    parser.add_argument("--max-variants", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--predict-only", action="store_true")
    stage.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    if args.max_variants < 0:
        parser.error("--max-variants must be 0 or greater")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, num-shards)")
    if args.force and args.num_shards > 1:
        parser.error("--force cannot be used by parallel prediction shards")
    tasks = _tasks(args.tasks, parser)

    model_config = json.loads(args.model_config.read_text())
    if args.model_id:
        model_config["model_id"] = args.model_id
    if args.model_name:
        model_config["name"] = args.model_name
    if args.sequence_length is not None:
        model_config["sequence_length"] = args.sequence_length
    if args.device is not None:
        model_config["device"] = args.device
    model = NTv3Model(**model_config)
    adapter = NTv3ModelAdapter(
        model, model_config=model_config, batch_size=args.batch_size
    )
    print(
        f"[run] model={model.name} model_id={model.model_id} "
        f"type={model.checkpoint_type} tasks={','.join(tasks)} "
        f"max_variants={args.max_variants} batch_size={args.batch_size} "
        f"data={args.data_dir.resolve()} output={args.output_dir.resolve()}",
        flush=True,
    )

    limit_tag = f"max_{args.max_variants}" if args.max_variants else "all"
    for task in tasks:
        if task == "eqtl":
            data = GTExEQTLDataModule(
                data_dir=args.data_dir / "eqtl",
                all_tissues=True,
                max_variants=args.max_variants,
                name=f"gtex-eqtl-{limit_tag}",
            )
            evaluator = GTExEQTLEvaluator()
        else:
            data = MatchedQTLDataModule(
                task=task,
                data_dir=args.data_dir,
                max_pairs=args.max_variants,
                name=f"{task}-{limit_tag}",
            )
            evaluator = MatchedQTLEvaluator()
        print(f"===== NTv3 {task} ({limit_tag}) =====", flush=True)
        started = time.perf_counter()
        metrics = BenchmarkRunner(
            data=data,
            model=adapter,
            evaluator=evaluator,
            output_dir=args.output_dir / limit_tag,
            options={
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
            },
        ).run(
            predict=not args.evaluate_only,
            evaluate=not args.predict_only,
            force=args.force,
        )
        if metrics is not None:
            print(metrics.to_string(index=False), flush=True)
        print(
            f"[performance] task={task} elapsed_seconds="
            f"{time.perf_counter() - started:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
