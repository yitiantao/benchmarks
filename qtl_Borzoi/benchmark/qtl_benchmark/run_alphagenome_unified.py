#!/usr/bin/env python3
"""Run AlphaGenome on any or all QTL tasks through the unified framework."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .data import GTExEQTLDataModule, MatchedQTLDataModule
from .evaluation import GTExEQTLEvaluator, MatchedQTLEvaluator
from .interfaces import TidyVariantModelAdapter
from .models.alphagenome import AlphaGenomeModel
from .runner import BenchmarkRunner


TASKS = ("eqtl", "sqtl", "paqtl", "ipaqtl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="eqtl")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/alphagenome_qtl"))
    parser.add_argument("--max-variants", type=int, default=0)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument(
        "--variant-batch-size",
        type=int,
        default=1,
        help=(
            "Number of variants concurrently scored by each local AlphaGenome "
            "worker process (default: 1)"
        ),
    )
    parser.add_argument("--force", action="store_true")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--predict-only", action="store_true")
    stage.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.max_variants < 0:
        parser.error("--max-variants must be 0 or greater")
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if args.variant_batch_size < 1:
        parser.error("--variant-batch-size must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, num-shards)")
    if args.force and args.num_shards > 1:
        parser.error("--force cannot be used by parallel prediction shards")
    tasks = TASKS if args.tasks == "all" else tuple(args.tasks.split(","))
    unknown = set(tasks) - set(TASKS)
    if unknown:
        parser.error(f"unknown tasks: {sorted(unknown)}")
    print(
        f"[run] tasks={','.join(tasks)} max_variants={args.max_variants} "
        f"variant_batch_size={args.variant_batch_size} "
        f"data={args.data_dir.resolve()} output={args.output_dir.resolve()}",
        flush=True,
    )

    model_config = json.loads(args.model_config.read_text())
    if args.checkpoint_path is not None:
        model_config["checkpoint_path"] = str(args.checkpoint_path.resolve())
    if args.model_name is not None:
        model_config["name"] = args.model_name
    model = AlphaGenomeModel(**model_config)
    print(
        f"[model] name={model.name} checkpoint={model.checkpoint_path}",
        flush=True,
    )
    adapter = TidyVariantModelAdapter(
        model,
        model_class="qtl_benchmark.models.alphagenome:AlphaGenomeModel",
        model_config=model_config,
        batch_size=args.variant_batch_size,
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
        print(f"===== AlphaGenome {task} ({limit_tag}) =====", flush=True)
        started = time.perf_counter()
        metrics = BenchmarkRunner(
            data=data,
            model=adapter,
            evaluator=evaluator,
            output_dir=args.output_dir / limit_tag,
            options={
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "variant_batch_size": args.variant_batch_size,
            },
        ).run(
            predict=not args.evaluate_only,
            evaluate=not args.predict_only,
            force=args.force,
        )
        if metrics is not None:
            print(metrics.to_string(index=False), flush=True)
        dataset_name = f"gtex-eqtl-{limit_tag}" if task == "eqtl" else f"{task}-{limit_tag}"
        if metrics is not None:
            print(
                f"[result] metrics="
                f"{(args.output_dir / limit_tag / model.name / dataset_name / 'metrics.tsv').resolve()}",
                flush=True,
            )
        print(
            f"[performance] task={task} elapsed_seconds="
            f"{time.perf_counter() - started:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
