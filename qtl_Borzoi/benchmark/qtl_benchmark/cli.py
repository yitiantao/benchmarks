#!/usr/bin/env python3
"""Unified command line: data module -> model adapter -> evaluator."""

from __future__ import annotations

import argparse

from .config import build_runner, load_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Benchmark JSON configuration")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--predict-only", action="store_true")
    stage.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Discard cached predictions")
    args = parser.parse_args()

    runner = build_runner(load_json(args.config))
    metrics = runner.run(
        predict=not args.evaluate_only,
        evaluate=not args.predict_only,
        force=args.force,
    )
    if metrics is not None:
        print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
