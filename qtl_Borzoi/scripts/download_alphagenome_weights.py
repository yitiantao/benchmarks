#!/usr/bin/env python3
"""Download an AlphaGenome Orbax checkpoint without loading the model."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_CHECKPOINT_FILES = ("_CHECKPOINT_METADATA", "_METADATA", "manifest.ocdbt")


def validate_checkpoint(path: Path) -> None:
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (path / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Downloaded directory is not a complete Orbax checkpoint: {path}; "
            f"missing {missing}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download AlphaGenome weights only; do not initialize JAX/model"
    )
    parser.add_argument("--source", choices=("huggingface", "kaggle"), default="huggingface")
    parser.add_argument("--version", default="all_folds")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../../alphagenome_models/all_folds"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate an existing local checkpoint without network access.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if args.check_only or (output_dir.is_dir() and not args.force):
        validate_checkpoint(output_dir)
        downloaded = output_dir
    elif args.source == "huggingface":
        import huggingface_hub

        repo = f"google/alphagenome-{args.version.replace('_', '-').lower()}"
        downloaded = Path(
            huggingface_hub.snapshot_download(
                repo_id=repo,
                local_dir=output_dir,
                force_download=args.force,
            )
        )
    else:
        import kagglehub

        handle = f"google/alphagenome/jax/{args.version.lower()}"
        downloaded = Path(
            kagglehub.model_download(
                handle,
                output_dir=str(output_dir),
                force_download=args.force,
            )
        )

    validate_checkpoint(downloaded)
    total_bytes = sum(path.stat().st_size for path in downloaded.rglob("*") if path.is_file())
    print(f"checkpoint: {downloaded}")
    print(f"size: {total_bytes / 1024**3:.3f} GiB")


if __name__ == "__main__":
    main()
