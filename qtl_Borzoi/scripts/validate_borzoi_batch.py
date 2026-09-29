#!/usr/bin/env python
"""Fail fast when a Borzoi inference batch exceeds TensorFlow GPU limits."""

import argparse
import json
from pathlib import Path

from qtl_benchmark.borzoi_batching import (
    borzoi_batch_geometry,
    validate_borzoi_variant_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("params", type=Path)
    parser.add_argument("variant_batch_size", type=int)
    args = parser.parse_args()

    with args.params.open() as handle:
        params_model = json.load(handle)["model"]
    try:
        max_safe = validate_borzoi_variant_batch(
            params_model, args.variant_batch_size
        )
    except ValueError as error:
        parser.error(str(error))
    work_elements, sequence_count, _ = borzoi_batch_geometry(
        params_model, args.variant_batch_size
    )
    print(
        f"[batch-check] variants={args.variant_batch_size} "
        f"sequences={sequence_count} first_conv_elements={work_elements:,} "
        f"maximum_safe_variants={max_safe}"
    )


if __name__ == "__main__":
    main()
