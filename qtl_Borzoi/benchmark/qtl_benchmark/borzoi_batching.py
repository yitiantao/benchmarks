"""Batch REF/ALT sequence pairs for Borzoi inference."""

from collections.abc import Callable, Iterator, Sequence
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np


INT32_MAX = 2**31 - 1


def backfill_missing_options(options: Any, parsed_defaults: Any) -> Any:
    """Add newly introduced CLI defaults to an older pickled optparse Values."""
    for name, value in vars(parsed_defaults).items():
        if not hasattr(options, name):
            setattr(options, name, value)
    return options


def sequence_batch_size(variant_batch_size: int) -> int:
    """Return the number of sequences in a batch of REF/ALT variant pairs."""
    if isinstance(variant_batch_size, bool) or variant_batch_size < 1:
        raise ValueError("variant_batch_size must be a positive integer")
    return 2 * variant_batch_size


def borzoi_batch_geometry(
    params_model: dict[str, Any], variant_batch_size: int
) -> tuple[int, int, int]:
    """Return first-convolution work elements and the safe paired-batch limit."""
    sequence_count = sequence_batch_size(variant_batch_size)
    seq_length = int(params_model["seq_length"])
    conv_dna = next(
        (block for block in params_model["trunk"] if block.get("name") == "conv_dna"),
        None,
    )
    if conv_dna is None or "filters" not in conv_dna:
        raise ValueError("Borzoi model parameters do not define conv_dna filters")
    filters = int(conv_dna["filters"])
    elements_per_sequence = seq_length * filters
    work_elements = sequence_count * elements_per_sequence
    max_variant_batch_size = (INT32_MAX // elements_per_sequence) // 2
    return work_elements, sequence_count, max_variant_batch_size


def validate_borzoi_variant_batch(
    params_model: dict[str, Any], variant_batch_size: int
) -> int:
    """Reject batches that overflow TensorFlow GPU kernel launch counters."""
    work_elements, sequence_count, max_variant_batch_size = borzoi_batch_geometry(
        params_model, variant_batch_size
    )
    if work_elements > INT32_MAX:
        raise ValueError(
            f"--variant-batch-size {variant_batch_size} creates {sequence_count} "
            f"REF/ALT sequences and {work_elements:,} elements in Borzoi's first "
            f"convolution, exceeding TensorFlow's int32 GPU launch limit "
            f"({INT32_MAX:,}); maximum safe value for this model is "
            f"{max_variant_batch_size} (1 or 2 recommended)"
        )
    return max_variant_batch_size


def prediction_stream_size(variant_batch_size: int, minimum: int = 32) -> int:
    """Choose a stream buffer containing a whole number of inference batches."""
    batch_size = sequence_batch_size(variant_batch_size)
    batches = max(1, (minimum + batch_size - 1) // batch_size)
    return batches * batch_size


def iter_ref_alt_predictions(
    model: Callable[[np.ndarray], Any],
    variants: Sequence[Any],
    make_pair: Callable[[Any], Sequence[np.ndarray]],
    variant_batch_size: int,
) -> Iterator[tuple[int, Any, Any]]:
    """Yield ordered ``(variant index, REF prediction, ALT prediction)`` tuples.

    ``make_pair`` must return exactly two one-hot sequences in REF, ALT order.
    The final model call may contain fewer variants than the requested batch.
    """
    for _, _, predictions, _ in iter_ref_alt_prediction_batches(
        model, variants, make_pair, variant_batch_size
    ):
        yield from predictions


def iter_ref_alt_prediction_batches(
    model: Callable[[np.ndarray], Any],
    variants: Sequence[Any],
    make_pair: Callable[[Any], Sequence[np.ndarray]],
    variant_batch_size: int,
) -> Iterator[tuple[int, int, list[tuple[int, Any, Any]], float]]:
    """Yield ordered REF/ALT predictions grouped by model-call batch.

    Each result is ``(start, stop, predictions, inference_elapsed)``.  The
    elapsed value includes sequence construction and the model call; callers
    can add their post-processing time before printing batch progress.
    """
    variant_batch_size = int(variant_batch_size)
    sequence_batch_size(variant_batch_size)

    for start in range(0, len(variants), variant_batch_size):
        batch_started = time.perf_counter()
        stop = min(start + variant_batch_size, len(variants))
        sequences = []
        for variant in variants[start:stop]:
            pair = make_pair(variant)
            if len(pair) != 2:
                raise ValueError("Borzoi inference requires one REF and one ALT sequence")
            sequences.extend(pair)

        model_predictions = model(np.asarray(sequences))
        expected = 2 * (stop - start)
        if len(model_predictions) != expected:
            raise ValueError(
                f"Borzoi returned {len(model_predictions)} predictions for "
                f"{expected} sequences"
            )

        predictions = [
            (
                start + offset,
                model_predictions[2 * offset],
                model_predictions[2 * offset + 1],
            )
            for offset in range(stop - start)
        ]
        yield start, stop, predictions, time.perf_counter() - batch_started


def prediction_progress_label(
    model_file: str | None = None,
    vcf_file: str | None = None,
    worker_index: int | None = None,
) -> str:
    """Build an unambiguous model/replicate/split/shard progress label."""
    parts = [os.environ.get("BORZOI_MODEL_NAME", "borzoi")]

    if model_file:
        replicate_match = re.search(r"(?:^|/)(f\d+c\d+)(?:/|$)", str(model_file))
        if replicate_match:
            parts.append(replicate_match.group(1))

    if vcf_file:
        vcf_name = Path(vcf_file).name.lower()
        split_match = re.search(r"(?:^|[_.-])(pos|neg)(?:[_.-]|$)", vcf_name)
        if split_match:
            parts.append(split_match.group(1))

    if worker_index is not None:
        parts.append(f"job{worker_index}")
    return "/".join(parts)


def log_prediction_progress(
    label: str,
    completed: int,
    total: int,
    rows: int,
    batch_elapsed: float,
) -> None:
    """Print one AlphaGenome-compatible progress record for a variant batch."""
    print(
        f"[predict/{label}] {completed}/{total} rows={rows} "
        f"batch_elapsed={batch_elapsed:.1f}s",
        flush=True,
    )
