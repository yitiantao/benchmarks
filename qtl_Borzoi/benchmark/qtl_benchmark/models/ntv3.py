"""NTv3 adapter for the unified QTL benchmark.

NTv3 does not publish task-specific QTL scorers.  This adapter therefore
exposes two explicit, reproducible scoring modes:

* post-trained checkpoints: ALT minus REF changes in functional/annotation
  heads selected for each QTL task;
* pre-trained checkpoints: masked-base log-odds, log P(ALT) - log P(REF).

Heavy PyTorch/Transformers imports are intentionally delayed until setup so
the benchmark CLI and unit tests remain usable in the Borzoi environment.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..core import BenchmarkContext, InferenceDataset
from ..interfaces import ModelAdapter, TidyVariantModelAdapter
from ..model_base import QTLModel, TaskContext, VariantRecord
from ..predictions import PredictionStore
from .borzoi import TISSUE_KEYWORDS


DEFAULT_POST_HEADS: Mapping[str, tuple[str, ...]] = {
    # Legacy fallback for custom configs without eQTL track metadata. The
    # repository's pinned post-training configs use broad GTEx tissue mapping.
    "eqtl": ("ENCSR056HPM", "ENCSR561FEE_P"),
    "sqtl": ("splice_donor", "splice_acceptor"),
    "paqtl": ("polyA_signal",),
    "ipaqtl": ("polyA_signal", "3UTR+", "3UTR-"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_track_id(
    identifier: str, file_path: str, available: frozenset[str]
) -> str | None:
    """Resolve Borzoi source metadata rows to official NTv3 track IDs."""
    if identifier in available:
        return identifier
    match = re.search(r"/(ENCSR[A-Z0-9]+)/", file_path)
    if match is None:
        return None
    experiment = match.group(1)
    if identifier.endswith("+"):
        candidate = f"{experiment}_P"
    elif identifier.endswith("-"):
        candidate = f"{experiment}_M"
    else:
        candidate = experiment
    return candidate if candidate in available else None


def _select_broad_eqtl_tracks(
    metadata_path: Path,
    available_tracks: Sequence[str],
    tissue_keywords: Mapping[str, str],
) -> tuple[
    tuple[str, ...],
    dict[str, tuple[int, ...]],
    dict[str, str],
]:
    """Select GTEx RNA tracks using the same broad tissue map as Borzoi.

    Official NTv3 track IDs use experiment-level names for ENCODE RNA tracks,
    whereas the source metadata stores file accessions. GTEx sample IDs are
    already identical. GTEx is always preferred; ENCODE is used only when a
    broad tissue has no GTEx track in the checkpoint (currently heart).
    """
    metadata = pd.read_csv(metadata_path, sep="\t")
    required = {"identifier", "file", "description"}
    missing_columns = required - set(metadata.columns)
    if missing_columns:
        raise ValueError(
            f"NTv3 track metadata is missing columns: {sorted(missing_columns)}"
        )

    available = tuple(str(track) for track in available_tracks)
    available_set = frozenset(available)
    records: list[tuple[str, str, str]] = []
    for row in metadata.itertuples(index=False):
        identifier = str(row.identifier)
        description = str(row.description).lower()
        if not description.startswith("rna:"):
            continue
        track_id = _metadata_track_id(identifier, str(row.file), available_set)
        if track_id is None:
            continue
        source = "gtex" if identifier.startswith("GTEX-") else "encode"
        records.append((track_id, description.removeprefix("rna:"), source))

    full_index = {track: index for index, track in enumerate(available)}
    full_by_tissue: dict[str, tuple[int, ...]] = {}
    source_by_tissue: dict[str, str] = {}
    selected_full_indices: list[int] = []
    for tissue, keyword in tissue_keywords.items():
        matches: list[int] = []
        selected_source = ""
        for source in ("gtex", "encode"):
            for track_id, description, record_source in records:
                matches_keyword = keyword.lower() in description
                if keyword == "blood" and "blood_vessel" in description:
                    matches_keyword = False
                if record_source == source and matches_keyword:
                    matches.append(full_index[track_id])
            if matches:
                selected_source = source
                break
        if not matches:
            raise ValueError(
                f"No NTv3 RNA tracks matched {tissue} ({keyword}) in GTEx or ENCODE"
            )
        indexes = tuple(sorted(set(matches)))
        full_by_tissue[tissue] = indexes
        source_by_tissue[tissue] = selected_source
        selected_full_indices.extend(indexes)

    selected_full_indices = sorted(set(selected_full_indices))
    remap = {full: selected for selected, full in enumerate(selected_full_indices)}
    selected_tracks = tuple(available[index] for index in selected_full_indices)
    tissue_indices = {
        tissue: tuple(remap[index] for index in indexes)
        for tissue, indexes in full_by_tissue.items()
    }
    return selected_tracks, tissue_indices, source_by_tissue


@dataclass(frozen=True)
class GeneInterval:
    chrom: str
    start: int
    end: int
    gene_id: str


class GeneIndex:
    """Small in-memory overlap index built from GENCODE gene records."""

    def __init__(self, genes: Sequence[GeneInterval]) -> None:
        by_chrom: dict[str, list[GeneInterval]] = {}
        for gene in genes:
            by_chrom.setdefault(gene.chrom, []).append(gene)
        self._genes: dict[str, tuple[GeneInterval, ...]] = {}
        self._starts: dict[str, tuple[int, ...]] = {}
        for chrom, chrom_genes in by_chrom.items():
            ordered = tuple(
                sorted(chrom_genes, key=lambda gene: (gene.start, gene.end))
            )
            self._genes[chrom] = ordered
            self._starts[chrom] = tuple(gene.start for gene in ordered)

    @classmethod
    def from_gtf(cls, path: str | Path) -> "GeneIndex":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"GENCODE GTF not found: {path}")
        gene_pattern = re.compile(r'(?:^|;\s*)gene_id\s+"([^"]+)"')
        genes: list[GeneInterval] = []
        with path.open() as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t", 8)
                if len(fields) != 9 or fields[2] != "gene":
                    continue
                match = gene_pattern.search(fields[8])
                if match is None:
                    continue
                genes.append(
                    GeneInterval(
                        chrom=fields[0],
                        start=int(fields[3]) - 1,
                        end=int(fields[4]),
                        gene_id=match.group(1).split(".", 1)[0],
                    )
                )
        if not genes:
            raise ValueError(f"No gene records found in {path}")
        return cls(genes)

    def overlapping(self, chrom: str, start: int, end: int) -> tuple[GeneInterval, ...]:
        genes = self._genes.get(chrom, ())
        starts = self._starts.get(chrom, ())
        stop = bisect_right(starts, end - 1)
        return tuple(gene for gene in genes[:stop] if gene.end > start)


@dataclass(frozen=True)
class SequenceWindow:
    sequence: str
    start: int
    variant_offset: int


def _signed_max_abs(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    finite = np.isfinite(values)
    if not finite.any():
        return 0.0
    safe = np.where(finite, values, 0.0)
    return float(safe.reshape(-1)[np.abs(safe).argmax()])


class NTv3Model(QTLModel):
    """Run a gated Hugging Face NTv3 checkpoint on human QTL variants."""

    supported_tasks = frozenset({"eqtl", "sqtl", "paqtl", "ipaqtl"})

    def __init__(
        self,
        model_id: str,
        fasta_path: str,
        gtf_path: str,
        root_dir: str = ".",
        name: str = "ntv3",
        checkpoint_type: str = "auto",
        sequence_length: int = 32768,
        device: str = "cuda",
        dtype: str = "auto",
        species: str = "human",
        revision: str | None = None,
        code_revision: str | None = None,
        cache_dir: str = ".benchmark_deps/ntv3_hf",
        local_files_only: bool = False,
        strict_reference: bool = True,
        aggregation: str = "max_abs",
        task_heads: Mapping[str, Sequence[str]] | None = None,
        eqtl_track_metadata_path: str | None = None,
        eqtl_track_metadata_sha256: str | None = None,
        tissue_keywords: Mapping[str, str] | None = None,
    ) -> None:
        self.name = str(name)
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.model_id = str(model_id)
        self.fasta_path = self._resolve(fasta_path)
        self.gtf_path = self._resolve(gtf_path)
        self.cache_dir = self._resolve(cache_dir)
        self.checkpoint_type = self._checkpoint_type(checkpoint_type, self.model_id)
        self.sequence_length = int(sequence_length)
        self.device_name = str(device)
        self.dtype_name = str(dtype)
        self.species = str(species)
        self.revision = revision
        self.code_revision = code_revision
        self.local_files_only = bool(local_files_only)
        self.strict_reference = bool(strict_reference)
        self.aggregation = str(aggregation)
        self.eqtl_track_metadata_path = (
            self._resolve(eqtl_track_metadata_path)
            if eqtl_track_metadata_path is not None
            else None
        )
        self.eqtl_track_metadata_sha256 = None
        if self.eqtl_track_metadata_path is not None:
            if not self.eqtl_track_metadata_path.is_file():
                raise FileNotFoundError(
                    "NTv3 eQTL track metadata not found: "
                    f"{self.eqtl_track_metadata_path}"
                )
            self.eqtl_track_metadata_sha256 = _sha256(self.eqtl_track_metadata_path)
            if (
                eqtl_track_metadata_sha256
                and eqtl_track_metadata_sha256 != self.eqtl_track_metadata_sha256
            ):
                raise ValueError(
                    "NTv3 eQTL track metadata SHA-256 mismatch: "
                    f"expected={eqtl_track_metadata_sha256} "
                    f"observed={self.eqtl_track_metadata_sha256}"
                )
        self.tissue_keywords = dict(TISSUE_KEYWORDS)
        self.tissue_keywords.update(tissue_keywords or {})
        self.task_heads = {
            task: tuple(heads) for task, heads in DEFAULT_POST_HEADS.items()
        }
        if task_heads:
            self.task_heads.update(
                {task: tuple(heads) for task, heads in task_heads.items()}
            )

        if self.sequence_length <= 0 or self.sequence_length % 128:
            raise ValueError("sequence_length must be positive and divisible by 128")
        if self.checkpoint_type not in {"pre", "post"}:
            raise ValueError("checkpoint_type must be auto, pre, or post")
        if self.aggregation not in {"max_abs", "mean"}:
            raise ValueError("aggregation must be max_abs or mean")
        unknown = set(self.task_heads) - self.supported_tasks
        if unknown:
            raise ValueError(f"Unknown task_heads keys: {sorted(unknown)}")

        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._genome: Any | None = None
        self._gene_index: GeneIndex | None = None
        self._device: Any | None = None
        self._dtype: Any | None = None
        self._active_task: str | None = None
        self._active_tissues: tuple[str, ...] = ()
        self._head_indices: tuple[int, ...] = ()
        self._head_names: tuple[str, ...] = ()
        self._tissue_head_indices: dict[str, tuple[int, ...]] = {}
        self._tissue_head_sources: dict[str, str] = {}
        self._fasta_alt_matches = 0
        self._forced_reference_bases = 0

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.root_dir / path
        return path.resolve()

    @staticmethod
    def _checkpoint_type(value: str, model_id: str) -> str:
        if value != "auto":
            return value
        name = model_id.rstrip("/").split("/")[-1].lower()
        if "_post" in name:
            return "post"
        if "_pre" in name:
            return "pre"
        raise ValueError(
            "Cannot infer checkpoint_type from model_id; set checkpoint_type explicitly"
        )

    def _require_assets(self) -> None:
        missing = [
            str(path)
            for path in (
                self.fasta_path,
                Path(str(self.fasta_path) + ".fai"),
                self.gtf_path,
            )
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing NTv3 benchmark assets:\n" + "\n".join(missing)
            )

    def _select_dtype(self, torch: Any, device: Any, config: Any | None = None) -> Any:
        if self.dtype_name == "float32":
            return torch.float32
        if self.dtype_name == "float16":
            return torch.float16
        if self.dtype_name == "bfloat16":
            return torch.bfloat16
        if self.dtype_name != "auto":
            raise ValueError("dtype must be auto, float32, float16, or bfloat16")

        # NTv3 checkpoints carry an explicit precision contract. In particular,
        # the released post-trained checkpoints declare float32 for every compute
        # and parameter group. Choosing BF16 merely because an Ampere+ GPU is
        # available casts LinearHead weights to BF16, while the upstream forward
        # intentionally casts its LayerNorm output to FP32. The following linear
        # operation then fails with ``float != c10::BFloat16``. Thus "auto"
        # means checkpoint dtype, not GPU-preferred dtype.
        configured = None
        if config is not None:
            configured = getattr(config, "dtype", None)
            if configured is None:
                configured = getattr(config, "torch_dtype", None)
        configured_name = str(configured).lower().removeprefix("torch.")
        configured_dtypes = {
            "float32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return configured_dtypes.get(configured_name, torch.float32)

    def setup_task(self, context: TaskContext) -> None:
        self.validate_task(context.task)
        self._require_assets()
        self._active_task = context.task
        self._active_tissues = tuple(context.options.get("tissues", ()))
        self._fasta_alt_matches = 0
        self._forced_reference_bases = 0
        if self._model is None:
            try:
                import torch
                from pyfaidx import Fasta
                from transformers import (
                    AutoConfig,
                    AutoModel,
                    AutoModelForMaskedLM,
                    AutoTokenizer,
                )
            except ImportError as error:
                raise RuntimeError(
                    "NTv3 dependencies are missing. Run scripts/setup_ntv3_benchmark.sh"
                ) from error

            if self.device_name.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(
                    "NTv3 requested CUDA, but torch.cuda.is_available() is false"
                )
            device = torch.device(self.device_name)
            token = os.getenv("HF_TOKEN") or None
            common = {
                "revision": self.revision,
                "code_revision": self.code_revision,
                "cache_dir": str(self.cache_dir),
                "local_files_only": self.local_files_only,
                "token": token,
                "trust_remote_code": True,
            }
            common = {key: value for key, value in common.items() if value is not None}
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            started = time.time()
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **common)
            config = AutoConfig.from_pretrained(self.model_id, **common)
            dtype = self._select_dtype(torch, device, config)
            model_class = (
                AutoModel if self.checkpoint_type == "post" else AutoModelForMaskedLM
            )
            self._model = model_class.from_pretrained(
                self.model_id, torch_dtype=dtype, **common
            ).to(device)
            self._model.eval()
            self._torch = torch
            self._device = device
            self._dtype = dtype
            self._genome = Fasta(
                str(self.fasta_path), as_raw=True, sequence_always_upper=True
            )
            self._gene_index = GeneIndex.from_gtf(self.gtf_path)
            self._config = config
            print(
                f"[{self.name}] loaded {self.model_id} type={self.checkpoint_type} "
                f"device={device} dtype={dtype} sequence_length={self.sequence_length} "
                f"in {time.time() - started:.1f}s",
                flush=True,
            )
        self._configure_task_heads(context.task)

    def _configure_task_heads(self, task: str) -> None:
        self._tissue_head_indices = {}
        self._tissue_head_sources = {}
        if self.checkpoint_type == "pre":
            self._head_names = ("masked_base_log_odds",)
            self._head_indices = ()
            return
        if task == "eqtl":
            available = tuple(self._config.bigwigs_per_species[self.species])
            if self.eqtl_track_metadata_path is not None:
                (
                    self._head_names,
                    self._tissue_head_indices,
                    self._tissue_head_sources,
                ) = _select_broad_eqtl_tracks(
                    self.eqtl_track_metadata_path,
                    available,
                    self.tissue_keywords,
                )
                full_index = {track: index for index, track in enumerate(available)}
                self._head_indices = tuple(
                    full_index[track] for track in self._head_names
                )
                fallback_count = sum(
                    source != "gtex" for source in self._tissue_head_sources.values()
                )
                print(
                    f"[{self.name}/eqtl] selected_rna_tracks={len(self._head_names)} "
                    f"benchmark_tissues={len(self._tissue_head_indices)} "
                    f"fallback_tissues={fallback_count} mapping=borzoi-broad",
                    flush=True,
                )
                return
        else:
            available = tuple(self._config.bed_elements_names)
        requested = self.task_heads[task]
        missing = [head for head in requested if head not in available]
        if missing:
            raise ValueError(
                f"NTv3 {task} heads absent from checkpoint: {missing}; "
                f"available count={len(available)}"
            )
        self._head_names = requested
        self._head_indices = tuple(available.index(head) for head in requested)
        print(f"[{self.name}/{task}] heads={','.join(requested)}", flush=True)

    def _window(self, variant: VariantRecord) -> SequenceWindow:
        if self._genome is None:
            raise RuntimeError("setup_task() must be called before sequence extraction")
        if len(variant.ref) != 1 or len(variant.alt) != 1:
            raise ValueError(
                f"NTv3 QTL adapter currently supports SNVs only: {variant.variant_id}"
            )
        center = variant.pos - 1
        start = center - self.sequence_length // 2
        end = start + self.sequence_length
        chrom_length = len(self._genome[variant.chrom])
        fetch_start = max(0, start)
        fetch_end = min(chrom_length, end)
        sequence = "N" * max(0, -start)
        sequence += str(self._genome[variant.chrom][fetch_start:fetch_end]).upper()
        sequence += "N" * max(0, end - chrom_length)
        if len(sequence) != self.sequence_length:
            raise RuntimeError(
                f"Reference extraction returned {len(sequence)} bp, expected {self.sequence_length}"
            )
        offset = center - start
        observed = sequence[offset : offset + len(variant.ref)]
        if observed != variant.ref.upper():
            if observed == variant.alt.upper():
                # The published QTL VCFs contain a small number of rows whose
                # REF/ALT orientation is reversed relative to hg38. Borzoi's
                # snp_seq1 handles these by replacing the genomic ALT base
                # with the VCF REF base before constructing the ALT sequence.
                # Mirror that behavior so the score remains ALT-REF in VCF
                # orientation and effect-allele flipping stays correct.
                self._fasta_alt_matches += 1
                print(
                    f"[reference/orientation] {variant.variant_id} FASTA={observed} "
                    f"matches VCF ALT; constructing VCF REF={variant.ref} background",
                    flush=True,
                )
                sequence = (
                    sequence[:offset]
                    + variant.ref.upper()
                    + sequence[offset + len(variant.ref) :]
                )
            else:
                message = (
                    f"Reference mismatch for {variant.variant_id}: VCF={variant.ref} "
                    f"ALT={variant.alt} FASTA={observed} at "
                    f"{variant.chrom}:{variant.pos}"
                )
                if self.strict_reference:
                    raise ValueError(message)
                self._forced_reference_bases += 1
                print(
                    f"[warning] {message}; forcing the VCF REF background", flush=True
                )
                sequence = (
                    sequence[:offset]
                    + variant.ref.upper()
                    + sequence[offset + len(variant.ref) :]
                )
        return SequenceWindow(sequence, start, offset)

    def _tokenize(self, sequences: Sequence[str]) -> Any:
        batch = self._tokenizer(
            list(sequences),
            add_special_tokens=False,
            padding=False,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"]
        if tuple(input_ids.shape) != (len(sequences), self.sequence_length):
            raise RuntimeError(
                "NTv3 tokenizer is not operating at single-base resolution: "
                f"got {tuple(input_ids.shape)}, expected "
                f"({len(sequences)}, {self.sequence_length})"
            )
        return input_ids.to(self._device)

    def _reduce(self, values: np.ndarray) -> float:
        if self.aggregation == "mean":
            return float(np.nanmean(values)) if values.size else 0.0
        return _signed_max_abs(values)

    def _genes(self, chrom: str, start: int, end: int) -> tuple[GeneInterval, ...]:
        if self._gene_index is None:
            raise RuntimeError("setup_task() must be called before gene lookup")
        return self._gene_index.overlapping(chrom, start, end)

    @staticmethod
    def _rows_for_matched(
        variant: VariantRecord, score: float, n_tracks: int
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "gene_id": [variant.gene_id or f"variant:{variant.variant_id}"],
                "tissue": [""],
                "score": [score],
                "n_tracks": [n_tracks],
            }
        )

    def _rows_for_eqtl(
        self,
        variant: VariantRecord,
        genes: Sequence[GeneInterval],
        gene_scores: Mapping[str, float],
        fallback_score: float,
        n_tracks: int,
    ) -> pd.DataFrame:
        gene_ids = tuple(dict.fromkeys(gene.gene_id for gene in genes))
        if not gene_ids:
            gene_ids = (f"variant:{variant.variant_id}",)
        tissues = self._active_tissues or (
            (variant.tissue,) if variant.tissue else ("",)
        )
        rows = []
        for gene_id in gene_ids:
            score = float(gene_scores.get(gene_id, fallback_score))
            for tissue in tissues:
                rows.append(
                    {
                        "gene_id": gene_id,
                        "tissue": tissue,
                        "score": score,
                        "n_tracks": n_tracks,
                    }
                )
        return pd.DataFrame(rows)

    def _rows_for_broad_eqtl(
        self,
        genes: Sequence[GeneInterval],
        variant_delta: np.ndarray,
        prediction_start: int,
    ) -> pd.DataFrame:
        """Aggregate official NTv3 RNA tracks with Borzoi's tissue map."""
        tissues = self._active_tissues or tuple(self._tissue_head_indices)
        unknown = [
            tissue for tissue in tissues if tissue not in self._tissue_head_indices
        ]
        if unknown:
            raise ValueError(f"NTv3 has no broad eQTL track map for: {unknown}")
        rows = []
        prediction_end = prediction_start + int(variant_delta.shape[0])
        for gene in genes:
            left = max(gene.start, prediction_start) - prediction_start
            right = min(gene.end, prediction_end) - prediction_start
            for tissue in tissues:
                indexes = self._tissue_head_indices[tissue]
                rows.append(
                    {
                        "gene_id": gene.gene_id,
                        "tissue": tissue,
                        "score": self._reduce(variant_delta[left:right, indexes]),
                        "n_tracks": len(indexes),
                    }
                )
        return pd.DataFrame(rows, columns=["gene_id", "tissue", "score", "n_tracks"])

    def _score_pre(
        self, variants: Sequence[VariantRecord], windows: Sequence[SequenceWindow]
    ) -> list[pd.DataFrame]:
        torch = self._torch
        input_ids = self._tokenize([window.sequence for window in windows])
        offsets = torch.tensor(
            [window.variant_offset for window in windows], device=self._device
        )
        masked = input_ids.clone()
        masked[torch.arange(len(windows), device=self._device), offsets] = (
            self._tokenizer.mask_token_id
        )
        with torch.inference_mode():
            output = self._model(input_ids=masked)
            logits = output["logits"][
                torch.arange(len(windows), device=self._device), offsets
            ].float()
            log_probs = torch.log_softmax(logits, dim=-1)

        scores: list[float] = []
        for index, variant in enumerate(variants):
            ref_ids = self._tokenizer(variant.ref.upper(), add_special_tokens=False)[
                "input_ids"
            ]
            alt_ids = self._tokenizer(variant.alt.upper(), add_special_tokens=False)[
                "input_ids"
            ]
            if len(ref_ids) != 1 or len(alt_ids) != 1:
                raise RuntimeError("NTv3 tokenizer did not map an allele to one token")
            scores.append(
                float(
                    (log_probs[index, alt_ids[0]] - log_probs[index, ref_ids[0]]).cpu()
                )
            )

        tables = []
        for variant, window, score in zip(variants, windows, scores):
            if self._active_task == "eqtl":
                genes = self._genes(
                    variant.chrom, window.start, window.start + self.sequence_length
                )
                tables.append(
                    self._rows_for_eqtl(variant, genes, {}, score, n_tracks=1)
                )
            else:
                tables.append(self._rows_for_matched(variant, score, n_tracks=1))
        return tables

    def _score_post(
        self, variants: Sequence[VariantRecord], windows: Sequence[SequenceWindow]
    ) -> list[pd.DataFrame]:
        torch = self._torch
        sequences: list[str] = []
        for variant, window in zip(variants, windows):
            reference = window.sequence
            alternate = (
                reference[: window.variant_offset]
                + variant.alt.upper()
                + reference[window.variant_offset + 1 :]
            )
            sequences.extend((reference, alternate))
        input_ids = self._tokenize(sequences)
        species_ids = self._model.encode_species([self.species] * len(sequences))
        if hasattr(species_ids, "to"):
            species_ids = species_ids.to(self._device)
        with torch.inference_mode():
            output = self._model(input_ids=input_ids, species_ids=species_ids)
            if self._active_task == "eqtl":
                values = output["bigwig_tracks_logits"][:, :, self._head_indices]
            else:
                logits = output["bed_tracks_logits"][
                    :, :, self._head_indices, :
                ].float()
                values = torch.softmax(logits, dim=-1)[..., 1]
            delta = (values[1::2].float() - values[0::2].float()).cpu().numpy()

        tables = []
        for variant, window, variant_delta in zip(variants, windows, delta):
            global_score = self._reduce(variant_delta)
            if self._active_task != "eqtl":
                tables.append(
                    self._rows_for_matched(
                        variant, global_score, n_tracks=len(self._head_indices)
                    )
                )
                continue

            output_length = int(variant_delta.shape[0])
            prediction_start = (
                window.start + (self.sequence_length - output_length) // 2
            )
            prediction_end = prediction_start + output_length
            genes = self._genes(variant.chrom, prediction_start, prediction_end)
            if self._tissue_head_indices:
                tables.append(
                    self._rows_for_broad_eqtl(genes, variant_delta, prediction_start)
                )
                continue
            gene_scores: dict[str, float] = {}
            for gene in genes:
                left = max(gene.start, prediction_start) - prediction_start
                right = min(gene.end, prediction_end) - prediction_start
                gene_scores[gene.gene_id] = self._reduce(variant_delta[left:right])
            tables.append(
                self._rows_for_eqtl(
                    variant,
                    genes,
                    gene_scores,
                    global_score,
                    n_tracks=len(self._head_indices),
                )
            )
        return tables

    def score_variant_tables(
        self, variants: Sequence[VariantRecord]
    ) -> list[pd.DataFrame]:
        if self._model is None or self._active_task is None:
            raise RuntimeError("setup_task() must be called before scoring")
        windows = [self._window(variant) for variant in variants]
        if self.checkpoint_type == "post":
            return self._score_post(variants, windows)
        return self._score_pre(variants, windows)

    def score_variant_table(self, variant: VariantRecord) -> pd.DataFrame:
        return self.score_variant_tables((variant,))[0]

    def predict_batch(
        self,
        task: str,
        variants: Sequence[VariantRecord],
        context: TaskContext,
    ):
        raise NotImplementedError(
            "Use NTv3ModelAdapter; NTv3 emits gene/tissue prediction tables"
        )

    def teardown_task(self, context: TaskContext) -> None:
        run_info = {
            "model": self.name,
            "model_id": self.model_id,
            "revision": self.revision,
            "code_revision": self.code_revision,
            "checkpoint_type": self.checkpoint_type,
            "scoring_method": (
                "alt_minus_ref_selected_heads"
                if self.checkpoint_type == "post"
                else "masked_base_log_p_alt_minus_log_p_ref"
            ),
            "task": context.task,
            "heads": list(self._head_names),
            "sequence_length": self.sequence_length,
            "species": self.species,
            "device": str(self._device),
            "dtype": str(self._dtype),
            "aggregation": self.aggregation,
            "tissue_specific": bool(self._tissue_head_indices),
            "eqtl_tissue_resolution": (
                "borzoi_broad" if self._tissue_head_indices else "tissue_agnostic"
            ),
            "eqtl_track_metadata": (
                str(self.eqtl_track_metadata_path)
                if self.eqtl_track_metadata_path is not None
                else None
            ),
            "eqtl_track_metadata_sha256": self.eqtl_track_metadata_sha256,
            "eqtl_tissue_track_sources": self._tissue_head_sources,
            "fasta_matches_vcf_alt": self._fasta_alt_matches,
            "forced_reference_bases": self._forced_reference_bases,
            "fasta": str(self.fasta_path),
            "gtf": str(self.gtf_path),
        }
        context.output_dir.mkdir(parents=True, exist_ok=True)
        (context.output_dir / "model_run.json").write_text(
            json.dumps(run_info, indent=2, sort_keys=True) + "\n"
        )


class NTv3ModelAdapter(ModelAdapter):
    """Batched, sharded adapter for NTv3 tidy predictions."""

    def __init__(
        self,
        model: NTv3Model,
        model_config: Mapping[str, Any] | None = None,
        batch_size: int = 1,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.name = model.name
        self.model_config = dict(model_config or {})
        self.batch_size = int(batch_size)

    def identity_payload(self) -> Mapping[str, Any]:
        return {
            "model_class": "qtl_benchmark.models.ntv3:NTv3Model",
            "model_config": self.model_config,
            "batch_size": self.batch_size,
        }

    def predict(
        self,
        dataset: InferenceDataset,
        store: PredictionStore,
        context: BenchmarkContext,
    ) -> None:
        num_shards = int(context.options.get("num_shards", 1))
        shard_index = int(context.options.get("shard_index", 0))
        if num_shards < 1 or not 0 <= shard_index < num_shards:
            raise ValueError(f"Invalid prediction shard {shard_index}/{num_shards}")
        assigned = tuple(
            record
            for index, record in enumerate(dataset.variants)
            if index % num_shards == shard_index
        )
        pending = [record for record in assigned if not store.completed(record)]
        print(
            f"[predict/{self.name}] shard={shard_index + 1}/{num_shards} "
            f"variants={len(assigned)} cached={len(assigned) - len(pending)} "
            f"pending={len(pending)} batch_size={self.batch_size}",
            flush=True,
        )
        if not pending:
            return
        task_context = TaskContext(
            task=dataset.task,
            data_dir=dataset.data_dir,
            output_dir=context.output_dir,
            options={**context.options, "tissues": dataset.tissues},
        )
        self.model.setup_task(task_context)
        try:
            for batch_start in range(0, len(pending), self.batch_size):
                batch = pending[batch_start : batch_start + self.batch_size]
                started = time.time()
                outputs = self.model.score_variant_tables(batch)
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        f"NTv3 returned {len(outputs)} tables for {len(batch)} variants"
                    )
                elapsed = (time.time() - started) / len(batch)
                for record, output in zip(batch, outputs):
                    normalized = TidyVariantModelAdapter._normalize_model_output(
                        output, record
                    )
                    store.save(record, normalized, elapsed)
                print(
                    f"[predict/{self.name}] {min(batch_start + len(batch), len(pending))}/"
                    f"{len(pending)} rows={sum(len(table) for table in outputs)} "
                    f"batch_elapsed={elapsed * len(batch):.1f}s",
                    flush=True,
                )
        finally:
            self.model.teardown_task(task_context)
