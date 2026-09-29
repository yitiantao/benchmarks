"""Adapter for the locally implemented, post-trained DNA-FM (NTv3) model.

DNA-FM is built by the sibling ``DNA_FM`` repository rather than Hugging Face.
This adapter reconstructs the exact PyTorch post-training model from the YAML
saved with a training run, loads an Accelerate ``model.safetensors`` file, and
emits the same tidy QTL prediction schema as AlphaGenome.
"""

from __future__ import annotations

from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..model_base import QTLModel, TaskContext, VariantRecord
from .borzoi import TISSUE_KEYWORDS
from .ntv3 import GeneIndex, SequenceWindow, _signed_max_abs


DEFAULT_ANNOTATION_HEADS: Mapping[str, tuple[str, ...]] = {
    "sqtl": ("splice_donor", "splice_acceptor"),
    "paqtl": ("polyA_signal",),
    "ipaqtl": ("polyA_signal", "3UTR_plus", "3UTR_minus"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256(source_root: Path) -> str:
    """Fingerprint the inference-critical DNA-FM implementation files."""
    relative_paths = (
        "src/dna_ntv3/models/tokenizer.py",
        "src/dna_ntv3/models/ntv3/configuration.py",
        "src/dna_ntv3/models/ntv3/layers.py",
        "src/dna_ntv3/models/ntv3/modeling.py",
        "src/dna_ntv3/posttraining/modeling.py",
    )
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = source_root / relative
        if not path.is_file():
            return ""
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


class DNAFMModel(QTLModel):
    """Score QTL variants with a local DNA-FM post-training checkpoint."""

    supported_tasks = frozenset({"eqtl", "sqtl", "paqtl", "ipaqtl"})

    def __init__(
        self,
        source_root: str,
        checkpoint_path: str,
        run_config_path: str,
        fasta_path: str,
        gtf_path: str,
        root_dir: str = ".",
        canonical_manifest_path: str | None = None,
        name: str = "DNA-FM",
        sequence_length: int = 131072,
        species: str = "human",
        device: str = "cuda",
        mixed_precision: str = "float32",
        strict_reference: bool = True,
        aggregation: str = "max_abs",
        checkpoint_sha256: str | None = None,
        run_config_sha256: str | None = None,
        tissue_keywords: Mapping[str, str] | None = None,
        annotation_heads: Mapping[str, Sequence[str]] | None = None,
        eqtl_dataset: str = "gtex",
        eqtl_fallback_datasets: Sequence[str] = ("encode_v3",),
    ) -> None:
        self.name = str(name)
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.source_root = self._resolve(source_root)
        self.checkpoint_path = self._model_file(self._resolve(checkpoint_path))
        self.run_config_path = self._resolve(run_config_path)
        self.fasta_path = self._resolve(fasta_path)
        self.gtf_path = self._resolve(gtf_path)
        self.canonical_manifest_path = (
            self._resolve(canonical_manifest_path)
            if canonical_manifest_path is not None
            else None
        )
        self.sequence_length = int(sequence_length)
        self.species = str(species)
        self.device_name = str(device)
        self.mixed_precision = str(mixed_precision)
        self.strict_reference = bool(strict_reference)
        self.aggregation = str(aggregation)
        self.eqtl_dataset = str(eqtl_dataset).lower()
        self.eqtl_fallback_datasets = tuple(
            str(dataset).lower() for dataset in eqtl_fallback_datasets
        )
        self.tissue_keywords = dict(TISSUE_KEYWORDS)
        self.tissue_keywords.update(tissue_keywords or {})
        self.annotation_heads = dict(DEFAULT_ANNOTATION_HEADS)
        if annotation_heads:
            self.annotation_heads.update(
                {task: tuple(labels) for task, labels in annotation_heads.items()}
            )

        if self.sequence_length <= 0 or self.sequence_length % 128:
            raise ValueError("sequence_length must be positive and divisible by 128")
        if self.aggregation not in {"max_abs", "mean"}:
            raise ValueError("aggregation must be max_abs or mean")
        if self.mixed_precision not in {"float32", "bfloat16", "float16"}:
            raise ValueError("mixed_precision must be float32, bfloat16, or float16")
        unknown = set(self.annotation_heads) - {"sqtl", "paqtl", "ipaqtl"}
        if unknown:
            raise ValueError(f"Unknown annotation_heads tasks: {sorted(unknown)}")

        self.checkpoint_sha256 = _sha256(self.checkpoint_path)
        self.run_config_sha256 = _sha256(self.run_config_path)
        self.source_sha256 = _source_sha256(self.source_root)
        if checkpoint_sha256 and checkpoint_sha256 != self.checkpoint_sha256:
            raise ValueError(
                "DNA-FM checkpoint SHA-256 mismatch: "
                f"expected={checkpoint_sha256} observed={self.checkpoint_sha256}"
            )
        if run_config_sha256 and run_config_sha256 != self.run_config_sha256:
            raise ValueError(
                "DNA-FM run config SHA-256 mismatch: "
                f"expected={run_config_sha256} observed={self.run_config_sha256}"
            )

        self._torch: Any | None = None
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._genome: Any | None = None
        self._gene_index: GeneIndex | None = None
        self._device: Any | None = None
        self._run_config: dict[str, Any] | None = None
        self._manifest: dict[str, Any] | None = None
        self._full_human_head: Any | None = None
        self._annotation_head: Any | None = None
        self._active_task: str | None = None
        self._annotation_indices: tuple[int, ...] = ()
        self._annotation_names: tuple[str, ...] = ()
        self._selected_track_ids: tuple[str, ...] = ()
        self._tissue_track_indices: dict[str, tuple[int, ...]] = {}
        self._tissue_track_sources: dict[str, str] = {}
        self._fasta_alt_matches = 0
        self._forced_reference_bases = 0

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.root_dir / path
        return path.resolve()

    @staticmethod
    def _model_file(path: Path) -> Path:
        if path.is_file():
            return path
        if path.is_dir():
            for filename in ("model.safetensors", "pytorch_model.bin"):
                candidate = path / filename
                if candidate.is_file():
                    return candidate
        raise FileNotFoundError(
            f"DNA-FM checkpoint must be a model file or checkpoint directory: {path}"
        )

    def identity_metadata(self) -> dict[str, str]:
        """Content identities added to the benchmark cache fingerprint."""
        return {
            "resolved_checkpoint": str(self.checkpoint_path),
            "resolved_checkpoint_sha256": self.checkpoint_sha256,
            "resolved_run_config": str(self.run_config_path),
            "resolved_run_config_sha256": self.run_config_sha256,
            "resolved_source_sha256": self.source_sha256,
        }

    def _require_assets(self) -> None:
        source = self.source_root / "src"
        required = {
            "DNA-FM source package": source / "dna_ntv3",
            "checkpoint": self.checkpoint_path,
            "run config": self.run_config_path,
            "FASTA": self.fasta_path,
            "FASTA index": Path(str(self.fasta_path) + ".fai"),
            "GTF": self.gtf_path,
        }
        missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing DNA-FM benchmark assets:\n" + "\n".join(missing))

    def _load_training_metadata(self) -> None:
        if self._run_config is not None:
            return
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("DNA-FM inference requires PyYAML") from error

        config = yaml.safe_load(self.run_config_path.read_text())
        if not isinstance(config, dict):
            raise ValueError(f"Invalid DNA-FM YAML config: {self.run_config_path}")
        post = config.get("posttraining", {})
        species = tuple(post.get("species", ()))
        if self.species not in species:
            raise ValueError(f"species={self.species!r} is absent from checkpoint config")
        tracks_by_species = post.get("functional_tracks", {})
        configured_tracks = tuple(tracks_by_species.get(self.species, ()))
        if not configured_tracks:
            raise ValueError(f"DNA-FM config has no functional tracks for {self.species}")

        manifest_path = self.canonical_manifest_path
        if manifest_path is None:
            value = config.get("data", {}).get("canonical_manifest")
            if not value:
                raise ValueError(
                    "run config has no data.canonical_manifest; set canonical_manifest_path"
                )
            manifest_path = Path(value).expanduser().resolve()
            self.canonical_manifest_path = manifest_path
        if not manifest_path.is_file():
            raise FileNotFoundError(f"DNA-FM canonical manifest not found: {manifest_path}")
        expected_manifest_hash = config.get("data", {}).get("canonical_manifest_sha256")
        if expected_manifest_hash:
            observed = _sha256(manifest_path)
            if observed != expected_manifest_hash:
                raise ValueError(
                    "DNA-FM canonical manifest SHA-256 mismatch: "
                    f"expected={expected_manifest_hash} observed={observed}"
                )
        manifest = json.loads(manifest_path.read_text())
        manifest_tracks = manifest["species"][self.species]["tracks"]
        by_id = {str(track["track_id"]): track for track in manifest_tracks}
        if len(by_id) != len(manifest_tracks):
            raise ValueError("DNA-FM canonical manifest contains duplicate track IDs")
        missing = sorted(set(configured_tracks) - set(by_id))
        if missing:
            raise ValueError(f"DNA-FM manifest is missing configured tracks: {missing[:8]}")

        full_index = {track_id: index for index, track_id in enumerate(configured_tracks)}
        selected_full_indices: list[int] = []
        full_indices_by_tissue: dict[str, tuple[int, ...]] = {}
        source_by_tissue: dict[str, str] = {}
        for tissue, keyword in self.tissue_keywords.items():
            indexes = []
            selected_dataset = ""
            for candidate_dataset in (
                self.eqtl_dataset,
                *self.eqtl_fallback_datasets,
            ):
                for track_id in configured_tracks:
                    track = by_id[track_id]
                    track_tissue = str(track.get("tissue", "")).lower()
                    dataset = str(track.get("dataset", "")).lower()
                    matches_keyword = keyword.lower() in track_tissue
                    if keyword == "blood" and "blood_vessel" in track_tissue:
                        matches_keyword = False
                    if (
                        bool(track.get("is_rna"))
                        and dataset == candidate_dataset
                        and matches_keyword
                    ):
                        indexes.append(full_index[track_id])
                if indexes:
                    selected_dataset = candidate_dataset
                    break
            if not indexes:
                raise ValueError(
                    "No RNA tracks matched "
                    f"{tissue} ({keyword}) in datasets "
                    f"{(self.eqtl_dataset, *self.eqtl_fallback_datasets)}"
                )
            full_indices_by_tissue[tissue] = tuple(indexes)
            source_by_tissue[tissue] = selected_dataset
            selected_full_indices.extend(indexes)

        selected_full_indices = sorted(set(selected_full_indices))
        remap = {full: selected for selected, full in enumerate(selected_full_indices)}
        self._selected_track_ids = tuple(
            configured_tracks[index] for index in selected_full_indices
        )
        self._tissue_track_indices = {
            tissue: tuple(remap[index] for index in indexes)
            for tissue, indexes in full_indices_by_tissue.items()
        }
        self._tissue_track_sources = source_by_tissue
        self._selected_full_track_indices = tuple(selected_full_indices)
        self._run_config = config
        self._manifest = manifest

    def _build_selected_functional_head(self) -> Any:
        torch = self._torch
        full = self._full_human_head
        if torch is None or full is None:
            raise RuntimeError("DNA-FM model has not been loaded")
        indexes = torch.tensor(self._selected_full_track_indices, dtype=torch.long)

        class SelectedRegressionHead(torch.nn.Module):
            def __init__(self, layer_norm, projection) -> None:
                super().__init__()
                self.layer_norm = layer_norm
                self.head = projection

            def forward(self, values):
                return torch.nn.functional.softplus(
                    self.head(self.layer_norm(values))
                )

        layer_norm = copy.deepcopy(full.layer_norm)
        projection = torch.nn.Linear(
            full.head.in_features,
            len(self._selected_full_track_indices),
            bias=full.head.bias is not None,
        )
        with torch.no_grad():
            projection.weight.copy_(full.head.weight.index_select(0, indexes))
            if projection.bias is not None:
                projection.bias.copy_(full.head.bias.index_select(0, indexes))
        return SelectedRegressionHead(layer_norm, projection)

    def _configure_task_outputs(self, task: str) -> None:
        torch, model = self._torch, self._model
        if torch is None or model is None:
            raise RuntimeError("DNA-FM model has not been loaded")
        if task == "eqtl":
            selected_head = self._build_selected_functional_head().to(self._device)
            model.functional_heads = torch.nn.ModuleDict({self.species: selected_head})
            model.functional_track_names = {
                name: self._selected_track_ids if name == self.species else ()
                for name in model.species
            }
            model.annotation_head = None
            self._annotation_indices = ()
            self._annotation_names = ()
        else:
            model.functional_heads = torch.nn.ModuleDict()
            model.functional_track_names = {name: () for name in model.species}
            model.annotation_head = self._annotation_head
            requested = tuple(self.annotation_heads[task])
            available = tuple(model.annotation_labels)
            missing = [name for name in requested if name not in available]
            if missing:
                raise ValueError(
                    f"DNA-FM {task} annotation heads are absent: {missing}"
                )
            self._annotation_names = requested
            self._annotation_indices = tuple(available.index(name) for name in requested)

    def setup_task(self, context: TaskContext) -> None:
        self.validate_task(context.task)
        self._require_assets()
        self._load_training_metadata()
        self._active_task = context.task
        self._fasta_alt_matches = 0
        self._forced_reference_bases = 0
        if self._model is None:
            source = self.source_root / "src"
            if str(source) not in sys.path:
                sys.path.insert(0, str(source))
            try:
                import torch
                from pyfaidx import Fasta
                from safetensors.torch import load_file
                from dna_ntv3.models.ntv3.configuration import NTv3Config
                from dna_ntv3.models.tokenizer import NucleotideTokenizer
                from dna_ntv3.posttraining.modeling import NTv3ForPostTraining
            except ImportError as error:
                raise RuntimeError(
                    "DNA-FM dependencies are missing; use an environment with PyTorch, "
                    "safetensors, PyYAML, pandas, and pyfaidx"
                ) from error

            imported_source = Path(sys.modules["dna_ntv3"].__file__).resolve()
            expected_source = (self.source_root / "src").resolve()
            if expected_source not in imported_source.parents:
                raise RuntimeError(
                    "Imported dna_ntv3 from the wrong source tree: "
                    f"expected below {expected_source}, got {imported_source}"
                )

            if self.device_name.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("DNA-FM requested CUDA, but torch.cuda.is_available() is false")
            device = torch.device(self.device_name)
            architecture = self._run_config["model"]["architecture"]
            post = self._run_config["posttraining"]
            model = NTv3ForPostTraining(
                NTv3Config.from_mapping(architecture),
                post["species"],
                post.get("functional_tracks", {}),
                annotation_labels=post.get("annotation_labels", ()),
                target_fraction=float(post.get("target_fraction", 0.375)),
            )
            started = time.time()
            if self.checkpoint_path.suffix == ".safetensors":
                state = load_file(str(self.checkpoint_path), device="cpu")
            else:
                state = torch.load(
                    self.checkpoint_path, map_location="cpu", weights_only=True
                )
            state = {key.removeprefix("module."): value for key, value in state.items()}
            model.load_state_dict(state, strict=True)
            del state

            self._full_human_head = model.functional_heads[self.species].cpu()
            model.functional_heads = torch.nn.ModuleDict()
            model.functional_track_names = {name: () for name in model.species}
            model.to(device).eval()
            self._annotation_head = model.annotation_head
            self._torch = torch
            self._model = model
            self._tokenizer = NucleotideTokenizer()
            self._genome = Fasta(
                str(self.fasta_path), as_raw=True, sequence_always_upper=True
            )
            self._gene_index = GeneIndex.from_gtf(self.gtf_path)
            self._device = device
            print(
                f"[{self.name}] loaded {self.checkpoint_path} device={device} "
                f"sequence_length={self.sequence_length} "
                f"mixed_precision={self.mixed_precision} "
                f"checkpoint_sha256={self.checkpoint_sha256} "
                f"in {time.time() - started:.1f}s",
                flush=True,
            )
        self._configure_task_outputs(context.task)
        if context.task == "eqtl":
            print(
                f"[{self.name}/eqtl] selected_rna_tracks={len(self._selected_track_ids)} "
                f"benchmark_tissues={len(self._tissue_track_indices)} "
                f"fallback_tissues="
                f"{sum(source != self.eqtl_dataset for source in self._tissue_track_sources.values())}",
                flush=True,
            )
        else:
            print(
                f"[{self.name}/{context.task}] annotation_heads="
                f"{','.join(self._annotation_names)}",
                flush=True,
            )

    def _window(self, variant: VariantRecord) -> SequenceWindow:
        if self._genome is None:
            raise RuntimeError("setup_task() must be called before sequence extraction")
        if len(variant.ref) != 1 or len(variant.alt) != 1:
            raise ValueError(
                f"DNA-FM QTL adapter currently supports SNVs only: {variant.variant_id}"
            )
        center = variant.pos - 1
        start = center - self.sequence_length // 2
        end = start + self.sequence_length
        chrom_length = len(self._genome[variant.chrom])
        fetch_start, fetch_end = max(0, start), min(chrom_length, end)
        sequence = "N" * max(0, -start)
        sequence += str(self._genome[variant.chrom][fetch_start:fetch_end]).upper()
        sequence += "N" * max(0, end - chrom_length)
        if len(sequence) != self.sequence_length:
            raise RuntimeError(
                f"Reference extraction returned {len(sequence)} bp, "
                f"expected {self.sequence_length}"
            )
        offset = center - start
        observed = sequence[offset]
        if observed != variant.ref.upper():
            if observed == variant.alt.upper():
                self._fasta_alt_matches += 1
                sequence = sequence[:offset] + variant.ref.upper() + sequence[offset + 1 :]
            else:
                message = (
                    f"Reference mismatch for {variant.variant_id}: REF={variant.ref} "
                    f"ALT={variant.alt} FASTA={observed} at {variant.chrom}:{variant.pos}"
                )
                if self.strict_reference:
                    raise ValueError(message)
                self._forced_reference_bases += 1
                print(f"[warning] {message}; forcing VCF REF background", flush=True)
                sequence = sequence[:offset] + variant.ref.upper() + sequence[offset + 1 :]
        return SequenceWindow(sequence, start, offset)

    def _reduce(self, values: np.ndarray) -> float:
        if self.aggregation == "mean":
            return float(np.nanmean(values)) if values.size else 0.0
        return _signed_max_abs(values)

    def _autocast(self):
        if self._device.type != "cuda" or self.mixed_precision == "float32":
            return nullcontext()
        dtype = (
            self._torch.bfloat16
            if self.mixed_precision == "bfloat16"
            else self._torch.float16
        )
        return self._torch.autocast(device_type="cuda", dtype=dtype)

    def _forward(self, sequences: Sequence[str]) -> Any:
        token_rows = self._tokenizer.batch_encode(sequences)
        input_ids = self._torch.tensor(
            token_rows, dtype=self._torch.long, device=self._device
        )
        species_ids = self._model.encode_species(
            [self.species] * len(sequences), device=self._device
        )
        with self._torch.inference_mode(), self._autocast():
            return self._model(input_ids=input_ids, species_ids=species_ids)

    def _score_eqtl(
        self,
        variants: Sequence[VariantRecord],
        windows: Sequence[SequenceWindow],
        output: Any,
    ) -> list[pd.DataFrame]:
        values = output.functional_logits
        if values is None:
            raise RuntimeError("DNA-FM eQTL forward returned no functional logits")
        delta = (values[1::2].float() - values[0::2].float()).cpu().numpy()
        tables: list[pd.DataFrame] = []
        for variant, window, variant_delta in zip(variants, windows, delta, strict=True):
            output_length = int(variant_delta.shape[0])
            prediction_start = window.start + (self.sequence_length - output_length) // 2
            prediction_end = prediction_start + output_length
            if self._gene_index is None:
                raise RuntimeError("DNA-FM gene index is not initialized")
            genes = self._gene_index.overlapping(
                variant.chrom, prediction_start, prediction_end
            )
            rows = []
            for gene in genes:
                left = max(gene.start, prediction_start) - prediction_start
                right = min(gene.end, prediction_end) - prediction_start
                for tissue, indexes in self._tissue_track_indices.items():
                    score = self._reduce(variant_delta[left:right, indexes])
                    rows.append(
                        {
                            "gene_id": gene.gene_id,
                            "tissue": tissue,
                            "score": score,
                        }
                    )
            tables.append(pd.DataFrame(rows, columns=["gene_id", "tissue", "score"]))
        return tables

    def _score_annotation(
        self, variants: Sequence[VariantRecord], output: Any
    ) -> list[pd.DataFrame]:
        logits = output.annotation_logits
        if logits is None:
            raise RuntimeError("DNA-FM QTL forward returned no annotation logits")
        selected = logits[:, :, self._annotation_indices, :].float()
        probabilities = self._torch.softmax(selected, dim=-1)[..., 1]
        delta = (probabilities[1::2] - probabilities[0::2]).cpu().numpy()
        tables = []
        for variant, variant_delta in zip(variants, delta, strict=True):
            tables.append(
                pd.DataFrame(
                    {
                        "gene_id": [variant.gene_id or ""],
                        "tissue": [""],
                        "score": [self._reduce(variant_delta)],
                    }
                )
            )
        return tables

    def score_variant_tables(
        self, variants: Sequence[VariantRecord]
    ) -> list[pd.DataFrame]:
        if self._model is None or self._active_task is None:
            raise RuntimeError("setup_task() must be called before scoring")
        if not variants:
            return []
        windows = [self._window(variant) for variant in variants]
        sequences: list[str] = []
        for variant, window in zip(variants, windows, strict=True):
            reference = window.sequence
            alternate = (
                reference[: window.variant_offset]
                + variant.alt.upper()
                + reference[window.variant_offset + 1 :]
            )
            sequences.extend((reference, alternate))
        output = self._forward(sequences)
        if self._active_task == "eqtl":
            return self._score_eqtl(variants, windows, output)
        return self._score_annotation(variants, output)

    def score_variant_table(self, variant: VariantRecord) -> pd.DataFrame:
        return self.score_variant_tables((variant,))[0]

    def predict_batch(self, task, variants, context):
        raise NotImplementedError(
            "Use TidyVariantModelAdapter; DNA-FM emits gene/tissue prediction tables"
        )

    def teardown_task(self, context: TaskContext) -> None:
        run_info = {
            "model": self.name,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "run_config": str(self.run_config_path),
            "run_config_sha256": self.run_config_sha256,
            "source_root": str(self.source_root),
            "source_sha256": self.source_sha256,
            "task": context.task,
            "scoring_method": "alt_minus_ref",
            "sequence_length": self.sequence_length,
            "target_fraction": float(self._model.target_fraction),
            "species": self.species,
            "device": str(self._device),
            "mixed_precision": self.mixed_precision,
            "aggregation": self.aggregation,
            "eqtl_dataset": self.eqtl_dataset,
            "eqtl_fallback_datasets": list(self.eqtl_fallback_datasets),
            "eqtl_track_count": len(self._selected_track_ids),
            "eqtl_tissue_track_sources": self._tissue_track_sources,
            "annotation_heads": list(self._annotation_names),
            "fasta_matches_vcf_alt": self._fasta_alt_matches,
            "forced_reference_bases": self._forced_reference_bases,
            "fasta": str(self.fasta_path),
            "gtf": str(self.gtf_path),
            "canonical_manifest": str(self.canonical_manifest_path),
        }
        context.output_dir.mkdir(parents=True, exist_ok=True)
        (context.output_dir / "model_run.json").write_text(
            json.dumps(run_info, indent=2, sort_keys=True) + "\n"
        )
