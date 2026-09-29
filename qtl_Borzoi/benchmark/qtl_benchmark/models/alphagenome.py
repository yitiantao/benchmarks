"""Local AlphaGenome adapter for the model-independent QTL benchmark.

The adapter owns AlphaGenome-specific setup and variant scoring only.  VCF
reading, batching, prediction-table validation, and benchmark metrics remain in
the shared :mod:`qtl_benchmark` pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from ..model_base import QTLModel, TaskContext, VariantPrediction, VariantRecord


class AlphaGenomeModel(QTLModel):
    """Score all supported QTL tasks with a local AlphaGenome checkpoint.

    The adapter selects AlphaGenome's recommended task scorer and returns its
    gene/track tidy table. Aggregation, persistence, and matched evaluation are
    owned by the shared framework rather than this model implementation.

    Imports of JAX and AlphaGenome are intentionally delayed until
    :meth:`setup_task`, allowing the benchmark package and CLI help to work in
    environments that do not have AlphaGenome installed.
    """

    supported_tasks = frozenset({"eqtl", "sqtl", "paqtl", "ipaqtl"})
    DEFAULT_TASK_SCORERS = {
        "eqtl": "RNA_SEQ",
        "sqtl": "SPLICE_JUNCTIONS",
        "paqtl": "POLYADENYLATION",
        "ipaqtl": "POLYADENYLATION",
    }

    def __init__(
        self,
        checkpoint_path: str,
        fasta_path: str,
        gtf_feather_path: str,
        splice_site_starts_feather_path: str,
        splice_site_ends_feather_path: str,
        pas_feather_path: str | None = None,
        research_source: str | None = None,
        root_dir: str = ".",
        name: str = "alphagenome-local",
        sequence_length: int = 2**20,
        device: int = 0,
        aggregation: str = "max_abs",
        gene_id: str | None = None,
        gtex_tissues: Sequence[str] | None = None,
        ontology_terms: Sequence[str] | None = None,
        empty_score: float = 0.0,
        save_raw_scores: bool = False,
        task_scorers: Mapping[str, str] | None = None,
    ) -> None:
        self.name = name
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.checkpoint_path = self._resolve(checkpoint_path)
        self.fasta_path = self._resolve(fasta_path)
        self.gtf_feather_path = self._resolve(gtf_feather_path)
        self.splice_site_starts_feather_path = self._resolve(
            splice_site_starts_feather_path
        )
        self.splice_site_ends_feather_path = self._resolve(
            splice_site_ends_feather_path
        )
        self.pas_feather_path = (
            self._resolve(pas_feather_path) if pas_feather_path is not None else None
        )
        self.research_source = (
            self._resolve(research_source) if research_source is not None else None
        )
        self.sequence_length = int(sequence_length)
        self.device_index = int(device)
        self.aggregation = aggregation
        self.gene_id = self._patchless(gene_id) if gene_id else None
        self.gtex_tissues = frozenset(gtex_tissues or ())
        self.ontology_terms = frozenset(ontology_terms or ())
        self.empty_score = float(empty_score)
        self.save_raw_scores = bool(save_raw_scores)
        self.task_scorers = dict(self.DEFAULT_TASK_SCORERS)
        self.task_scorers.update(task_scorers or {})
        unknown_tasks = set(self.task_scorers) - self.supported_tasks
        if unknown_tasks:
            raise ValueError(f"Unknown task_scorers keys: {sorted(unknown_tasks)}")

        valid_lengths = {2**power for power in range(17, 21)}
        if self.sequence_length not in valid_lengths:
            raise ValueError(
                f"sequence_length must be one of {sorted(valid_lengths)}; "
                f"got {self.sequence_length}"
            )
        if self.device_index < 0:
            raise ValueError("device must be non-negative")
        if self.aggregation != "max_abs":
            raise ValueError("AlphaGenomeModel currently supports aggregation='max_abs'")

        self._model: Any | None = None
        self._genome: Any | None = None
        self._variant_scorers: Any | None = None
        self._device: Any | None = None
        self._active_task: str | None = None

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.root_dir / path
        return path.resolve()

    @staticmethod
    def _patchless(value: str) -> str:
        return str(value).split(".", 1)[0]

    def _require_assets(self, task: str) -> None:
        required = {
            "checkpoint": self.checkpoint_path,
            "FASTA": self.fasta_path,
            "FASTA index": Path(str(self.fasta_path) + ".fai"),
            "GTF Feather": self.gtf_feather_path,
            "splice-site starts Feather": self.splice_site_starts_feather_path,
            "splice-site ends Feather": self.splice_site_ends_feather_path,
        }
        missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
        if task in {"paqtl", "ipaqtl"}:
            if self.pas_feather_path is None:
                missing.append(
                    "polyadenylation Feather: set pas_feather_path for paQTL/iPaQTL"
                )
            elif not self.pas_feather_path.exists():
                missing.append(f"polyadenylation Feather: {self.pas_feather_path}")
        if missing:
            raise FileNotFoundError("Missing local AlphaGenome assets:\n" + "\n".join(missing))

    def setup_task(self, context: TaskContext) -> None:
        self.validate_task(context.task)
        self._require_assets(context.task)
        self._active_task = context.task
        # A multi-task suite reuses the same checkpoint/model instance. The
        # recommended scorer is selected per call in score_variant_table().
        if self._model is not None:
            return
        if self.research_source is not None:
            source = self.research_source / "src"
            if not (source / "alphagenome_research").is_dir():
                raise FileNotFoundError(
                    f"AlphaGenome Research source package not found below {source}"
                )
            if str(source) not in sys.path:
                sys.path.insert(0, str(source))

        import jax
        from alphagenome.models import dna_model as dna_model_api
        from alphagenome.data import genome
        from alphagenome.models import variant_scorers
        from alphagenome_research.model import dna_model

        devices = jax.devices("gpu")
        if self.device_index >= len(devices):
            raise RuntimeError(
                f"Requested GPU {self.device_index}, but JAX found {len(devices)}: {devices}"
            )
        self._device = devices[self.device_index]

        # Preserve both organisms' bundled output metadata so the architecture
        # exactly matches all_folds.  Only human receives local reference files;
        # this prevents the default remote FASTA/GTF/calibration downloads.
        organism_settings = {
            dna_model_api.Organism.HOMO_SAPIENS: dna_model.OrganismSettings(
                fasta_path=str(self.fasta_path),
                gtf_feather_path=str(self.gtf_feather_path),
                pas_feather_path=(
                    str(self.pas_feather_path)
                    if self.pas_feather_path is not None
                    else None
                ),
                splice_site_starts_feather_path=str(
                    self.splice_site_starts_feather_path
                ),
                splice_site_ends_feather_path=str(
                    self.splice_site_ends_feather_path
                ),
            ),
            dna_model_api.Organism.MUS_MUSCULUS: dna_model.OrganismSettings(),
        }
        started = time.time()
        self._model = dna_model.create(
            str(self.checkpoint_path),
            organism_settings=organism_settings,
            device=self._device,
        )
        self._genome = genome
        self._variant_scorers = variant_scorers
        print(
            f"[{self.name}] loaded {self.checkpoint_path} on {self._device} "
            f"in {time.time() - started:.1f}s",
            flush=True,
        )

    def _filter_scores(self, table, variant: VariantRecord):
        table = table.copy()
        # ``variant_scorers.tidy_scores`` returns an empty, columnless frame
        # when a valid variant has no rows for the selected scorer (for
        # example, no matching splice junction/gene in the scoring window).
        # This is a normal zero-prediction result, not a malformed response.
        if table.empty:
            return table
        if "raw_score" not in table:
            raise ValueError(
                "AlphaGenome score table is missing raw_score; "
                f"columns={list(table.columns)} variant={variant.variant_id}"
            )
        table["raw_score"] = table["raw_score"].astype(float)
        table = table[np.isfinite(table.raw_score)]

        gene_id = self._patchless(variant.gene_id) if variant.gene_id else self.gene_id
        if gene_id is not None and "gene_id" in table:
            table = table[
                table.gene_id.astype(str).map(self._patchless) == gene_id
            ]
        if self.gtex_tissues and "gtex_tissue" in table:
            table = table[table.gtex_tissue.isin(self.gtex_tissues)]
        if self.ontology_terms and "ontology_curie" in table:
            table = table[table.ontology_curie.isin(self.ontology_terms)]
        return table

    def score_variant_table(self, variant: VariantRecord):
        """Return the task-appropriate tidy gene/track score table.

        The tissue-aware GTEx evaluator uses this model-facing boundary and
        applies Borzoi's gene/tissue aggregation outside the adapter.
        """
        if self._model is None or self._genome is None or self._variant_scorers is None:
            raise RuntimeError("setup_task() must be called before scoring variants")

        ag_variant = self._genome.Variant(
            chromosome=variant.chrom,
            position=variant.pos,
            reference_bases=variant.ref,
            alternate_bases=variant.alt,
        )
        interval = ag_variant.reference_interval.resize(self.sequence_length)
        if self._active_task is None:
            raise RuntimeError("setup_task() must be called before scoring variants")
        scorer_name = self.task_scorers[self._active_task]
        scorer = self._variant_scorers.RECOMMENDED_VARIANT_SCORERS[scorer_name]
        results = self._model.score_variant(
            interval,
            ag_variant,
            variant_scorers=[scorer],
        )
        table = self._variant_scorers.tidy_scores(
            results,
            match_gene_strand=True,
        )
        if table is None:
            import pandas as pd

            return pd.DataFrame()
        return self._filter_scores(table, variant)

    def score_variant_tables(self, variants: Sequence[VariantRecord]):
        """Score a small group through AlphaGenome's local threaded API.

        ``DnaModel.score_variants`` runs one local ``score_variant`` call per
        worker; it is concurrency over the local JAX model rather than a single
        stacked tensor batch. Keeping that distinction here lets the benchmark
        expose a useful throughput knob without changing score semantics.
        """
        if not variants:
            return []
        if self._model is None or self._genome is None or self._variant_scorers is None:
            raise RuntimeError("setup_task() must be called before scoring variants")
        if self._active_task is None:
            raise RuntimeError("setup_task() must be called before scoring variants")

        ag_variants = [
            self._genome.Variant(
                chromosome=variant.chrom,
                position=variant.pos,
                reference_bases=variant.ref,
                alternate_bases=variant.alt,
            )
            for variant in variants
        ]
        intervals = [
            ag_variant.reference_interval.resize(self.sequence_length)
            for ag_variant in ag_variants
        ]
        scorer_name = self.task_scorers[self._active_task]
        scorer = self._variant_scorers.RECOMMENDED_VARIANT_SCORERS[scorer_name]
        results_by_variant = self._model.score_variants(
            intervals=intervals,
            variants=ag_variants,
            variant_scorers=[scorer],
            progress_bar=False,
            max_workers=len(variants),
        )

        import pandas as pd

        tables = []
        for results, variant in zip(results_by_variant, variants, strict=True):
            table = self._variant_scorers.tidy_scores(
                results,
                match_gene_strand=True,
            )
            if table is None:
                table = pd.DataFrame()
            tables.append(self._filter_scores(table, variant))
        return tables

    def _score_variant(
        self,
        variant: VariantRecord,
        context: TaskContext,
    ) -> VariantPrediction:
        table = self.score_variant_table(variant)

        if table.empty:
            return VariantPrediction(
                variant_id=variant.variant_id,
                score=self.empty_score,
                metadata={
                    "score_rows": 0,
                    "selected_gene_id": "",
                    "selected_gtex_tissue": "",
                    "selected_track": "",
                },
            )

        best = table.loc[table.raw_score.abs().idxmax()]
        if self.save_raw_scores:
            raw_dir = context.output_dir / "raw_scores" / variant.split
            raw_dir.mkdir(parents=True, exist_ok=True)
            safe_id = variant.variant_id.replace("/", "_")
            table.to_csv(
                raw_dir / f"{safe_id}.tsv.gz",
                sep="\t",
                index=False,
                compression="gzip",
            )

        metadata = {
            "score_rows": int(len(table)),
            "selected_gene_id": str(best.get("gene_id", "")),
            "selected_gtex_tissue": str(best.get("gtex_tissue", "")),
            "selected_track": str(best.get("track_name", "")),
        }
        return VariantPrediction(
            variant_id=variant.variant_id,
            score=float(best.raw_score),
            metadata=metadata,
        )

    def predict_batch(
        self,
        task: str,
        variants: Sequence[VariantRecord],
        context: TaskContext,
    ) -> Sequence[VariantPrediction]:
        self.validate_task(task)
        predictions = []
        for index, variant in enumerate(variants, 1):
            started = time.time()
            prediction = self._score_variant(variant, context)
            predictions.append(prediction)
            print(
                f"[{self.name}/{task}] {index}/{len(variants)} "
                f"{variant.split} {variant.variant_id} "
                f"score={prediction.score:.6g} ({time.time() - started:.1f}s)",
                flush=True,
            )
        return predictions

    def teardown_task(self, context: TaskContext) -> None:
        run_info = {
            "model": self.name,
            "checkpoint": str(self.checkpoint_path),
            "fasta": str(self.fasta_path),
            "gtf_feather": str(self.gtf_feather_path),
            "splice_site_starts_feather": str(
                self.splice_site_starts_feather_path
            ),
            "splice_site_ends_feather": str(self.splice_site_ends_feather_path),
            "pas_feather": (
                str(self.pas_feather_path)
                if self.pas_feather_path is not None
                else None
            ),
            "task": context.task,
            "task_scorer": self.task_scorers[context.task],
            "sequence_length": self.sequence_length,
            "device": str(self._device),
            "variant_batch_size": int(
                context.options.get("variant_batch_size", 1)
            ),
            "aggregation": self.aggregation,
            "gtex_tissues": sorted(self.gtex_tissues),
            "ontology_terms": sorted(self.ontology_terms),
        }
        context.output_dir.mkdir(parents=True, exist_ok=True)
        (context.output_dir / "model_run.json").write_text(
            json.dumps(run_info, indent=2) + "\n"
        )
