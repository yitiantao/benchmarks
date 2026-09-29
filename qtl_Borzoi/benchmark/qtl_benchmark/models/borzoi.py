"""Borzoi adapter for SED artifacts produced by the bundled inference scripts."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd

from ..core import BenchmarkContext, InferenceDataset, variant_key
from ..interfaces import ModelAdapter
from ..predictions import PredictionStore


# The terms used by Borzoi's GTEx output tracks are intentionally kept here,
# inside the model adapter rather than the dataset/evaluator.
TISSUE_KEYWORDS = {
    "Adipose_Subcutaneous": "adipose",
    "Adipose_Visceral_Omentum": "adipose",
    "Adrenal_Gland": "adrenal_gland",
    "Artery_Aorta": "blood_vessel",
    "Artery_Coronary": "blood_vessel",
    "Artery_Tibial": "blood_vessel",
    "Brain_Amygdala": "brain",
    "Brain_Anterior_cingulate_cortex_BA24": "brain",
    "Brain_Caudate_basal_ganglia": "brain",
    "Brain_Cerebellar_Hemisphere": "brain",
    "Brain_Cerebellum": "brain",
    "Brain_Cortex": "brain",
    "Brain_Frontal_Cortex_BA9": "brain",
    "Brain_Hippocampus": "brain",
    "Brain_Hypothalamus": "brain",
    "Brain_Nucleus_accumbens_basal_ganglia": "brain",
    "Brain_Putamen_basal_ganglia": "brain",
    "Brain_Spinal_cord_cervical_c-1": "brain",
    "Brain_Substantia_nigra": "brain",
    "Breast_Mammary_Tissue": "breast",
    "Cells_Cultured_fibroblasts": "skin",
    "Cells_EBV-transformed_lymphocytes": "blood",
    "Colon_Sigmoid": "colon",
    "Colon_Transverse": "colon",
    "Esophagus_Gastroesophageal_Junction": "esophagus",
    "Esophagus_Mucosa": "esophagus",
    "Esophagus_Muscularis": "esophagus",
    "Heart_Atrial_Appendage": "heart",
    "Heart_Left_Ventricle": "heart",
    "Kidney_Cortex": "kidney",
    "Liver": "liver",
    "Lung": "lung",
    "Minor_Salivary_Gland": "salivary_gland",
    "Muscle_Skeletal": "muscle",
    "Nerve_Tibial": "nerve",
    "Ovary": "ovary",
    "Pancreas": "pancreas",
    "Pituitary": "pituitary",
    "Prostate": "prostate",
    "Skin_Not_Sun_Exposed_Suprapubic": "skin",
    "Skin_Sun_Exposed_Lower_leg": "skin",
    "Small_Intestine_Terminal_Ileum": "small_intestine",
    "Spleen": "spleen",
    "Stomach": "stomach",
    "Testis": "testis",
    "Thyroid": "thyroid",
    "Uterus": "uterus",
    "Vagina": "vagina",
    "Whole_Blood": "blood",
}


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values]
    )


class BorzoiSEDModelAdapter(ModelAdapter):
    """Convert Borzoi tissue SED HDF5 files to the shared prediction schema.

    ``artifact_dir`` is normally the ``ensemble/eqtl_sed`` directory created
    by ``run_eqtl_benchmark.sh``. Neural-network execution remains in the
    upstream-compatible script; all data selection and evaluation are shared.
    """

    def __init__(
        self,
        artifact_dir: str | Path,
        statistic: str = "SED",
        transform: str = "arcsinh",
        name: str = "borzoi",
        tissue_keywords: Mapping[str, str] | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        self.statistic = statistic
        self.transform = transform
        self.name = name
        self.tissue_keywords = dict(TISSUE_KEYWORDS)
        self.tissue_keywords.update(tissue_keywords or {})
        if transform not in {"none", "arcsinh"}:
            raise ValueError("transform must be 'none' or 'arcsinh'")

    def identity_payload(self) -> Mapping[str, Any]:
        return {
            "class": f"{type(self).__module__}:{type(self).__qualname__}",
            "artifact_dir": str(self.artifact_dir),
            "statistic": self.statistic,
            "transform": self.transform,
        }

    def _file(self, tissue: str, split: str) -> Path:
        path = self.artifact_dir / f"{tissue}_{split}" / "sed.h5"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Borzoi prediction artifact: {path}")
        return path

    def _read_tissue(self, tissue: str, split: str) -> pd.DataFrame:
        path = self._file(tissue, split)
        keyword = self.tissue_keywords.get(tissue)
        if keyword is None:
            raise KeyError(
                f"No Borzoi target keyword for {tissue!r}; pass tissue_keywords"
            )
        with h5py.File(path, "r") as scores:
            if self.statistic not in scores:
                raise KeyError(
                    f"{path} has no {self.statistic!r}; keys={list(scores.keys())}"
                )
            labels = _decode(scores["target_labels"][:])
            ids = _decode(scores["target_ids"][:])
            target_mask = np.asarray(
                [
                    "GTEX" in identifier
                    and keyword in label
                    and not (keyword == "blood" and "vessel" in label)
                    for identifier, label in zip(ids, labels)
                ]
            )
            if not target_mask.any():
                raise ValueError(f"No Borzoi GTEx targets matched {tissue} ({keyword})")
            row_indexes = np.asarray(scores["si"][:], dtype=int)
            if len(row_indexes) == 0:
                return pd.DataFrame(
                    columns=["variant_key", "gene_id", "tissue", "score", "n_tracks"]
                )
            values = np.asarray(scores[self.statistic], dtype=np.float32)
            if values.ndim != 2:
                raise ValueError(f"Unexpected {self.statistic} shape in {path}: {values.shape}")
            values = values[:, target_mask].mean(axis=1)
            if self.transform == "arcsinh":
                values = np.arcsinh(values)
            chrom = _decode(scores["chr"][:])[row_indexes]
            pos = np.asarray(scores["pos"][:], dtype=int)[row_indexes]
            ref = _decode(scores["ref_allele"][:])[row_indexes]
            alt = _decode(scores["alt_allele"][:])[row_indexes]
            genes = _decode(scores["gene"][:])
        return pd.DataFrame(
            {
                "variant_key": [
                    f"{item_chrom}:{item_pos}:{item_ref}:{item_alt}"
                    for item_chrom, item_pos, item_ref, item_alt in zip(chrom, pos, ref, alt)
                ],
                "gene_id": pd.Series(genes).str.split(".").str[0],
                "tissue": tissue,
                "score": values,
                "n_tracks": int(target_mask.sum()),
            }
        )

    def predict(
        self,
        dataset: InferenceDataset,
        store: PredictionStore,
        context: BenchmarkContext,
    ) -> None:
        del context
        pending = {variant_key(record): record for record in dataset.variants if not store.completed(record)}
        print(
            f"[predict/{self.name}] variants={len(dataset.variants)} "
            f"cached={len(dataset.variants) - len(pending)} pending={len(pending)}",
            flush=True,
        )
        if not pending:
            return
        started = time.time()
        tables = [
            self._read_tissue(tissue, split)
            for tissue in dataset.tissues
            for split in ("pos", "neg")
        ]
        predictions = pd.concat(tables, ignore_index=True)
        predictions = predictions[predictions.variant_key.isin(pending)].copy()
        # A variant may occur in several tissue VCFs. Duplicate model rows for
        # the same normalized key must agree; averaging is also replicate-safe.
        if not predictions.empty:
            predictions = (
                predictions.groupby(
                    ["variant_key", "gene_id", "tissue"], sort=False, as_index=False
                )
                .agg(score=("score", "mean"), n_tracks=("n_tracks", "max"))
            )
        elapsed = time.time() - started
        for key, record in pending.items():
            store.save(
                record,
                predictions[predictions.variant_key == key],
                elapsed_seconds=elapsed / max(len(pending), 1),
            )


class BorzoiQTLModelAdapter(ModelAdapter):
    """Read Borzoi sQTL/paQTL/iPaQTL fold or ensemble artifacts.

    The defaults exactly follow the bundled Borzoi benchmark drivers:
    ``nDi`` from ``sqtl_span`` for sQTL and ``COVR`` from the corresponding
    polyadenylation output for paQTL/iPaQTL.  The paper sQTL analysis averages
    the final 89 GTEx RNA tracks rather than every RNA target in
    ``targets_rna.txt``; ``sqtl_gtex_track_count`` preserves that behavior.
    """

    TASK_CONFIG = {
        "sqtl": {"output": "sqtl_span", "statistic": "nDi"},
        "paqtl": {"output": "paqtl", "statistic": "COVR"},
        "ipaqtl": {"output": "ipaqtl", "statistic": "COVR"},
    }

    def __init__(
        self,
        experiment_dir: str | Path,
        task: str,
        output_name: str | None = None,
        statistic: str | None = None,
        name: str = "borzoi",
        sqtl_gtex_track_count: int = 89,
    ) -> None:
        if task not in self.TASK_CONFIG:
            raise ValueError(
                f"task must be one of {sorted(self.TASK_CONFIG)}; got {task!r}"
            )
        defaults = self.TASK_CONFIG[task]
        self.experiment_dir = Path(experiment_dir).expanduser().resolve()
        self.task = task
        self.output_name = output_name or defaults["output"]
        self.statistic = statistic or defaults["statistic"]
        self.name = name
        self.sqtl_gtex_track_count = int(sqtl_gtex_track_count)
        if self.sqtl_gtex_track_count < 1:
            raise ValueError("sqtl_gtex_track_count must be positive")

    def identity_payload(self) -> Mapping[str, Any]:
        return {
            "class": f"{type(self).__module__}:{type(self).__qualname__}",
            "experiment_dir": str(self.experiment_dir),
            "task": self.task,
            "output_name": self.output_name,
            "statistic": self.statistic,
            "sqtl_gtex_track_count": self.sqtl_gtex_track_count,
        }

    def _select_task_tracks(self, values: np.ndarray, path: Path) -> np.ndarray:
        """Select the output tracks used by the paper-compatible task score."""
        if self.task != "sqtl" or values.ndim != 2:
            return values
        if values.shape[1] < self.sqtl_gtex_track_count:
            raise ValueError(
                f"{path} has {values.shape[1]} {self.statistic} tracks; "
                f"Borzoi sQTL requires the final "
                f"{self.sqtl_gtex_track_count} GTEx RNA tracks"
            )
        return values[:, -self.sqtl_gtex_track_count :]

    def _files(self, split: str) -> list[Path]:
        relative = Path(self.output_name) / f"merge_{split}" / "sed.h5"
        candidates = (
            self.experiment_dir / "ensemble" / relative,
            self.experiment_dir / relative,
            self.experiment_dir / f"merge_{split}" / "sed.h5",
        )
        for candidate in candidates:
            if candidate.is_file():
                return [candidate]
        folds = sorted(self.experiment_dir.glob(f"f0c*/{relative}"))
        if folds:
            return folds
        tried = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"No completed Borzoi {self.task} {split} artifact; tried {tried} "
            f"and f0c*/{relative}"
        )

    def _read_split(
        self, split: str, records_by_id: Mapping[str, Any]
    ) -> pd.DataFrame:
        files = self._files(split)
        score_sum: np.ndarray | None = None
        first_meta: dict[str, np.ndarray] | None = None
        n_tracks = 1
        for path in files:
            with h5py.File(path, "r") as scores:
                if self.statistic not in scores:
                    raise KeyError(
                        f"{path} has no {self.statistic!r}; keys={list(scores.keys())}"
                    )
                current = np.asarray(scores[self.statistic], dtype=np.float32)
                if current.ndim not in {1, 2}:
                    raise ValueError(
                        f"Unexpected {self.statistic} shape in {path}: {current.shape}"
                    )
                current = self._select_task_tracks(current, path)
                n_tracks = current.shape[1] if current.ndim == 2 else 1
                meta = {
                    "snp": _decode(scores["snp"][:]),
                    "si": np.asarray(scores["si"][:], dtype=np.int64),
                    "gene": _decode(scores["gene"][:]),
                }
                if first_meta is None:
                    first_meta = meta
                    score_sum = current
                else:
                    if current.shape != score_sum.shape:
                        raise ValueError(f"Score shape mismatch in {path}")
                    for key, values in first_meta.items():
                        if not np.array_equal(meta[key], values):
                            raise ValueError(f"Row metadata mismatch ({key}) in {path}")
                    score_sum += current

        assert first_meta is not None and score_sum is not None
        averaged = score_sum / len(files)
        scalar = np.nanmean(averaged, axis=1) if averaged.ndim == 2 else averaged
        row_ids = first_meta["snp"][first_meta["si"]]
        rows = []
        for variant_id, gene_id, score in zip(
            row_ids, first_meta["gene"], scalar, strict=True
        ):
            record = records_by_id.get(str(variant_id))
            if record is None:
                continue
            rows.append(
                {
                    "variant_key": variant_key(record),
                    "gene_id": str(gene_id).split(".", 1)[0],
                    "tissue": "",
                    "score": float(score),
                    "n_tracks": n_tracks,
                }
            )
        table = pd.DataFrame(
            rows, columns=["variant_key", "gene_id", "tissue", "score", "n_tracks"]
        )
        if table.empty:
            return table
        return (
            table.groupby(
                ["variant_key", "gene_id", "tissue"], sort=False, as_index=False
            )
            .agg(score=("score", "mean"), n_tracks=("n_tracks", "max"))
        )

    def predict(
        self,
        dataset: InferenceDataset,
        store: PredictionStore,
        context: BenchmarkContext,
    ) -> None:
        del context
        if dataset.task != self.task:
            raise ValueError(
                f"Configured Borzoi adapter is for {self.task}, got {dataset.task}"
            )
        pending = {
            variant_key(record): record
            for record in dataset.variants
            if not store.completed(record)
        }
        print(
            f"[predict/{self.name}/{self.task}] variants={len(dataset.variants)} "
            f"cached={len(dataset.variants) - len(pending)} pending={len(pending)}",
            flush=True,
        )
        if not pending:
            return
        by_id = {record.variant_id: record for record in pending.values()}
        started = time.time()
        tables = [self._read_split(split, by_id) for split in ("pos", "neg")]
        predictions = pd.concat(tables, ignore_index=True)
        elapsed = (time.time() - started) / max(len(pending), 1)
        for key, record in pending.items():
            store.save(
                record,
                predictions[predictions.variant_key.eq(key)],
                elapsed_seconds=elapsed,
            )
