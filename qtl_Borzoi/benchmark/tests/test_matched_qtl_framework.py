from __future__ import annotations

import gzip
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

from qtl_benchmark.core import BenchmarkContext
from qtl_benchmark.data.matched_qtl import MatchedQTLDataModule
from qtl_benchmark.evaluation.borzoi_sqtl import BorzoiSQTLEvaluator
from qtl_benchmark.evaluation.matched_qtl import MatchedQTLEvaluator
from qtl_benchmark.interfaces import TidyVariantModelAdapter
from qtl_benchmark.model_base import QTLModel, VariantRecord
from qtl_benchmark.models.borzoi import BorzoiQTLModelAdapter
from qtl_benchmark.predictions import PredictionStore


HEADER = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"


def write_vcf(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.write(HEADER + "".join(rows))


class EmptyTissueModel(QTLModel):
    name = "empty-tissue"
    supported_tasks = frozenset({"sqtl"})

    def score_variant_table(self, record):
        return pd.DataFrame({"gene_id": [record.gene_id], "raw_score": [1.0]})

    def predict_batch(self, task, variants, context):
        raise AssertionError("tidy adapter should be used")


class MatchedQTLFrameworkTest(unittest.TestCase):
    def _dataset(self, root: Path, task: str = "sqtl"):
        distance_key = "SD" if task == "sqtl" else "PD"
        write_vcf(
            root / task / "pos_merge.vcf.gz",
            [
                f"chr1\t101\tp1\tA\tG\t.\t.\tMT=G1.grp;{distance_key}=10;PI=p1\n",
                f"chr1\t102\tp2\tA\tC\t.\t.\tMT=G2.grp;{distance_key}=20;PI=p2\n",
            ],
        )
        write_vcf(
            root / task / "neg_merge.vcf.gz",
            [
                f"chr1\t103\tn1\tA\tT\t.\t.\tMT=N1.grp;{distance_key}=11;PI=p1\n",
                f"chr1\t104\tn2\tA\tC\t.\t.\tMT=N2.grp;{distance_key}=21;PI=p2\n",
            ],
        )
        return MatchedQTLDataModule(task, root).load()

    def test_loader_and_evaluator_preserve_matched_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = self._dataset(root)
            self.assertEqual(len(dataset.variants), 4)
            self.assertEqual(dataset.tissues, ())
            self.assertEqual(set(dataset.memberships.gene_id), {"G1", "G2", "N1", "N2"})

            predictions = dataset.memberships.loc[
                :, ["variant_key", "gene_id", "tissue"]
            ].copy()
            predictions["score"] = [4.0, 3.0, 0.2, 0.1]
            predictions["n_tracks"] = 1
            metrics = MatchedQTLEvaluator(
                repeats=3, sample_fraction=1.0
            ).evaluate(dataset, predictions, root / "out")
            self.assertTrue((metrics.matched_positive == 2).all())
            self.assertTrue((metrics.auroc_mean == 1.0).all())
            self.assertTrue((metrics.auprc_mean == 1.0).all())

    def test_borzoi_sqtl_evaluator_has_an_independent_boundary(self) -> None:
        self.assertFalse(issubclass(BorzoiSQTLEvaluator, MatchedQTLEvaluator))
        self.assertEqual(BorzoiSQTLEvaluator.name, "borzoi-sqtl")

    def test_borzoi_sqtl_evaluator_preserves_shared_metric_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = self._dataset(root)
            predictions = dataset.memberships.loc[
                :, ["variant_key", "gene_id", "tissue"]
            ].copy()
            predictions["score"] = [4.0, 3.0, 0.2, 0.1]
            predictions["n_tracks"] = 1
            shared = MatchedQTLEvaluator(
                repeats=3, sample_fraction=1.0
            ).evaluate(dataset, predictions, root / "shared")
            borzoi = BorzoiSQTLEvaluator(
                repeats=3, sample_fraction=1.0
            ).evaluate(dataset, predictions, root / "borzoi")
            pd.testing.assert_frame_equal(borzoi, shared)

    def test_smoke_limit_selects_real_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._dataset(root)
            dataset = MatchedQTLDataModule("sqtl", root, max_pairs=1).load()
            self.assertEqual(len(dataset.memberships), 2)
            negative = dataset.memberships[dataset.memberships.split.eq("neg")].iloc[0]
            positive_ids = set(
                dataset.memberships[dataset.memberships.split.eq("pos")].variant_id
            )
            self.assertIn(negative.matched_positive_id, positive_ids)

    def test_tidy_adapter_keeps_task_outputs_without_tissue(self) -> None:
        record = VariantRecord("sqtl", "test", "chr1", 101, "p1", "A", "G", "G1")
        table = TidyVariantModelAdapter._normalize_model_output(
            EmptyTissueModel().score_variant_table(record), record
        )
        self.assertEqual(table.loc[0, "gene_id"], "G1")
        self.assertEqual(table.loc[0, "tissue"], "")

    def test_borzoi_adapter_averages_folds_and_selected_sqtl_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = self._dataset(root)
            for fold, offset in (("f0c0", 0.0), ("f0c1", 2.0)):
                for split, ids, genes, base in (
                    ("pos", ["p1", "p2"], ["G1", "G2"], 4.0),
                    ("neg", ["n1", "n2"], ["N1", "N2"], 0.0),
                ):
                    path = root / "experiment" / fold / "sqtl_span" / f"merge_{split}" / "sed.h5"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with h5py.File(path, "w") as handle:
                        handle.create_dataset("snp", data=np.asarray(ids, dtype="S"))
                        handle.create_dataset("si", data=np.arange(2, dtype=np.int32))
                        handle.create_dataset("gene", data=np.asarray(genes, dtype="S"))
                        handle.create_dataset(
                            "nDi",
                            data=np.asarray(
                                [[100.0, base + offset, base + offset + 2]] * 2,
                                dtype=np.float32,
                            ),
                        )

            adapter = BorzoiQTLModelAdapter(
                root / "experiment", "sqtl", sqtl_gtex_track_count=2
            )
            context = BenchmarkContext(root / "output")
            with PredictionStore(root / "scores.sqlite") as store:
                adapter.predict(dataset.for_model(), store, context)
                observed = store.read().sort_values("variant_key").reset_index(drop=True)
            self.assertEqual(len(observed), 4)
            positive = observed[observed.gene_id.isin(["G1", "G2"])]
            negative = observed[observed.gene_id.isin(["N1", "N2"])]
            self.assertTrue((positive.score == 6.0).all())
            self.assertTrue((negative.score == 2.0).all())
            self.assertTrue((observed.n_tracks == 2).all())


if __name__ == "__main__":
    unittest.main()
