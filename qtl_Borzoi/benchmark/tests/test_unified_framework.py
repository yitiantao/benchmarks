from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from qtl_benchmark.data.gtex_eqtl import GTExEQTLDataModule
from qtl_benchmark.evaluation.gtex_eqtl import GTExEQTLEvaluator
from qtl_benchmark.interfaces import TidyVariantModelAdapter
from qtl_benchmark.model_base import QTLModel, VariantRecord
from qtl_benchmark.runner import BenchmarkRunner


HEADER = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"


def write_gzip(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.write(value)


class FakeStructuredModel(QTLModel):
    name = "fake-structured"
    supported_tasks = frozenset({"eqtl"})

    def score_variant_table(self, record: VariantRecord) -> pd.DataFrame:
        if record.split != "test":
            raise AssertionError("Model adapter received a positive/negative label")
        score = {"p1": 2.0, "p2": 1.0, "n1": 0.2, "n2": 0.1}[record.variant_id]
        gene = {"p1": "G1.2", "p2": "G2.1", "n1": "N1", "n2": "N2"}[record.variant_id]
        return pd.DataFrame(
            {
                "gene_id": [gene, gene],
                "gtex_tissue": ["Liver", "Liver"],
                "raw_score": [score, score],
            }
        )

    def predict_batch(self, task, variants, context):
        raise AssertionError("The structured adapter should use score_variant_table")


class FakeBatchedStructuredModel(FakeStructuredModel):
    name = "fake-batched-structured"

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def score_variant_table(self, record: VariantRecord) -> pd.DataFrame:
        raise AssertionError("The batched adapter should use score_variant_tables")

    def score_variant_tables(self, records) -> list[pd.DataFrame]:
        self.batches.append([record.variant_id for record in records])
        return [FakeStructuredModel.score_variant_table(self, record) for record in records]


class UnifiedFrameworkTest(unittest.TestCase):
    def _data(self, root: Path) -> None:
        records = {
            "pos": [
                "chr1\t101\tp1\tA\tG\t.\t.\t.\n",
                "chr1\t102\tp2\tA\tC\t.\t.\t.\n",
            ],
            "neg": [
                "chr1\t103\tn1\tA\tT\t.\t.\t.\n",
                "chr1\t104\tn2\tA\tC\t.\t.\t.\n",
            ],
        }
        for split, lines in records.items():
            write_gzip(root / f"Liver_{split}.vcf.gz", HEADER + "".join(lines))
        causal = pd.DataFrame(
            {
                "variant": ["p1", "p2"],
                "gene": ["G1", "G2"],
                "pip": [0.99, 0.95],
                "beta_posterior": [-2.0, 1.0],
                "allele1": ["G", "A"],
            }
        )
        causal.to_csv(root / "tables/Liver.tsv.gz", sep="\t")

    def test_eqtl_data_reports_an_empty_input_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data = GTExEQTLDataModule(temp_dir, all_tissues=True)

            with self.assertRaisesRegex(
                FileNotFoundError, "No GTEx tissue VCFs found"
            ):
                data.load()

    def test_one_runner_shares_data_and_evaluation_across_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            (data_dir / "tables").mkdir(parents=True)
            self._data(data_dir)
            data = GTExEQTLDataModule(data_dir, tissues=["Liver"])
            model = TidyVariantModelAdapter(FakeStructuredModel())
            runner = BenchmarkRunner(
                data,
                model,
                GTExEQTLEvaluator(min_variants=0),
                root / "outputs",
            )

            captured = io.StringIO()
            with redirect_stdout(captured):
                metrics = runner.run()
            run_dir = root / "outputs/fake-structured/gtex-eqtl"

            self.assertEqual(metrics.loc[0, "auroc_sign"], 1.0)
            self.assertEqual(metrics.loc[0, "auroc_class"], 1.0)
            predictions = pd.read_csv(run_dir / "predictions.tsv.gz", sep="\t")
            self.assertEqual(len(predictions), 4)
            self.assertEqual(predictions.n_tracks.tolist(), [2, 2, 2, 2])
            manifest = json.loads((run_dir / "run.json").read_text())
            self.assertEqual(manifest["dataset"]["variants"], 4)
            self.assertEqual(manifest["dataset"]["targets"], 2)
            metric_log = captured.getvalue()
            self.assertIn("[metrics] model=fake-structured", metric_log)
            self.assertIn('"auroc_sign":1.0', metric_log)
            self.assertIn('"spearmanr":', metric_log)
            self.assertIn('"pearsonr":1.0', metric_log)
            self.assertIn('"auroc_class":1.0', metric_log)
            logged_metrics = json.loads((run_dir / "metrics.json").read_text())
            self.assertEqual(logged_metrics["columns"], metrics.columns.tolist())
            self.assertEqual(logged_metrics["records"][0]["tissue"], "Liver")

            second = runner.run()
            self.assertEqual(second.loc[0, "auroc_class"], 1.0)

    def test_prediction_shards_share_one_cache_then_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            (data_dir / "tables").mkdir(parents=True)
            self._data(data_dir)
            data = GTExEQTLDataModule(data_dir, tissues=["Liver"])
            model = TidyVariantModelAdapter(FakeStructuredModel())
            output_dir = root / "outputs"

            def run_shard(shard_index: int):
                return BenchmarkRunner(
                    data,
                    model,
                    GTExEQTLEvaluator(min_variants=0),
                    output_dir,
                    options={"num_shards": 2, "shard_index": shard_index},
                ).run(evaluate=False)

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(run_shard, range(2)))
            self.assertEqual(results, [None, None])

            run_dir = output_dir / "fake-structured/gtex-eqtl"
            self.assertFalse((run_dir / "predictions.tsv.gz").exists())
            metrics = BenchmarkRunner(
                data,
                model,
                GTExEQTLEvaluator(min_variants=0),
                output_dir,
            ).run(predict=False)
            self.assertEqual(metrics.loc[0, "auroc_class"], 1.0)
            predictions = pd.read_csv(run_dir / "predictions.tsv.gz", sep="\t")
            self.assertEqual(len(predictions), 4)

    def test_tidy_adapter_batches_without_changing_cache_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            (data_dir / "tables").mkdir(parents=True)
            self._data(data_dir)
            data = GTExEQTLDataModule(data_dir, tissues=["Liver"])
            local_model = FakeBatchedStructuredModel()
            batched = TidyVariantModelAdapter(local_model, batch_size=3)
            unbatched = TidyVariantModelAdapter(local_model, batch_size=1)

            self.assertEqual(batched.fingerprint(), unbatched.fingerprint())
            BenchmarkRunner(
                data,
                batched,
                GTExEQTLEvaluator(min_variants=0),
                root / "outputs",
            ).run(evaluate=False)

            self.assertEqual(local_model.batches, [["p1", "p2", "n1"], ["n2"]])


if __name__ == "__main__":
    unittest.main()
