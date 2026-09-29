from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from qtl_benchmark.alphagenome_eqtl import (
    ScoreStore,
    _variant_key,
    evaluate_tissue,
    score_variants,
    summarize_score_table,
)
from qtl_benchmark.model_base import QTLModel, TaskContext, VariantRecord
from qtl_benchmark.models.alphagenome import AlphaGenomeModel


def variant(variant_id: str = "v1") -> VariantRecord:
    return VariantRecord(
        task="eqtl",
        split="pos",
        chrom="chr1",
        pos=101,
        variant_id=variant_id,
        ref="A",
        alt="G",
        tissue="Liver",
    )


class FakeTidyModel(QTLModel):
    name = "fake-tidy"
    supported_tasks = frozenset({"eqtl"})

    def __init__(self) -> None:
        self.setup_calls = 0
        self.teardown_calls = 0
        self.score_calls = 0

    def setup_task(self, context: TaskContext) -> None:
        self.setup_calls += 1

    def teardown_task(self, context: TaskContext) -> None:
        self.teardown_calls += 1

    def score_variant_table(self, record: VariantRecord) -> pd.DataFrame:
        self.score_calls += 1
        return pd.DataFrame(
            {
                "gene_id": ["ENSG1.2", "ENSG1.2", "ENSG2.1"],
                "gtex_tissue": ["Liver", "Liver", "Whole_Blood"],
                "raw_score": [1.0, 3.0, -4.0],
            }
        )

    def predict_batch(self, task, variants, context):
        raise AssertionError("The tissue benchmark must use score_variant_table")


class AlphaGenomeEqtlTest(unittest.TestCase):
    def test_empty_tidy_score_table_is_a_valid_zero_row_prediction(self) -> None:
        model = AlphaGenomeModel(
            checkpoint_path="checkpoint",
            fasta_path="reference.fa",
            gtf_feather_path="genes.feather",
            splice_site_starts_feather_path="starts.feather",
            splice_site_ends_feather_path="ends.feather",
        )

        observed = model._filter_scores(pd.DataFrame(), variant())

        self.assertTrue(observed.empty)

    def test_cache_rejects_a_different_model_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ScoreStore(Path(temp_dir) / "scores.sqlite")
            try:
                store.ensure_identity("model-a")
                store.save(
                    variant(),
                    pd.DataFrame(
                        {
                            "gene_id": ["ENSG1"],
                            "gtex_tissue": ["Liver"],
                            "score": [1.0],
                            "n_tracks": [1],
                        }
                    ),
                    0.1,
                )
                with self.assertRaisesRegex(RuntimeError, "different model/config"):
                    store.ensure_identity("model-b")
            finally:
                store.close()

    def test_empty_cache_accepts_updated_model_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ScoreStore(Path(temp_dir) / "scores.sqlite")
            try:
                store.ensure_identity("model-a")
                store.ensure_identity("model-b")
                identity = store.connection.execute(
                    "SELECT value FROM metadata WHERE key = 'model_identity'"
                ).fetchone()[0]
            finally:
                store.close()
            self.assertEqual(identity, "model-b")

    def test_summarize_averages_tracks_by_gene_and_tissue(self) -> None:
        table = pd.DataFrame(
            {
                "gene_id": ["ENSG1.2", "ENSG1.2", "ENSG2.1", "ENSG3"],
                "gtex_tissue": ["Liver", "Liver", "Liver", None],
                "raw_score": [1.0, 3.0, -2.0, 99.0],
            }
        )
        observed = summarize_score_table(table)
        self.assertEqual(observed.to_dict("records"), [
            {"gene_id": "ENSG1", "gtex_tissue": "Liver", "score": 2.0, "n_tracks": 2},
            {"gene_id": "ENSG2", "gtex_tissue": "Liver", "score": -2.0, "n_tracks": 1},
        ])

    def test_scoring_uses_cache_on_second_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            context = TaskContext("eqtl", output_dir, output_dir)
            store = ScoreStore(output_dir / "scores.sqlite")
            model = FakeTidyModel()
            record = variant()
            try:
                score_variants(model, store, [record], context)
                score_variants(model, store, [record], context)
                scores = store.tissue_scores("Liver")
            finally:
                store.close()

            self.assertEqual(model.score_calls, 1)
            self.assertEqual(model.setup_calls, 1)
            self.assertEqual(model.teardown_calls, 1)
            self.assertEqual(scores.loc[0, "variant_key"], _variant_key(record))
            self.assertEqual(scores.loc[0, "gene_id"], "ENSG1")
            self.assertEqual(scores.loc[0, "score"], 2.0)
            self.assertEqual(scores.loc[0, "n_tracks"], 2)

    def test_tissue_metrics_match_borzoi_orientation_and_classification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "tables").mkdir()
            records = [
                VariantRecord("eqtl", "pos", "chr1", 101, "p1", "A", "G"),
                VariantRecord("eqtl", "pos", "chr1", 102, "p2", "A", "C"),
                VariantRecord("eqtl", "neg", "chr1", 103, "n1", "A", "T"),
                VariantRecord("eqtl", "neg", "chr1", 104, "n2", "A", "C"),
            ]
            causal = pd.DataFrame(
                {
                    "variant": ["p1", "p2"],
                    "gene": ["ENSG1.1", "ENSG2.2"],
                    "pip": [0.99, 0.95],
                    "beta_posterior": [-1.0, 1.0],
                    "allele1": ["G", "A"],
                }
            )
            causal.to_csv(root / "tables" / "Liver.tsv.gz", sep="\t")

            store = ScoreStore(root / "scores.sqlite")
            try:
                for record, score, gene in zip(
                    records, [2.0, 1.0, 0.1, 0.2], ["ENSG1", "ENSG2", "ENSG3", "ENSG4"]
                ):
                    store.save(
                        record,
                        pd.DataFrame(
                            {
                                "gene_id": [gene],
                                "gtex_tissue": ["Liver"],
                                "score": [score],
                                "n_tracks": [1],
                            }
                        ),
                        0.1,
                    )
                metrics = evaluate_tissue(
                    store,
                    root,
                    root / "output",
                    "Liver",
                    records[:2],
                    records[2:],
                    pip_threshold=0.9,
                    min_variants=0,
                )
                coefficients = pd.read_csv(
                    root / "output/predictions/Liver_coefficients.tsv.gz", sep="\t"
                )
            finally:
                store.close()

            self.assertEqual(coefficients.score.tolist(), [-2.0, 1.0])
            self.assertEqual(metrics["auroc_sign"], 1.0)
            self.assertAlmostEqual(metrics["spearmanr"], 1.0)
            self.assertAlmostEqual(metrics["pearsonr"], 1.0)
            self.assertEqual(metrics["auroc_class"], 1.0)


if __name__ == "__main__":
    unittest.main()
