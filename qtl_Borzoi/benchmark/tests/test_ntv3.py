from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd

from qtl_benchmark.core import BenchmarkContext, InferenceDataset
from qtl_benchmark.model_base import TaskContext, VariantRecord
from qtl_benchmark.models.ntv3 import (
    GeneInterval,
    GeneIndex,
    NTv3Model,
    NTv3ModelAdapter,
    _select_broad_eqtl_tracks,
    _signed_max_abs,
)
from qtl_benchmark.predictions import PredictionStore


class FakeBatchedNTv3:
    name = "fake-ntv3"

    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.setup_options = None
        self.teardown_calls = 0

    def setup_task(self, context: TaskContext) -> None:
        self.setup_options = context.options

    def score_variant_tables(self, records):
        self.batches.append([record.variant_id for record in records])
        return [
            pd.DataFrame(
                {
                    "gene_id": [record.gene_id or "ENSG1"],
                    "tissue": [record.tissue or ""],
                    "score": [float(record.pos)],
                }
            )
            for record in records
        ]

    def teardown_task(self, context: TaskContext) -> None:
        self.teardown_calls += 1


class NTv3Test(unittest.TestCase):
    def test_signed_max_abs_preserves_direction(self) -> None:
        self.assertEqual(_signed_max_abs(np.array([0.2, -3.0, 2.0])), -3.0)
        self.assertEqual(_signed_max_abs(np.array([])), 0.0)

    def test_gene_index_uses_zero_based_half_open_coordinates(self) -> None:
        gtf = (
            'chr1\ttest\tgene\t101\t200\t.\t+\t.\tgene_id "ENSG1.2";\n'
            'chr1\ttest\tgene\t301\t400\t.\t-\t.\tgene_id "ENSG2";\n'
            'chr1\ttest\texon\t500\t600\t.\t+\t.\tgene_id "IGNORED";\n'
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "genes.gtf"
            path.write_text(gtf)
            index = GeneIndex.from_gtf(path)
        self.assertEqual(
            [gene.gene_id for gene in index.overlapping("chr1", 199, 301)],
            ["ENSG1", "ENSG2"],
        )
        self.assertEqual(index.overlapping("chr2", 0, 1000), ())

    def test_config_validation_and_checkpoint_detection(self) -> None:
        common = {
            "fasta_path": "reference.fa",
            "gtf_path": "genes.gtf",
            "root_dir": ".",
        }
        pre = NTv3Model(model_id="InstaDeepAI/NTv3_8M_pre", **common)
        post = NTv3Model(model_id="InstaDeepAI/NTv3_100M_post", **common)
        self.assertEqual(pre.checkpoint_type, "pre")
        self.assertEqual(post.checkpoint_type, "post")
        with self.assertRaisesRegex(ValueError, "divisible by 128"):
            NTv3Model(
                model_id="InstaDeepAI/NTv3_8M_pre",
                sequence_length=1000,
                **common,
            )

    def test_auto_dtype_respects_checkpoint_config(self) -> None:
        model = NTv3Model(
            model_id="InstaDeepAI/NTv3_650M_post",
            fasta_path="reference.fa",
            gtf_path="genes.gtf",
            dtype="auto",
        )
        fake_torch = SimpleNamespace(float32="fp32", float16="fp16", bfloat16="bf16")
        device = SimpleNamespace(type="cuda")
        self.assertEqual(
            model._select_dtype(fake_torch, device, SimpleNamespace(dtype="float32")),
            "fp32",
        )
        self.assertEqual(
            model._select_dtype(fake_torch, device, SimpleNamespace(dtype="bfloat16")),
            "bf16",
        )
        self.assertEqual(
            model._select_dtype(fake_torch, device, SimpleNamespace()),
            "fp32",
        )

    def test_broad_eqtl_tracks_prefer_gtex_and_fallback_to_encode(self) -> None:
        metadata = pd.DataFrame(
            [
                {
                    "identifier": "GTEX-LIVER",
                    "file": "/rna/recount3/liver/GTEX-LIVER/coverage.w5",
                    "description": "RNA:liver",
                },
                {
                    "identifier": "GTEX-BLOOD",
                    "file": "/rna/recount3/blood/GTEX-BLOOD/coverage.w5",
                    "description": "RNA:blood",
                },
                {
                    "identifier": "GTEX-VESSEL",
                    "file": "/rna/recount3/blood_vessel/GTEX-VESSEL/coverage.w5",
                    "description": "RNA:blood_vessel",
                },
                {
                    "identifier": "GTEX-HEART-MISSING",
                    "file": "/rna/recount3/heart/GTEX-HEART-MISSING/coverage.w5",
                    "description": "RNA:heart",
                },
                {
                    "identifier": "ENCFF000AAA+",
                    "file": "/rna/encode/ENCSR000AAA/summary/coverage+.w5",
                    "description": "RNA:heart left ventricle tissue adult",
                },
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tracks.tsv"
            metadata.to_csv(path, sep="\t", index=False)
            tracks, indexes, sources = _select_broad_eqtl_tracks(
                path,
                (
                    "ENCSR000AAA_P",
                    "GTEX-LIVER",
                    "GTEX-BLOOD",
                    "GTEX-VESSEL",
                ),
                {
                    "Liver": "liver",
                    "Heart": "heart",
                    "Whole_Blood": "blood",
                    "Artery": "blood_vessel",
                },
            )
        self.assertEqual(
            tracks,
            (
                "ENCSR000AAA_P",
                "GTEX-LIVER",
                "GTEX-BLOOD",
                "GTEX-VESSEL",
            ),
        )
        self.assertEqual(indexes["Heart"], (0,))
        self.assertEqual(indexes["Liver"], (1,))
        self.assertEqual(indexes["Whole_Blood"], (2,))
        self.assertEqual(indexes["Artery"], (3,))
        self.assertEqual(sources["Heart"], "encode")
        self.assertEqual(sources["Liver"], "gtex")

    def test_broad_eqtl_rows_use_tissue_specific_track_groups(self) -> None:
        model = NTv3Model(
            model_id="InstaDeepAI/NTv3_100M_post",
            fasta_path="reference.fa",
            gtf_path="genes.gtf",
        )
        model._active_tissues = ("Liver", "Heart")
        model._tissue_head_indices = {"Liver": (0, 1), "Heart": (2,)}
        genes = (GeneInterval("chr1", 100, 102, "ENSG1"),)
        delta = np.array([[0.2, -0.8, 0.1], [0.3, 0.4, 1.5]])
        table = model._rows_for_broad_eqtl(genes, delta, prediction_start=100)
        scores = dict(zip(table.tissue, table.score, strict=True))
        self.assertEqual(scores, {"Liver": -0.8, "Heart": 1.5})

    def test_fasta_alt_match_uses_vcf_reference_orientation(self) -> None:
        model = NTv3Model(
            model_id="InstaDeepAI/NTv3_8M_pre",
            fasta_path="reference.fa",
            gtf_path="genes.gtf",
            sequence_length=128,
        )
        model._genome = {"chr1": "A" * 128}
        record = VariantRecord(
            task="eqtl",
            split="test",
            chrom="chr1",
            pos=65,
            variant_id="chr1_65_G_A_b38",
            ref="G",
            alt="A",
        )
        window = model._window(record)
        self.assertEqual(window.sequence[window.variant_offset], "G")
        self.assertEqual(model._fasta_alt_matches, 1)

    def test_adapter_batches_and_persists_predictions(self) -> None:
        variants = tuple(
            VariantRecord(
                task="sqtl",
                split="test",
                chrom="chr1",
                pos=100 + index,
                variant_id=f"v{index}",
                ref="A",
                alt="G",
                gene_id=f"ENSG{index}",
            )
            for index in range(3)
        )
        dataset = InferenceDataset(
            name="sqtl-test",
            task="sqtl",
            data_dir=Path("."),
            variants=variants,
            tissues=(),
        )
        model = FakeBatchedNTv3()
        adapter = NTv3ModelAdapter(model, {"name": model.name}, batch_size=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with PredictionStore(output_dir / "predictions.sqlite") as store:
                store.ensure_identity(adapter.fingerprint())
                adapter.predict(
                    dataset,
                    store,
                    BenchmarkContext(output_dir, {"num_shards": 1, "shard_index": 0}),
                )
                predictions = store.read()
        self.assertEqual(model.batches, [["v0", "v1"], ["v2"]])
        self.assertEqual(len(predictions), 3)
        self.assertEqual(predictions.score.tolist(), [100.0, 101.0, 102.0])
        self.assertEqual(model.setup_options["tissues"], ())
        self.assertEqual(model.teardown_calls, 1)


if __name__ == "__main__":
    unittest.main()
