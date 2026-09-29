from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from qtl_benchmark.model_base import VariantRecord
from qtl_benchmark.models.dna_fm import DNAFMModel


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DNAFMTest(unittest.TestCase):
    def _model(self, root: Path, *, expected_hash: bool = True) -> DNAFMModel:
        source = root / "source/src/dna_ntv3"
        source.mkdir(parents=True)
        checkpoint = root / "checkpoint/model.safetensors"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"test DNA-FM checkpoint")
        manifest = root / "manifest.json"
        tracks = [
            {
                "track_id": "liver-a",
                "dataset": "gtex",
                "tissue": "liver",
                "is_rna": True,
            },
            {
                "track_id": "liver-b",
                "dataset": "gtex",
                "tissue": "right lobe of liver",
                "is_rna": True,
            },
            {
                "track_id": "blood-vessel",
                "dataset": "gtex",
                "tissue": "blood_vessel",
                "is_rna": True,
            },
            {
                "track_id": "blood",
                "dataset": "gtex",
                "tissue": "blood",
                "is_rna": True,
            },
            {
                "track_id": "encode-liver",
                "dataset": "encode_v3",
                "tissue": "liver",
                "is_rna": True,
            },
            {
                "track_id": "encode-heart",
                "dataset": "encode_v3",
                "tissue": "heart",
                "is_rna": True,
            },
        ]
        manifest.write_text(
            json.dumps({"species": {"human": {"tracks": tracks}}})
        )
        config = root / "config.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model": {"architecture": {}},
                    "posttraining": {
                        "species": ["human"],
                        "functional_tracks": {
                            "human": [track["track_id"] for track in tracks]
                        },
                        "annotation_labels": ["splice_donor", "splice_acceptor"],
                    },
                    "data": {
                        "canonical_manifest": str(manifest),
                        "canonical_manifest_sha256": digest(manifest),
                    },
                }
            )
        )
        fasta = root / "reference.fa"
        fasta.write_text(">chr1\nA\n")
        Path(str(fasta) + ".fai").write_text("chr1\t1\t6\t1\t2\n")
        gtf = root / "genes.gtf"
        gtf.write_text('chr1\tt\tgene\t1\t1\t.\t+\t.\tgene_id "G1";\n')
        return DNAFMModel(
            source_root=str(root / "source"),
            checkpoint_path=str(checkpoint.parent),
            checkpoint_sha256=digest(checkpoint) if expected_hash else None,
            run_config_path=str(config),
            run_config_sha256=digest(config),
            fasta_path=str(fasta),
            gtf_path=str(gtf),
            tissue_keywords={"Liver": "liver", "Whole_Blood": "blood"},
            sequence_length=128,
        )

    def test_checkpoint_directory_and_content_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = self._model(Path(temp_dir))
            self.assertEqual(model.mixed_precision, "float32")
            identity = model.identity_metadata()
            self.assertEqual(
                identity["resolved_checkpoint_sha256"], model.checkpoint_sha256
            )
            self.assertTrue(model.checkpoint_path.name == "model.safetensors")

    def test_checkpoint_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = self._model(root, expected_hash=False)
            with self.assertRaisesRegex(ValueError, "checkpoint SHA-256 mismatch"):
                DNAFMModel(
                    source_root=str(root / "source"),
                    checkpoint_path=str(model.checkpoint_path),
                    checkpoint_sha256="0" * 64,
                    run_config_path=str(model.run_config_path),
                    fasta_path=str(model.fasta_path),
                    gtf_path=str(model.gtf_path),
                    sequence_length=128,
                )

    def test_gtex_track_selection_and_blood_vessel_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = self._model(Path(temp_dir))
            model.tissue_keywords = {"Liver": "liver", "Whole_Blood": "blood"}
            model._load_training_metadata()
            self.assertEqual(
                model._selected_track_ids,
                ("liver-a", "liver-b", "blood"),
            )
            self.assertEqual(model._tissue_track_indices["Liver"], (0, 1))
            self.assertEqual(model._tissue_track_indices["Whole_Blood"], (2,))

    def test_missing_gtex_tissue_uses_explicit_encode_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = self._model(Path(temp_dir))
            model.tissue_keywords = {"Heart_Left_Ventricle": "heart"}
            model._load_training_metadata()
            self.assertEqual(model._selected_track_ids, ("encode-heart",))
            self.assertEqual(
                model._tissue_track_sources["Heart_Left_Ventricle"], "encode_v3"
            )

    def test_reference_orientation_matches_existing_ntv3_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = self._model(Path(temp_dir))
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

    def test_signed_max_abs_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = self._model(Path(temp_dir))
            self.assertEqual(model._reduce(np.asarray([0.1, -3.0, 2.0])), -3.0)


if __name__ == "__main__":
    unittest.main()
