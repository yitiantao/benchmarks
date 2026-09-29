from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd

from qtl_benchmark.compare_eqtl_models import (
    _alphagenome_group_source_pairs,
    _validate_subset_manifest,
    group_alphagenome_to_borzoi_tissues,
    input_statistics,
    read_prediction_pairs,
)


class CompareEQTLModelsTest(unittest.TestCase):
    def test_input_statistics_report_shared_counts_and_stable_hashes(self) -> None:
        dataset = SimpleNamespace(
            tissues=("Liver",),
            memberships=pd.DataFrame(
                {
                    "variant_key": ["v1", "v2"],
                    "variant_id": ["p1", "n1"],
                    "split": ["pos", "neg"],
                    "tissue": ["Liver", "Liver"],
                }
            ),
            targets=pd.DataFrame(
                {
                    "variant_key": ["v1"],
                    "variant_id": ["p1"],
                    "gene_id": ["G1"],
                    "tissue": ["Liver"],
                    "effect_size": [1.0],
                    "effect_allele": ["G"],
                    "ref": ["A"],
                }
            ),
        )

        observed = input_statistics(dataset)

        liver = observed[observed.tissue == "Liver"].iloc[0]
        self.assertEqual(liver.n_pos, 1)
        self.assertEqual(liver.n_neg, 1)
        self.assertEqual(liver.n_unique_variants, 2)
        self.assertEqual(liver.n_targets, 1)
        self.assertEqual(len(liver.membership_sha256), 64)
        self.assertEqual(len(liver.target_sha256), 64)
        self.assertIn("__ALL__", observed.tissue.tolist())

    def test_subset_manifest_scope_and_limit_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "subset_manifest.json").write_text(
                '{"limit_scope":"per-tissue",'
                '"max_variants_per_tissue_per_split":2000}'
            )

            manifest = _validate_subset_manifest(root, "per-tissue", 2000)

            self.assertEqual(manifest["limit_scope"], "per-tissue")
            with self.assertRaisesRegex(RuntimeError, "scope mismatch"):
                _validate_subset_manifest(root, "global", 2000)

    def test_alphagenome_scores_are_grouped_with_track_weights(self) -> None:
        predictions = pd.DataFrame(
            {
                "variant_key": ["chr1:1:A:G", "chr1:1:A:G"],
                "gene_id": ["G1", "G1"],
                "tissue": ["Brain_Amygdala", "Brain_Cortex"],
                "score": [1.0, 3.0],
                "n_tracks": [1, 3],
            }
        )

        grouped = group_alphagenome_to_borzoi_tissues(
            predictions, ["Brain_Amygdala", "Brain_Cortex"]
        ).sort_values("tissue")

        self.assertEqual(
            grouped.tissue.tolist(), ["Brain_Amygdala", "Brain_Cortex"]
        )
        self.assertEqual(grouped.score.tolist(), [2.5, 2.5])
        self.assertEqual(grouped.n_tracks.tolist(), [4, 4])
        self.assertEqual(predictions.score.tolist(), [1.0, 3.0])

    def test_group_source_pairs_expand_only_needed_groups(self) -> None:
        pairs = {
            ("v1", "Brain_Cortex"),
            ("v2", "Liver"),
        }

        expanded = _alphagenome_group_source_pairs(pairs)

        self.assertIn(("v1", "Brain_Amygdala"), expanded)
        self.assertIn(("v1", "Brain_Cortex"), expanded)
        self.assertNotIn(("v1", "Liver"), expanded)
        self.assertEqual({pair for pair in expanded if pair[0] == "v2"}, {("v2", "Liver")})

    def test_prediction_store_reader_keeps_only_requested_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "predictions.sqlite"
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """CREATE TABLE predictions (
                    variant_key TEXT NOT NULL,
                    gene_id TEXT NOT NULL,
                    tissue TEXT NOT NULL,
                    score REAL NOT NULL,
                    n_tracks INTEGER NOT NULL,
                    PRIMARY KEY (variant_key, gene_id, tissue))"""
                )
                connection.executemany(
                    "INSERT INTO predictions VALUES (?, ?, ?, ?, ?)",
                    [
                        ("v1", "G1", "Liver", 1.0, 1),
                        ("v1", "G1", "Lung", 2.0, 1),
                        ("v2", "G2", "Liver", 3.0, 1),
                    ],
                )

            result = read_prediction_pairs(
                path, {("v1", "Liver"), ("v2", "Liver")}
            )

            self.assertEqual(result.variant_key.tolist(), ["v1", "v2"])
            self.assertEqual(result.tissue.tolist(), ["Liver", "Liver"])
            self.assertEqual(result.score.tolist(), [1.0, 3.0])


if __name__ == "__main__":
    unittest.main()
