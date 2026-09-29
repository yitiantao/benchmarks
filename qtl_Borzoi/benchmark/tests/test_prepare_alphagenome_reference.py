from __future__ import annotations

import unittest

import pandas as pd

from scripts.prepare_alphagenome_reference import (
    extract_splice_sites,
    repair_missing_gene_rows,
)


class PrepareAlphaGenomeReferenceTest(unittest.TestCase):
    def test_repair_missing_gene_rows_reconstructs_parent_metadata(self) -> None:
        frame = pd.DataFrame(
            {
                "Chromosome": ["chr1", "chr1", "chr2"],
                "Source": ["test"] * 3,
                "Feature": ["transcript", "exon", "gene"],
                "Start": [20, 30, 100],
                "End": [80, 50, 200],
                "Score": ["."] * 3,
                "Strand": ["+", "+", "-"],
                "Frame": ["."] * 3,
                "gene_id": ["ORPHAN.1", "ORPHAN.1", "KNOWN.1"],
                "gene_type": ["protein_coding"] * 3,
                "gene_name": ["ORPHAN", "ORPHAN", "KNOWN"],
                "transcript_id": ["TX1.1", "TX1.1", None],
            }
        )

        repaired, gene_ids = repair_missing_gene_rows(frame)

        self.assertEqual(gene_ids, ("ORPHAN.1",))
        gene = repaired.loc[
            repaired.Feature.eq("gene") & repaired.gene_id.eq("ORPHAN.1")
        ].iloc[0]
        self.assertEqual((gene.Start, gene.End), (20, 80))
        self.assertEqual(gene.gene_name, "ORPHAN")
        self.assertTrue(pd.isna(gene.transcript_id))

    def test_extract_splice_sites_uses_transcript_exon_boundaries(self) -> None:
        frame = pd.DataFrame(
            {
                "Chromosome": ["chr1"] * 5,
                "Start": [10, 30, 50, 70, 5],
                "End": [20, 40, 60, 80, 90],
                "Strand": ["+", "+", "-", "-", "+"],
                "Feature": ["exon", "exon", "exon", "exon", "gene"],
                "transcript_id": ["tx1", "tx1", "tx2", "tx2", None],
            }
        )

        starts, ends = extract_splice_sites(frame)

        self.assertEqual(
            starts.to_dict("records"),
            [
                {"Chromosome": "chr1", "Start": 20, "Strand": "+"},
                {"Chromosome": "chr1", "Start": 60, "Strand": "-"},
            ],
        )
        self.assertEqual(
            ends.to_dict("records"),
            [
                {"Chromosome": "chr1", "End": 30, "Strand": "+"},
                {"Chromosome": "chr1", "End": 70, "Strand": "-"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
