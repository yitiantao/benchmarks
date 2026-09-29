from __future__ import annotations

import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
SCRIPT = BENCHMARK_DIR / "qtl_benchmark" / "subset_vcfs.py"
VCF_HEADER = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"


def write_gzip(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.write(text)


def vcf_record(identifier: str, position: int) -> str:
    return f"chr1\t{position}\t{identifier}\tA\tG\t.\t.\t.\n"


class SubsetVcfsTest(unittest.TestCase):
    def test_per_tissue_limit_matches_first_n_records_in_each_tissue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "source"
            destination = temp / "destination"
            source.mkdir()
            for split, offset in (("pos", 0), ("neg", 10)):
                tissue_a = [
                    vcf_record(f"a{split}1", offset + 1),
                    vcf_record(f"a{split}2", offset + 2),
                ]
                tissue_b = [
                    vcf_record(f"b{split}1", offset + 3),
                    vcf_record(f"b{split}2", offset + 4),
                ]
                write_gzip(
                    source / f"A_{split}.vcf.gz",
                    VCF_HEADER + "".join(tissue_a),
                )
                write_gzip(
                    source / f"B_{split}.vcf.gz",
                    VCF_HEADER + "".join(tissue_b),
                )
                write_gzip(
                    source / f"{split}_merge.vcf.gz",
                    VCF_HEADER + "".join(tissue_a + tissue_b),
                )
            for tissue in ("A", "B"):
                write_gzip(
                    source / "tables" / f"{tissue}.tsv.gz",
                    "variant\tgene\tpip\n"
                    f"{tissue.lower()}pos1\tG1\t0.99\n"
                    f"{tissue.lower()}pos2\tG2\t0.99\n",
                )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--max-variants",
                    "1",
                    "--limit-scope",
                    "per-tissue",
                    str(source),
                    str(destination),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            merged = [
                line.split("\t")[2]
                for line in (destination / "pos_merge.vcf").read_text().splitlines()
                if not line.startswith("#")
            ]
            self.assertEqual(merged, ["apos1", "bpos1"])
            for tissue in ("A", "B"):
                tissue_records = [
                    line
                    for line in (destination / f"{tissue}_pos.vcf").read_text().splitlines()
                    if not line.startswith("#")
                ]
                self.assertEqual(len(tissue_records), 1)
                with gzip.open(
                    destination / "tables" / f"{tissue}.tsv.gz", "rt"
                ) as handle:
                    self.assertEqual(len(handle.read().splitlines()), 2)

            manifest = json.loads((destination / "subset_manifest.json").read_text())
            self.assertEqual(manifest["limit_scope"], "per-tissue")
            self.assertEqual(manifest["max_variants_per_tissue_per_split"], 1)
            self.assertEqual(manifest["selected_unique_ids"], {"neg": 2, "pos": 2})
            self.assertEqual(
                manifest["selected_unique_ids_by_tissue"]["pos"],
                {"A_pos": 1, "B_pos": 1},
            )

    def test_limit_is_global_and_tables_match_selected_positives(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "source"
            destination = temp / "destination"
            source.mkdir()

            tissue_a_pos = [vcf_record("a1", 1), vcf_record("a2", 2)]
            tissue_b_pos = [vcf_record("b1", 3), vcf_record("b2", 4)]
            tissue_a_neg = [vcf_record("na1", 11), vcf_record("na2", 12)]
            tissue_b_neg = [vcf_record("nb1", 13), vcf_record("nb2", 14)]
            files = {
                "A_pos.vcf.gz": tissue_a_pos,
                "B_pos.vcf.gz": tissue_b_pos,
                "A_neg.vcf.gz": tissue_a_neg,
                "B_neg.vcf.gz": tissue_b_neg,
                "pos_merge.vcf.gz": tissue_a_pos + tissue_b_pos,
                "neg_merge.vcf.gz": tissue_a_neg + tissue_b_neg,
            }
            for name, records in files.items():
                write_gzip(source / name, VCF_HEADER + "".join(records))

            table = (
                "variant\tgene\tpip\n"
                "a1\tGENE1\t0.99\n"
                "a2\tGENE2\t0.99\n"
                "b1\tGENE3\t0.99\n"
                "b2\tGENE4\t0.99\n"
            )
            write_gzip(source / "tables" / "A.tsv.gz", table)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--max-variants",
                    "2",
                    str(source),
                    str(destination),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            merged_records = [
                line
                for line in (destination / "pos_merge.vcf").read_text().splitlines()
                if not line.startswith("#")
            ]
            self.assertEqual([line.split("\t")[2] for line in merged_records], ["a1", "b1"])

            with gzip.open(destination / "tables" / "A.tsv.gz", "rt") as handle:
                retained = handle.read().splitlines()
            self.assertEqual(retained, ["variant\tgene\tpip", "a1\tGENE1\t0.99", "b1\tGENE3\t0.99"])

            manifest = json.loads((destination / "subset_manifest.json").read_text())
            self.assertEqual(manifest["merged_counts"], {"neg": 2, "pos": 2})
            self.assertEqual(manifest["selected_unique_ids"], {"neg": 2, "pos": 2})
            self.assertEqual(manifest["causal_table_row_counts"], {"A": 2})

    def test_zero_keeps_all_variants_without_copying_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "source"
            destination = temp / "destination"
            source.mkdir()
            for split in ("pos", "neg"):
                records = [vcf_record(f"{split}1", 1), vcf_record(f"{split}2", 2)]
                write_gzip(source / f"Tissue_{split}.vcf.gz", VCF_HEADER + "".join(records))
                write_gzip(source / f"{split}_merge.vcf.gz", VCF_HEADER + "".join(records))
            write_gzip(source / "tables" / "Tissue.tsv.gz", "variant\tgene\tpip\npos1\tG\t1\n")

            subprocess.run(
                [sys.executable, str(SCRIPT), "--max-variants", "0", str(source), str(destination)],
                check=True,
                capture_output=True,
                text=True,
            )

            manifest = json.loads((destination / "subset_manifest.json").read_text())
            self.assertEqual(manifest["merged_counts"], {"neg": 2, "pos": 2})
            self.assertEqual(manifest["selected_unique_ids"], {"neg": None, "pos": None})
            self.assertFalse((destination / "tables").exists())


if __name__ == "__main__":
    unittest.main()
