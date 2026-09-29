import csv
import fcntl
import gzip
import io
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from clinvar_benchmark.cli import _progress_line, _shard, score, shard_path, task_slug
from clinvar_benchmark.data import Reference, Variant, dataset_path, iter_variants, list_tasks
from clinvar_benchmark.evaluation import (evaluate, ranking_metrics,
                                          read_recoverable_predictions, write_predictions)


class FrameworkTests(unittest.TestCase):
    def test_real_dataset_task_selection(self):
        path = dataset_path("2026-02")
        self.assertIn("missense", list_tasks(path))
        rows = list(iter_variants(path, "missense", 3))
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({r.key for r in rows}), 3)
        self.assertTrue(all(r.group == "missense" for r in rows))

    def test_task_paths_and_shards(self):
        self.assertNotEqual(task_slug("+"), task_slug("-"))
        self.assertEqual(_shard("1:10:A:G", 4), _shard("1:10:A:G", 4))
        line = _progress_line("[shard 1/4]", 1, 2,
                              {"ok": 1, "no_score": 0, "error": 0},
                              {"Benign": 1, "Pathogenic": 0}, 0)
        self.assertIn("1/2 (50%)", line)
        self.assertIn("B=1 P=0", line)

    def test_window_coordinates(self):
        try:
            from pyfaidx import Fasta
        except ImportError:
            self.skipTest("pyfaidx is installed in the model environments")
        with tempfile.TemporaryDirectory() as temporary:
            fasta = Path(temporary) / "test.fa"
            fasta.write_text(">chr1\nAACCGGTTAACCGGTT\n")
            Fasta(str(fasta)).close()
            ref = Reference(fasta)
            variant = Variant("1", 5, "G", "A", 1, "all", 1)
            sequence, alternate, offset = ref.window(variant, 8)
            self.assertEqual(sequence[offset], "G")
            self.assertEqual(alternate[offset], "A")
            self.assertEqual(len(sequence), 8)

    def test_evaluation_and_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "data.csv.gz"
            with gzip.open(source, "wt", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["#CHROM", "POS", "REF", "ALT",
                                                              "ClinVar_label", "ClinVar_gold_stars",
                                                              "group: splice"])
                writer.writeheader()
                for pos, label in ((2, 0), (3, 1)):
                    writer.writerow({"#CHROM": "1", "POS": pos, "REF": "A", "ALT": "G",
                                     "ClinVar_label": label, "ClinVar_gold_stars": 1,
                                     "group: splice": 1})
            output = Path(temporary) / "scores.csv.gz"
            rows = []
            for pos, label, score in ((2, 0, 0.1), (3, 1, 0.9)):
                rows.append(dict(model="ntv3_100m_post", version="2026-02", task="splice",
                                 variant_key=f"1:{pos}:A:G", chrom="1", pos=pos,
                                 ref="A", alt="G", label=label, stars=1,
                                 score_name="position_llr", raw_score=-score, score=score,
                                 status="ok", error="", extra_scores_json="{}"))
            write_predictions(output, rows)
            result = evaluate([output], source, "2026-02", "splice", "ntv3_100m_post", 0, 1)
            self.assertEqual(result["n_expected"], 2)
            self.assertEqual(result["n_expected_benign"], 1)
            self.assertEqual(result["n_expected_pathogenic"], 1)
            self.assertEqual(result["auroc"], 1)
            self.assertEqual(result["auprc"], 1)
            self.assertEqual(ranking_metrics([(1, 0), (1, 1)])["auroc"], .5)

    def test_progress_and_label_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "data.csv"
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["#CHROM", "POS", "REF", "ALT",
                                                              "ClinVar_label", "ClinVar_gold_stars"])
                writer.writeheader()
                for pos, label in ((2, 0), (3, 1)):
                    writer.writerow({"#CHROM": "1", "POS": pos, "REF": "A", "ALT": "G",
                                     "ClinVar_label": label, "ClinVar_gold_stars": 1})
            class FakeModel:
                name = "fake"
                default_score = "value"
                direction = 1

                def load(self):
                    pass

                def score(self, variant):
                    return {"value": float(variant.pos)}

            args = SimpleNamespace(fasta=Path("unused"), model="ntv3", device="cpu",
                                   shard_index=0, num_shards=1, task="all", limit=0,
                                   version="2026-02", force=False, resume=False)
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch("clinvar_benchmark.cli.Reference"), \
                 patch("clinvar_benchmark.cli.make_model", return_value=FakeModel()), \
                 redirect_stdout(stdout), redirect_stderr(stderr):
                score(args, source, Path(temporary) / "output")
            self.assertIn("selected 2 variants (Benign=1, Pathogenic=1)", stdout.getvalue())
            self.assertIn("scored=2 (Benign=1, Pathogenic=1)", stdout.getvalue())
            self.assertIn("2/2", stderr.getvalue())

    def test_resume_salvages_valid_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "data.csv"
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["#CHROM", "POS", "REF", "ALT",
                                                              "ClinVar_label", "ClinVar_gold_stars"])
                writer.writeheader()
                for pos, label in ((2, 0), (3, 1)):
                    writer.writerow({"#CHROM": "1", "POS": pos, "REF": "A", "ALT": "G",
                                     "ClinVar_label": label, "ClinVar_gold_stars": 1})
            folder = Path(temporary) / "output"
            output = shard_path(folder, 0, 1)
            write_predictions(output, [dict(model="fake", version="2026-02", task="all",
                                            variant_key="1:2:A:G", chrom="1", pos=2,
                                            ref="A", alt="G", label=0, stars=1,
                                            score_name="value", raw_score=2.0, score=2.0,
                                            status="ok", error="", extra_scores_json="{}")])
            with output.open("ab") as handle:
                handle.write(b"corrupt-tail")
            recovered, error = read_recoverable_predictions(output)
            self.assertEqual(len(recovered), 1)
            self.assertIsNotNone(error)

            class FakeModel:
                name = "fake"
                default_score = "value"
                direction = 1
                calls = []

                def load(self):
                    pass

                def score(self, variant):
                    self.calls.append(variant.key)
                    return {"value": float(variant.pos)}

            fake = FakeModel()
            args = SimpleNamespace(fasta=Path("unused"), model="ntv3", device="cpu",
                                   shard_index=0, num_shards=1, task="all", limit=0,
                                   version="2026-02", force=False, resume=True)
            with patch("clinvar_benchmark.cli.Reference"), \
                 patch("clinvar_benchmark.cli.make_model", return_value=fake), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                score(args, source, folder)
            self.assertEqual(fake.calls, ["1:3:A:G"])
            self.assertEqual(evaluate([output], source, "2026-02", "all", "fake", 0, 1)["n_scored"], 2)
            self.assertEqual(len(list(folder.glob("*.backup-*"))), 1)

    def test_concurrent_staging_names_are_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scores.csv.gz"
            def write_one(label):
                write_predictions(path, [dict(model=f"model-{label}", version="v",
                                              task="all", variant_key=f"1:{label}:A:G",
                                              chrom="1", pos=label, ref="A", alt="G",
                                              label=label, stars=1, score_name="value",
                                              raw_score=label, score=label, status="ok",
                                              error="", extra_scores_json="{}")])
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(write_one, (0, 1)))
            with gzip.open(path, "rt", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertIn(rows[0]["model"], {"model-0", "model-1"})
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])

    def test_duplicate_shard_is_locked(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            lock_path = folder / "shard-000-of-001.csv.gz.lock"
            with lock_path.open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                args = SimpleNamespace(shard_index=0, num_shards=1)
                with self.assertRaisesRegex(RuntimeError, "Another process is scoring"):
                    score(args, folder / "unused.csv", folder)


if __name__ == "__main__":
    unittest.main()
