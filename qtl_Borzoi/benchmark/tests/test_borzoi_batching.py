import numpy as np
import os
from io import StringIO
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from qtl_benchmark.borzoi_batching import (
    backfill_missing_options,
    borzoi_batch_geometry,
    iter_ref_alt_prediction_batches,
    iter_ref_alt_predictions,
    log_prediction_progress,
    prediction_progress_label,
    prediction_stream_size,
    sequence_batch_size,
    validate_borzoi_variant_batch,
)


class BorzoiBatchingTest(unittest.TestCase):
    BORZOI_MODEL = {
        "seq_length": 524288,
        "trunk": [{"name": "conv_dna", "filters": 512}],
    }

    def test_borzoi_batch_three_is_below_int32_launch_limit(self):
        work_elements, sequences, maximum = borzoi_batch_geometry(
            self.BORZOI_MODEL, 3
        )
        self.assertEqual(work_elements, 1_610_612_736)
        self.assertEqual(sequences, 6)
        self.assertEqual(maximum, 3)
        self.assertEqual(validate_borzoi_variant_batch(self.BORZOI_MODEL, 3), 3)

    def test_borzoi_batch_four_fails_before_tensorflow_abort(self):
        with self.assertRaisesRegex(
            ValueError, "2,147,483,648.*maximum safe value.*3"
        ):
            validate_borzoi_variant_batch(self.BORZOI_MODEL, 4)

    def test_backfills_new_fields_in_legacy_pickled_options(self):
        legacy = SimpleNamespace(out_dir="stored", rc=True)
        parsed = SimpleNamespace(
            out_dir="cli-default", rc=False, variant_batch_size=1, bedgraph=False
        )

        result = backfill_missing_options(legacy, parsed)

        self.assertIs(result, legacy)
        self.assertEqual(result.out_dir, "stored")
        self.assertTrue(result.rc)
        self.assertEqual(result.variant_batch_size, 1)
        self.assertFalse(result.bedgraph)

    def test_batches_pairs_and_preserves_order(self):
        calls = []

        def model(sequences):
            calls.append(sequences.copy())
            return sequences * 10

        def make_pair(variant):
            return [np.array([variant, 0]), np.array([variant, 1])]

        results = list(iter_ref_alt_predictions(model, [0, 1, 2], make_pair, 2))

        self.assertEqual([call.shape[0] for call in calls], [4, 2])
        self.assertEqual([index for index, _, _ in results], [0, 1, 2])
        self.assertEqual(
            [ref.tolist() for _, ref, _ in results], [[0, 0], [10, 0], [20, 0]]
        )
        self.assertEqual(
            [alt.tolist() for _, _, alt in results], [[0, 10], [10, 10], [20, 10]]
        )

    def test_batch_iterator_exposes_model_call_boundaries(self):
        def make_pair(variant):
            return [np.array([variant, 0]), np.array([variant, 1])]

        batches = list(
            iter_ref_alt_prediction_batches(
                lambda sequences: sequences * 10, [0, 1, 2], make_pair, 2
            )
        )

        self.assertEqual(
            [(start, stop) for start, stop, _, _ in batches], [(0, 2), (2, 3)]
        )
        self.assertEqual(
            [[index for index, _, _ in predictions] for _, _, predictions, _ in batches],
            [[0, 1], [2]],
        )
        self.assertTrue(all(elapsed >= 0 for _, _, _, elapsed in batches))

    def test_progress_label_and_line_identify_parallel_worker(self):
        with patch.dict(os.environ, {"BORZOI_MODEL_NAME": "borzoi-replicate-0"}):
            label = prediction_progress_label(
                "/tmp/f0c0/train/model0_best.h5", "/tmp/neg_merge.vcf", 3
            )
        self.assertEqual(label, "borzoi-replicate-0/f0c0/neg/job3")

        output = StringIO()
        with patch("sys.stdout", output):
            log_prediction_progress(label, 2, 100, 7, 1.234)
        self.assertEqual(
            output.getvalue(),
            "[predict/borzoi-replicate-0/f0c0/neg/job3] "
            "2/100 rows=7 batch_elapsed=1.2s\n",
        )

    def test_batch_sizes_are_ref_alt_aware_and_stream_aligned(self):
        self.assertEqual(sequence_batch_size(3), 6)
        self.assertEqual(prediction_stream_size(3), 36)
        self.assertEqual(prediction_stream_size(20), 40)

    def test_sequence_batch_size_rejects_non_positive_values(self):
        for value in (0, -1, False):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    sequence_batch_size(value)

    def test_rejects_non_pair(self):
        with self.assertRaisesRegex(ValueError, "one REF and one ALT"):
            list(iter_ref_alt_predictions(lambda value: value, [1], lambda _: [1], 1))


if __name__ == "__main__":
    unittest.main()
