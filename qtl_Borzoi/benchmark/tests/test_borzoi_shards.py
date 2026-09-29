from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from qtl_benchmark.borzoi_shards import collect_sed_shards


class BorzoiShardCollectionTest(unittest.TestCase):
    def test_collects_rows_and_reindexes_snps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for shard_index in range(2):
                shard_dir = root / f"job{shard_index}"
                shard_dir.mkdir()
                with h5py.File(shard_dir / "sed.h5", "w") as handle:
                    handle.create_dataset(
                        "snp", data=np.array([f"rs{shard_index}"], dtype="S")
                    )
                    handle.create_dataset("chr", data=np.array(["chr1"], dtype="S"))
                    handle.create_dataset("pos", data=np.array([10 + shard_index]))
                    handle.create_dataset("ref_allele", data=np.array(["A"], dtype="S"))
                    handle.create_dataset("alt_allele", data=np.array(["G"], dtype="S"))
                    handle.create_dataset("target_ids", data=np.array(["t0"], dtype="S"))
                    handle.create_dataset("target_labels", data=np.array(["T0"], dtype="S"))
                    handle.create_dataset("si", data=np.array([0, 0]))
                    handle.create_dataset(
                        "gene",
                        data=np.array(
                            [f"g{shard_index}a", f"g{shard_index}b"], dtype="S"
                        ),
                    )
                    handle.create_dataset(
                        "nDi",
                        data=np.array(
                            [[shard_index + 0.1], [shard_index + 0.2]],
                            dtype="float16",
                        ),
                    )

            merged_path = collect_sed_shards(root, 2)

            self.assertEqual(merged_path, root / "sed.h5")
            self.assertFalse((root / "sed.h5.tmp").exists())
            with h5py.File(merged_path, "r") as merged:
                self.assertEqual(list(merged["snp"].asstr()[:]), ["rs0", "rs1"])
                self.assertEqual(list(merged["si"][:]), [0, 0, 1, 1])
                self.assertEqual(
                    list(merged["gene"].asstr()[:]),
                    ["g0a", "g0b", "g1a", "g1b"],
                )
                self.assertEqual(merged["nDi"].shape, (4, 1))


if __name__ == "__main__":
    unittest.main()
