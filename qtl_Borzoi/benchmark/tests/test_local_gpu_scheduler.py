from __future__ import annotations

import os
from pathlib import Path
import shlex
import sys
import tempfile
import time
import unittest
from unittest import mock

from qtl_benchmark.slurm import Job, multi_run


class LocalGpuSchedulerTest(unittest.TestCase):
    def test_shards_are_parallel_and_assigned_round_robin(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            jobs = []
            for index in range(4):
                output = root / f"gpu-{index}.txt"
                code = "\n".join(
                    [
                        "import os",
                        "import time",
                        "from pathlib import Path",
                        f"Path({str(output)!r}).write_text("
                        "os.environ['CUDA_VISIBLE_DEVICES'])",
                        "time.sleep(0.25)",
                    ]
                )
                command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
                jobs.append(
                    Job(
                        command,
                        str(index),
                        "",
                        "",
                        completion_file=str(root / f"job-{index}.complete"),
                    )
                )

            with mock.patch.dict(
                os.environ,
                {"QTL_GPU_IDS": "2,5", "QTL_MAX_PROCS": "2"},
            ):
                started = time.monotonic()
                multi_run(jobs)
                elapsed = time.monotonic() - started

            self.assertEqual(
                [(root / f"gpu-{index}.txt").read_text() for index in range(4)],
                ["2", "5", "2", "5"],
            )
            self.assertTrue(
                all((root / f"job-{index}.complete").is_file() for index in range(4))
            )
            self.assertLess(elapsed, 0.9)

    def test_next_replicate_waits_for_all_current_replicate_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            marker0 = root / "replicate0-shard0"
            marker1 = root / "replicate0-shard1"

            jobs = []
            for index, delay in enumerate((0.05, 0.35)):
                marker = marker0 if index == 0 else marker1
                code = "\n".join(
                    [
                        "import time",
                        "from pathlib import Path",
                        f"time.sleep({delay})",
                        f"Path({str(marker)!r}).touch()",
                    ]
                )
                jobs.append(
                    Job(
                        f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}",
                        f"sqtl-f0c0_neg{index}",
                        "",
                        "",
                    )
                )

            verify_code = "\n".join(
                [
                    "from pathlib import Path",
                    f"assert Path({str(marker0)!r}).exists()",
                    f"assert Path({str(marker1)!r}).exists()",
                ]
            )
            for index in range(2):
                jobs.append(
                    Job(
                        f"{shlex.quote(sys.executable)} -c {shlex.quote(verify_code)}",
                        f"sqtl-f0c1_neg{index}",
                        "",
                        "",
                    )
                )

            with mock.patch.dict(
                os.environ,
                {"QTL_GPU_IDS": "0,1", "QTL_MAX_PROCS": "2"},
            ):
                multi_run(jobs)


if __name__ == "__main__":
    unittest.main()
