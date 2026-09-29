"""Small local replacement for the Slurm helper used by Borzoi launchers.

The paper launchers only use ``Job`` and ``multi_run``. This module keeps that
interface, runs data shards concurrently across local GPUs, and places a
barrier between model replicates.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Iterable, Union


class Job:
    def __init__(
        self,
        command: str,
        name: str,
        out_file: str,
        err_file: str,
        sb_file: str | None = None,
        **resources,
    ) -> None:
        self.command = command
        self.name = name
        self.out_file = out_file
        self.err_file = err_file
        self.sb_file = sb_file
        self.resources = resources


def _run_job(
    job: Union[Job, str], verbose: bool, gpu_id: str | None = None
) -> None:
    if isinstance(job, str):
        command = job
    else:
        command = job.command

    # The upstream launchers prepend shell setup such as
    # ``conda activate ...; echo $HOSTNAME; time``. The runner itself is
    # already in the requested environment, so execute only the final Python
    # command, directly and synchronously.
    command = command.rsplit(";", 1)[-1].strip()
    if command.startswith("time "):
        command = command[5:].lstrip()
    argv = shlex.split(command)
    if argv and argv[0].endswith(".py"):
        argv[0] = shutil.which(argv[0]) or argv[0]
        argv.insert(0, sys.executable)

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
    gpu_label = f"[gpu={gpu_id}] " if gpu_id is not None else ""
    print(f"+ {gpu_label}{shlex.join(argv)}", flush=True)
    completed = subprocess.run(argv, env=env)

    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}")
    if isinstance(job, Job):
        completion_file = job.resources.get("completion_file")
        if completion_file:
            completion_path = Path(completion_file)
            completion_path.parent.mkdir(parents=True, exist_ok=True)
            completion_path.touch()


def _run_gpu_queue(
    jobs: Iterable[Union[Job, str]], verbose: bool, gpu_id: str
) -> None:
    """Run one queue serially so two child processes never share a GPU."""
    for job in jobs:
        _run_job(job, verbose=verbose, gpu_id=gpu_id)


def _replicate_key(job: Union[Job, str]) -> str | None:
    """Return the fold/cross token used as a model-replicate barrier."""
    name = job.name if isinstance(job, Job) else ""
    match = re.search(r"f\d+c\d+", name)
    return match.group(0) if match else None


def _replicate_batches(
    jobs: list[Union[Job, str]],
) -> list[tuple[str | None, list[Union[Job, str]]]]:
    """Group consecutive jobs without reordering the launcher job list."""
    batches: list[tuple[str | None, list[Union[Job, str]]]] = []
    for job in jobs:
        key = _replicate_key(job)
        if not batches or batches[-1][0] != key:
            batches.append((key, []))
        batches[-1][1].append(job)
    return batches


def multi_run(
    jobs: Iterable[Union[Job, str]],
    max_proc: int | None = None,
    verbose: bool = False,
    launch_sleep: int | float = 0,
    update_sleep: int | float = 0,
    **kwargs,
) -> None:
    """Run jobs locally, optionally assigning one process to each configured GPU.

    The default remains sequential. ``QTL_GPU_IDS=0,1`` together with
    ``QTL_MAX_PROCS=2`` enables two concurrent data-shard processes and exposes
    exactly one physical GPU to each child. Jobs with the same ``fNcN`` model
    token run together; all of them must finish before the next model starts.
    """

    del max_proc, launch_sleep, update_sleep, kwargs
    job_list = list(jobs)
    gpu_ids = tuple(
        gpu.strip()
        for gpu in os.environ.get("QTL_GPU_IDS", "").split(",")
        if gpu.strip()
    )
    configured_workers = os.environ.get("QTL_MAX_PROCS", "1")
    try:
        worker_count = int(configured_workers)
    except ValueError as error:
        raise ValueError("QTL_MAX_PROCS must be a positive integer") from error
    if worker_count < 1:
        raise ValueError("QTL_MAX_PROCS must be a positive integer")
    if gpu_ids:
        worker_count = min(worker_count, len(gpu_ids))
    else:
        worker_count = 1
    if verbose:
        print(
            f"[local] jobs={len(job_list)} workers={worker_count} "
            f"gpus={','.join(gpu_ids) if gpu_ids else 'unrestricted'}",
            flush=True,
        )
    for replicate, batch in _replicate_batches(job_list):
        if verbose and replicate is not None:
            print(
                f"[local] model={replicate} shard_jobs={len(batch)} (barrier)",
                flush=True,
            )
        if worker_count == 1:
            gpu_id = gpu_ids[0] if gpu_ids else None
            for job in batch:
                _run_job(job, verbose=verbose, gpu_id=gpu_id)
            continue

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _run_gpu_queue,
                    batch[index::worker_count],
                    verbose,
                    gpu,
                )
                for index, gpu in enumerate(gpu_ids[:worker_count])
                if index < len(batch)
            ]
            for future in futures:
                future.result()
