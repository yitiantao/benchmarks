#!/usr/bin/env python3
"""Execute a bundled Borzoi script, including two compatibility fixes."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


def main() -> None:
    script_name = Path(sys.argv[0]).name
    scripts_dir = Path(os.environ["BORZOI_SCRIPTS_DIR"]).resolve()
    script = scripts_dir / script_name
    if not script.is_file():
        raise SystemExit(f"Borzoi script not found: {script}")

    # The downloaded SuSiE tables remain gzip compressed.  The upstream
    # coefficient scripts append '.tsv'; transparently use '.tsv.gz' instead.
    if script_name in {"borzoi_gtex_coef_sad.py", "borzoi_gtex_coef_sed.py"}:
        import pandas as pd

        original_read_csv = pd.read_csv

        def read_csv_compat(path, *args, **kwargs):
            if isinstance(path, (str, os.PathLike)):
                candidate = Path(path)
                compressed = Path(str(candidate) + ".gz")
                if not candidate.exists() and compressed.is_file():
                    path = compressed
            return original_read_csv(path, *args, **kwargs)

        pd.read_csv = read_csv_compat

    sys.path.insert(0, str(scripts_dir))
    # Keep the local sequential ``slurm`` compatibility module ahead of the
    # upstream module, which requires sbatch/squeue.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
