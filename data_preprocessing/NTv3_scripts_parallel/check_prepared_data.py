#!/usr/bin/env python3
"""Read several prepared samples and verify shapes, split containment and finite values."""

import argparse
from pathlib import Path

from ntv3_data import NTv3WindowSource, make_grain_loader


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tensorstore", type=Path)
    p.add_argument("manifest", type=Path)
    p.add_argument("--species", default="human")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--grain-workers", type=int, default=0,
                   help="Also read one batch through Grain with this many workers")
    p.add_argument("--annotations-only", action="store_true")
    a = p.parse_args()
    source = NTv3WindowSource(a.tensorstore, a.species, manifest=a.manifest,
                              load_functional=not a.annotations_only,
                              load_annotations=a.annotations_only)
    for i in range(min(a.samples, len(source))):
        x = source[i]
        assert len(x["sequence"]) == x["input_end"] - x["input_start"]
        if a.annotations_only:
            print(i, x["sample_id"], x["chrom"], x["input_start"], x["input_end"],
                  x["annotation_targets"].shape,
                  f"positives={x['annotation_targets'].sum()}")
        else:
            assert x["targets"].shape == x["target_mask"].shape
            print(i, x["sample_id"], x["chrom"], x["input_start"], x["input_end"],
                  x["targets"].shape, f"coverage={x['target_mask'].mean():.4f}")
    if a.grain_workers:
        loader = make_grain_loader(source, batch_size=1, shuffle=False,
                                   workers=a.grain_workers, prefetch_buffer=4)
        iterator = iter(loader)
        batch = next(iterator)
        key = "annotation_targets" if a.annotations_only else "targets"
        print("grain", f"workers={a.grain_workers}", batch[key].shape)
        if hasattr(iterator, "close"):
            iterator.close()


if __name__ == "__main__":
    main()
