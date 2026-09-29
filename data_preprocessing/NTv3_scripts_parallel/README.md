# NTv3 BigWig coverage tools

Coordinates are 0-based, half-open (`start` included, `end` excluded).

Install the two runtime dependencies:

```bash
python -m pip install numpy pyBigWig
```

Build the catalog from the benchmarks workspace (this scans every BigWig and can take a while):

```bash
python NTv3_scripts_parallel/build_coverage_catalog.py NTv3_benchmark_dataset \
  --output NTv3_scripts_parallel/bigwig_coverage_catalog
```

Outputs:

- `tracks.tsv`: track ID, species, assay, assembly, path and total covered bases.
- `track_chromosomes.tsv.gz`: chromosome size and first/last covered coordinate per track.
- `coverage_segments.tsv.gz`: non-overlapping intervals and the number of tracks covering each interval.
- `all_tracks_covered.bed.gz`: intervals covered by every track in a species/assay group.

Extract all human ATAC-seq signals over a region:

```bash
python NTv3_scripts_parallel/extract_signal.py NTv3_scripts_parallel/bigwig_coverage_catalog \
  --dataset NTv3_benchmark_dataset --species human --assay ATAC-seq \
  --region chr19:6700000-6831072 --bin-size 128 \
  --fill nan --output human_atac_chr19.npz
```

The NPZ contains `signal` (`tracks × bins`), `covered` (same-shape Boolean mask),
`file_ids`, and JSON `metadata`. Missing coordinates produce a warning. Prefer `nan`
plus the coverage mask for evaluation; use `--fill 0` only when downstream code cannot
handle NaNs and zero is scientifically appropriate.

## TensorStore training dataset

The scalable pipeline mirrors the NTv3 post-training storage layout:

- one TensorStore (Zarr) per species;
- `float16` signal values;
- zstd level 3 compression;
- chunks of `8192 bp × all species tracks`;
- optional equally-shaped coverage masks to distinguish true zero from missing data;
- Grain-compatible deterministic random-access loading.

Install dependencies:

```bash
python -m pip install -r NTv3_scripts_parallel/requirements.txt
```

## Parallel converter variant

This directory is an independent parallel-capable copy of `NTv3_scripts/`; the original
single-process scripts are intentionally unchanged. Prefer a separate output root while
testing so an interrupted experiment cannot replace an existing prepared dataset:

```bash
python NTv3_scripts_parallel/prepare_tensorstore.py NTv3_benchmark_dataset \
  NTv3_prepared_parallel/tensorstore --species human --workers 4

python NTv3_scripts_parallel/prepare_annotations.py NTv3_benchmark_dataset \
  NTv3_prepared_parallel/tensorstore --species human --workers 4
```

`--workers 1` is the deterministic sequential fallback and remains the default.
`--prefetch N` bounds the number of large results/tasks in flight; its default is
`2 * workers`. Do not launch two converter commands targeting the same species and
output directory simultaneously.

Functional conversion parallelizes BigWig reads and float16/mask construction by
genomic block. Every worker opens process-local BigWig handles; only the parent process
writes TensorStore chunks and `progress.json`, preserving ordered writes and block-level
resume. Each pending block occupies approximately
`block_bp * num_tracks * 3 bytes` for float16 signal plus uint8 coverage, excluding IPC
overhead. Lower `--block-chunks` and/or `--prefetch` if memory is constrained. Since each
worker opens all species tracks, the script automatically reduces the requested worker
count when the OS file-descriptor limit would be exceeded.

Annotation conversion parallelizes bounded genomic blocks. Workers only rasterize BED
intervals and return label buffers; the parent alone writes TensorStore in chromosome
and coordinate order. This avoids severe small-file contention when the store lives on
a network filesystem, and prints progress after every block rather than only after a
whole human chromosome. The parent aggregates `positive_bases` and writes final
metadata only after all blocks finish. A worker's main label buffer is approximately
`block_bp * num_annotation_elements bytes`.

Small-data multiprocessing overhead may cancel the benefit. On this machine, limited
local smoke benchmarks gave:

| Workload | 1 worker | 4 workers | Observed speedup |
|---|---:|---:|---:|
| human functional, first 500 kb of each selected contig, 34 tracks | 10.64 s | 5.18 s | 2.06× |
| tomato annotation, first 5 Mb of each selected contig, 4 elements | 2.50 s | 1.86 s | 1.35× |

These are implementation smoke benchmarks, not stable throughput guarantees. Start
with 4 workers, watch RAM, open-file count and disk utilization, then try 8 only if I/O
is not already saturated. Output compatibility was checked by converting identical
subsets with 1 and 2 workers: functional signal, coverage, annotation arrays and final
metadata were all exactly equal.

### 1. Build split manifests

Paper-compatible validation/test tiling uses stride equal to input length:

```bash
python NTv3_scripts_parallel/build_manifests.py NTv3_benchmark_dataset \
  NTv3_prepared/manifests/32k --window-size 32768 \
  --target-fraction 0.375 --paper-eval
```

Without `--paper-eval`, the script instead creates fixed-size random validation and
test manifests compatible with the simplified public tutorial (1,000 and 10,000
samples per species by default). In either mode, every complete 32 kb input—not only
its target crop—is constrained to one official `splits.bed` region.

#### What the manifests are for

A manifest is a coordinate index and experiment record; it does **not** contain DNA,
BigWig signal, or annotation tensors. It tells the loader which species/chromosome
interval to fetch from FASTA and TensorStore. Keeping coordinates separate from the
large arrays avoids copying data into train/validation/test datasets and makes the
same prepared TensorStore reusable across window sizes, sampling policies and runs.

The output directory contains three kinds of files:

| File | Purpose |
|---|---|
| `official_regions.parquet` | Tabular form of official `splits.bed` regions long enough for the configured window; used as the allowed coordinate pool for dynamic training windows |
| `<species>.val.parquet`, `<species>.test.parquet` | Materialized, deterministic evaluation windows; every row is one reproducible model example |
| `manifest.json` | Records how the files were generated: dataset/output paths, window size, target fraction, sample counts, seed, policy, paper-eval flag and selected species |

`official_regions.parquet` rows have `species`, `split`, `chrom`, `start`, `end`, and
`region_id`. During training, `NTv3WindowSource(regions=..., split="train")` selects or
computes windows only inside rows marked `train`. This preserves the official split
without writing millions of highly overlapping training rows to disk.

Each fixed validation/test manifest row has:

| Field | Meaning |
|---|---|
| `sample_id` | Stable hash identifying this species/chromosome/window |
| `species`, `assembly`, `split`, `chrom` | Genome and split identity |
| `input_start`, `input_end` | Full 0-based half-open DNA input window fetched from FASTA |
| `target_start`, `target_end` | Central supervised interval fetched from TensorStore and scored by the loss/metric |
| `region_id` | Official split region from which the window was derived |
| `sampling_policy`, `seed` | Provenance needed to reproduce how the row was selected |

The important leakage boundary is the **full input interval**: the builder accepts a
window only when `[input_start, input_end)` fits entirely inside one official region of
the same split. Thus a test target cannot receive sequence context extending into a
train region. Fixed val/test files should be generated once and reused unchanged for
all compared models. Training can remain dynamic because its split boundary is still
enforced by `official_regions.parquet`.

Conceptually, one loaded example is assembled as:

```text
manifest/region coordinates
        ├── FASTA ----------> sequence[input_start:input_end]
        ├── signal store ---> targets[target_start:target_end, tracks]
        ├── coverage store -> target_mask[target_start:target_end, tracks]
        └── annotation -----> annotation_targets[target_start:target_end, elements]
```

### 2. Convert BigWigs

Start with one species and inspect disk usage before converting all species:

```bash
python NTv3_scripts_parallel/prepare_tensorstore.py NTv3_benchmark_dataset \
  NTv3_prepared/tensorstore --species arabidopsis
```

Convert all species that have functional tracks:

```bash
python NTv3_scripts_parallel/prepare_tensorstore.py NTv3_benchmark_dataset \
  NTv3_prepared/tensorstore
```

The converter writes progress after every block and resumes an incomplete species.
Use `--overwrite` to rebuild a completed or partial store. TensorStore dimensions are
logical sizes; actual disk usage depends heavily on signal sparsity and zstd compression.

### 3. Train with Grain

```python
from NTv3_scripts.ntv3_data import NTv3WindowSource, make_grain_loader

train_source = NTv3WindowSource(
    "NTv3_prepared/tensorstore",
    "human",
    regions="NTv3_prepared/manifests/32k/official_regions.parquet",
    split="train",
    window_size=32_768,
    target_fraction=0.375,
    sampling_policy="paper_stride",
    assay="ATAC-seq",       # None loads every human track
    normalize=True,
    use_mask=True,
)
train_loader = make_grain_loader(
    train_source, batch_size=2, shuffle=True, seed=0, workers=8
)

for batch in train_loader:
    # batch["sequence"]: DNA strings
    # batch["targets"]: [batch, target_bp, tracks]
    # batch["target_mask"]: same shape, false where BigWig has no explicit value
    train_step(batch)
```

Paper-mode training uses stride `0.1% × sequence_length` (33 bp after rounding for
32,768 bp); validation/test use stride equal to sequence length. This creates a very
large virtual training dataset, but coordinates are computed from the index and are
not materialized in a huge manifest.

Fixed test loading:

```python
test_source = NTv3WindowSource(
    "NTv3_prepared/tensorstore",
    "human",
    manifest="NTv3_prepared/manifests/32k/human.test.parquet",
    assay="ATAC-seq",
)
test_loader = make_grain_loader(
    test_source, batch_size=2, shuffle=False, workers=8
)
```

Target normalization follows the paper: divide by per-track mean, apply `x**0.75`
to RNA-seq tracks, then square-root soft clipping above 10. Raw missing values are
stored as zero, while `target_mask` preserves whether a value was explicitly covered.

### 4. Validate a prepared store

```bash
python NTv3_scripts_parallel/check_prepared_data.py \
  NTv3_prepared/tensorstore \
  NTv3_prepared/manifests/32k/human.test.parquet \
  --species human --samples 3
```

For a cheap converter smoke test, add `--limit-bp 32768`; do not use a limited store
for real manifests or training.

## On-disk TensorStore schema

Each species is stored independently. A species with both functional and annotation
supervision (for example human) looks like:

```text
NTv3_prepared/tensorstore/<species>/
├── signal/                    # Zarr/TensorStore array, float16
│   ├── .zarray
│   └── <chunk files>
├── coverage/                  # optional Zarr/TensorStore array, uint8
│   ├── .zarray
│   └── <chunk files>
├── annotations/               # Zarr/TensorStore array, uint8
│   ├── .zarray
│   └── <chunk files>
├── metadata.json              # functional schema and track catalogue
└── annotation_metadata.json   # annotation schema and element catalogue
```

The files below `signal/`, `coverage/`, and `annotations/` are physical Zarr metadata
and compressed chunk files, not one record per training example. Training windows are
slices from these arrays; their coordinates come from a manifest/regions Parquet file.

### Functional arrays

| Store | Logical shape | On-disk dtype | Meaning |
|---|---:|---|---|
| `signal` | `[total_padded_genome_bp, num_tracks]` | `float16` | BigWig signal; missing positions are stored as `0.0` |
| `coverage` | same as `signal` | `uint8` | `1` means the BigWig supplied a finite value; `0` means missing/uncovered |

Both use chunks `[8192, num_tracks]` and zstd level 3. The second dimension is the
complete track set for that species, so column `j` must be interpreted through
`metadata.json -> tracks[j]`. A signal zero is not sufficient to decide whether a
position was measured; use the corresponding `coverage` value. If conversion used
`--no-coverage-mask`, the `coverage/` store is absent and `has_coverage_mask` is false.

#### Species and modality dimensions

Functional data is **not** forced into one globally uniform matrix across species.
There is one matrix per species:

```text
signal_<species>.shape = [padded genome length of that species,
                          actual number of tracks available for that species]
```

Consequently, both dimensions may differ between species. Within one species,
`signal` and `coverage` always have exactly the same shape. There is no separate
modality/assay dimension and no field embedded inside an array cell: every column is
one concrete experimental track, and its identity is stored in the sidecar
`metadata.json -> tracks[column_index]`. The relevant per-column fields are `index`,
`file_id`, `assay`, `mean`, `std`, and `bigwig_path`.

Multiple tracks can share one `assay`; for example, seven ATAC-seq experiments occupy
seven separate columns. `NTv3WindowSource(..., assay="ATAC-seq")` uses the `assay`
field to select those columns. With `assay=None`, it returns all columns in metadata
order. The loader returns `track_ids`, `track_assays`, and `track_indices` in exactly
the same order as the last dimension of `targets`, so no manual JSON lookup is needed
during training.

For clarity, rows are genomic base positions and columns are tracks. To inspect the
on-disk mapping directly:

```python
import json
from pathlib import Path

meta = json.loads(Path(
    "NTv3_prepared/tensorstore/human/metadata.json"
).read_text())

for track in meta["tracks"]:
    print(track["index"], track["file_id"], track["assay"])

# signal[:, track["index"]] is that track's genome-wide signal column.
```

For a loaded example, inspect the already aligned arrays instead:

```python
example = source[0]
for output_column, (store_column, file_id, assay) in enumerate(zip(
    example["track_indices"],
    example["track_ids"],
    example["track_assays"],
)):
    print(output_column, store_column, file_id, assay)
    # example["targets"][:, output_column] belongs to this file_id and assay.
```

When an assay filter is active, output column zero is not necessarily TensorStore
column zero. `track_indices[output_column]` gives the original TensorStore column;
the three returned metadata arrays always follow the filtered output-column order.

If a species has no track for a modality, that modality contributes **no column**—the
converter does not create an all-zero placeholder. If a species has no functional
tracks at all (cattle in this benchmark), it has no functional `signal`, `coverage`,
or `metadata.json` and must be loaded annotation-only. This is different from a track
that exists but lacks values over part of a chromosome: that track keeps its column,
with `signal=0` and `coverage=0` at missing positions.

The current benchmark metadata produces these per-species functional shapes in the
track dimension:

| Species | Number of tracks | Available assays/modalities |
|---|---:|---|
| arabidopsis | 4 | ribo-seq (4) |
| chicken | 14 | ATAC-seq (7), RNA-seq (7) |
| human | 34 | ATAC-seq (5), Histone ChIP-seq (4), PRO-cap (10), eCLIP (10), polyA plus RNA-seq (2), total RNA-seq (3) |
| maize | 8 | ribo-seq (8) |
| rice | 14 | ribo-seq (14) |
| tomato | 20 | ATAC-seq (7), RNA-seq (13) |
| cattle | 0 | annotation-only |

Because track counts and semantics differ, ordinary dense batches should contain one
species and a fixed selected assay/track set. A multi-species training scheduler can
alternate such species-homogeneous batches. If a future model requires mixed-species
examples in one dense tensor, define an explicit global modality/track schema, pad or
project each species to it, and provide a column-validity mask; zeros alone must not be
treated as valid missing-modality targets. The present storage deliberately avoids
that potentially very large global padded matrix.

`metadata.json` contains:

- `format_version`, `species`, `assembly`, `coordinate_system`;
- `dtype`, `compressor`, `chunk_shape`, `shape`, `missing_fill`;
- `float16_overflow_values_clipped`, `has_coverage_mask`, `fasta_path`;
- `chromosomes[]`: `name`, stored `size`, original `reference_size`, and global
  concatenated-array `offset`; annotation stores also record `fasta_name` when the
  public split/BED name differs from the FASTA header (for example cattle
  `name="chr1"`, `fasta_name="1"`);
- `tracks[]`: array-column `index`, `file_id`, `assay`, training normalization `mean`
  and `std`, and source `bigwig_path`.

Chromosomes do not form a separate array dimension. For chromosome interval
`[start, end)`, the physical row range is
`[chromosome.offset + start, chromosome.offset + end)`. Chromosome starts and stored
allocations are aligned to 8192 bp; `size` itself remains the real usable length, and
alignment padding is never sampled by valid manifests.

### Annotation array

| Store | Logical shape | On-disk dtype | Meaning |
|---|---:|---|---|
| `annotations` | `[total_padded_genome_bp, num_elements]` | `uint8` | Independent binary labels (`0` background, `1` positive) |

It also uses `[8192, num_elements]` chunks and zstd level 3. Column `j` is described by
`annotation_metadata.json -> elements[j]`. Its metadata contains the common schema
fields and `fasta_path`, `chromosomes[]`, plus:

- `elements[]`: column `index`, annotation `name`, source `interval_count`, and total
  `matched_interval_count` on selected contigs, and total `positive_bases` after
  interval union/rasterization;
- `strand_encoding`: documents that positive and negative strands are collapsed;
- `label_semantics`: documents independent binary/multi-label interpretation.

The annotation converter resolves the common optional `chr` prefix between
`splits.bed`/BED and FASTA names, while preserving split/manifest names as the public
coordinate system. It refuses to create a store when zero chromosomes or zero matching
BED intervals are found. An existing metadata file with `shape[0] == 0` is treated as
invalid and must be rebuilt with `--overwrite`; it is never accepted as complete.

New functional and annotation stores order selected chromosomes by their FASTA order,
not lexicographic split-file order (`1, 2, ..., 10` rather than `1, 10, 11, 2`). For
robustness with legacy stores, joint loading does not assume identical concatenated
offsets: it verifies equal public chromosome names/sizes and looks up the functional
and annotation offset independently. Thus ordering differences cannot silently return
labels from the wrong chromosome.

The functional converter applies the same preflight logic independently to split,
FASTA, and every BigWig. It saves each chromosome's public `name` and `fasta_name`,
resolves a per-track BigWig contig name, rejects tracks with zero mapped contigs, and
does not write final metadata if the entire conversion has zero covered values. Each
track metadata entry records `mapped_contig_count` and `covered_values`. The loader also
rejects empty stores and checks that all manifest/region contigs exist before reading.

Dataset-wide naming audit performed on the current benchmark:

| Species | Split → FASTA result | Action |
|---|---|---|
| cattle | 1,958/1,958 require `chr`-prefix alias | annotation store must be rebuilt |
| chicken | 205/205 require `chr`-prefix alias | legacy functional store must be rebuilt |
| arabidopsis | 5/5 exact | no naming issue |
| human | 23/23 exact | no naming issue |
| maize | 10/10 exact | no naming issue |
| rice | 12/12 exact | no naming issue |
| tomato | 12/12 exact | no naming issue |

Human and tomato annotation BEDs contain additional contigs outside the official
benchmark split. Their omission under the default mode is intentional; `--all-contigs`
is required only when supervision outside benchmark regions is desired. To replace the
known invalid legacy stores:

```bash
python NTv3_scripts_parallel/prepare_annotations.py NTv3_benchmark_dataset \
  NTv3_prepared/tensorstore --species cattle --overwrite

python NTv3_scripts_parallel/prepare_tensorstore.py NTv3_benchmark_dataset \
  NTv3_prepared/tensorstore --species chicken --overwrite
```

### Record returned to training code

`NTv3WindowSource[index]` combines FASTA, manifest coordinates and selected TensorStore
slices. Its fields are:

| Field | Type/shape | Meaning |
|---|---|---|
| `sample_id` | string | stable manifest ID (or deterministic dynamic-window ID) |
| `species`, `chrom` | string | source species and chromosome |
| `input_start`, `input_end` | integer | 0-based half-open input coordinates |
| `sequence` | string of length `input_bp` | reference DNA passed to tokenizer/model |
| `targets` | `float32 [target_bp, selected_tracks]` | functional signal, optionally normalized |
| `target_mask` | `bool [target_bp, selected_tracks]` | functional coverage/missing-value loss mask |
| `track_ids` | string array `[selected_tracks]` | column identity for `targets` |
| `track_assays` | string array `[selected_tracks]` | modality/assay of every returned target column |
| `track_indices` | `int64 [selected_tracks]` | original TensorStore column of every returned target column |
| `annotation_targets` | `uint8 [target_bp, elements]` | per-base annotation labels |
| `annotation_names` | string array `[elements]` | column identity for annotation labels |

The functional fields appear only when `load_functional=True`; annotation fields appear
only when `load_annotations=True`. `sequence` spans the full input window, while target
arrays contain only the central target interval defined by the manifest/source policy.

## Genome annotations

NTv3 treats annotation elements as independent per-nucleotide binary tasks. Build the
four benchmark labels (`exon`, `intron`, `splice_acceptor`, `start_codon`) with:

```bash
python NTv3_scripts_parallel/prepare_annotations.py NTv3_benchmark_dataset \
  NTv3_prepared/tensorstore
```

The output `annotations` array has shape `[genome_bp, elements]`, dtype `uint8`, zstd
level 3 compression, and chunks `8192 bp × all elements`. BED strand is collapsed:
a base is positive when the element is annotated on either strand. Labels are
multi-label, so exon and start-codon may both be positive at one base.

Both converters include only contigs referenced by `splits.bed` by default. This avoids
thousands of unused alternate-contig chunk files. Pass `--all-contigs` only if full
assembly coverage is needed outside the benchmark.

Annotation-only loading (including cattle):

```python
source = NTv3WindowSource(
    "NTv3_prepared/tensorstore", "cattle",
    regions="NTv3_prepared/manifests/32k_paper/official_regions.parquet",
    split="train", sampling_policy="paper_stride",
    load_functional=False, load_annotations=True,
)
```

Joint functional and annotation loading for human or tomato:

```python
source = NTv3WindowSource(
    "NTv3_prepared/tensorstore", "human",
    manifest="NTv3_prepared/manifests/32k_paper/human.test.parquet",
    load_functional=True, load_annotations=True,
)
```

This adds `annotation_targets` (`uint8 [target_bp, elements]`) and
`annotation_names` to every record. The NTv3 methods use a weighted focal loss with
`gamma=2.0` for annotation fine-tuning/model selection and report MCC for benchmark
comparison. Each element remains an independent per-base binary target; an equivalent
implementation may use one binary logit or two class logits per element.
