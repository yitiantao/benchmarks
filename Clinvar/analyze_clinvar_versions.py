#!/usr/bin/env python3
"""Compare ClinVar variant_summary releases using the VEP-eval filter logic.

This intentionally stops before MANE/reference-genome annotation.  It reproduces
the raw ClinVar filtering portion of VEP_ClinVar_Benchmarking_RefSeq.ipynb and
adds release-level descriptive statistics and exact cross-release comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


BASIC_CHROMOSOMES = {str(i) for i in range(1, 23)} | {"X", "Y"}
BASES = {"A", "C", "G", "T"}
CLINICAL_SIGNIFICANCE_TO_LABEL = {
    "Benign": 0,
    "Likely benign": 0.1,
    "Benign/Likely benign": 0.1,
    "Pathogenic": 1,
    "Likely pathogenic": 1.1,
    "Pathogenic/Likely pathogenic": 1.1,
    "Uncertain significance": 2,
}
REVIEW_STATUS_TO_GOLD_STARS = {
    "criteria provided, single submitter": 1,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, conflicting interpretations": 1,
    "no assertion criteria provided": np.nan,
    "reviewed by expert panel": 3,
    "no assertion provided": np.nan,
    "no interpretation for the single variant": np.nan,
    "practice guideline": 4,
}
USECOLS = [
    "Type",
    "ClinicalSignificance",
    "OriginSimple",
    "Assembly",
    "Chromosome",
    "ReviewStatus",
    "VariationID",
    "PositionVCF",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def update_counter(counter: Counter, values: pd.Series) -> None:
    counter.update(values.fillna("<NA>").astype(str).tolist())


def exact_keys(frame: pd.DataFrame) -> set[str]:
    """Return exact GRCh38 CHROM:POS:REF:ALT keys for a filtered frame."""
    return set(
        frame["Chromosome"].astype(str)
        + ":"
        + frame["PositionVCF"].astype(str)
        + ":"
        + frame["ReferenceAlleleVCF"].astype(str)
        + ":"
        + frame["AlternateAlleleVCF"].astype(str)
    )


def analyse_release(name: str, path: Path, chunksize: int) -> dict:
    distributions = {
        "assembly_raw": Counter(),
        "origin_raw": Counter(),
        "type_after_grch38_nonsomatic_basic_chromosome": Counter(),
        "clinical_significance_after_snv": Counter(),
        "review_status_after_selected_significance": Counter(),
        "chromosome_after_grch38_nonsomatic": Counter(),
        "label_after_star_filter": Counter(),
        "base_substitution_after_canonical_alleles": Counter(),
    }
    funnel = Counter()
    raw_ids: list[np.ndarray] = []
    grch38_ids: list[np.ndarray] = []
    candidate_ids: list[np.ndarray] = []
    candidate_keys: set[str] = set()
    exact_pb_ids: list[np.ndarray] = []
    exact_pb_keys: set[str] = set()
    samples: list[pd.DataFrame] = []

    reader = pd.read_csv(
        path,
        sep="\t",
        usecols=USECOLS,
        dtype={"Chromosome": "string"},
        low_memory=False,
        compression="gzip",
        chunksize=chunksize,
    )
    for chunk in reader:
        funnel["raw_rows"] += len(chunk)
        ids = pd.to_numeric(chunk["VariationID"], errors="coerce").dropna().to_numpy(dtype=np.int64)
        raw_ids.append(ids)
        update_counter(distributions["assembly_raw"], chunk["Assembly"])
        update_counter(distributions["origin_raw"], chunk["OriginSimple"])

        is_grch38 = chunk["Assembly"].eq("GRCh38")
        grch38 = chunk.loc[is_grch38]
        funnel["grch38_rows"] += len(grch38)
        grch38_ids.append(
            pd.to_numeric(grch38["VariationID"], errors="coerce").dropna().to_numpy(dtype=np.int64)
        )

        nonsomatic = grch38.loc[grch38["OriginSimple"].ne("somatic")]
        funnel["after_exclude_exact_somatic"] += len(nonsomatic)
        update_counter(distributions["chromosome_after_grch38_nonsomatic"], nonsomatic["Chromosome"])

        basic = nonsomatic.loc[nonsomatic["Chromosome"].isin(BASIC_CHROMOSOMES)]
        funnel["after_basic_chromosomes"] += len(basic)
        update_counter(
            distributions["type_after_grch38_nonsomatic_basic_chromosome"], basic["Type"]
        )

        snv = basic.loc[basic["Type"].eq("single nucleotide variant")].copy()
        funnel["after_single_nucleotide_variant"] += len(snv)
        update_counter(
            distributions["clinical_significance_after_snv"], snv["ClinicalSignificance"]
        )

        snv["ClinVar_label"] = snv["ClinicalSignificance"].map(
            CLINICAL_SIGNIFICANCE_TO_LABEL
        )
        selected = snv.loc[snv["ClinVar_label"].isin({0, 0.1, 1, 1.1})].copy()
        funnel["after_selected_clinical_significance"] += len(selected)
        update_counter(
            distributions["review_status_after_selected_significance"], selected["ReviewStatus"]
        )

        selected["gold_stars"] = selected["ReviewStatus"].map(REVIEW_STATUS_TO_GOLD_STARS)
        starred = selected.loc[selected["gold_stars"].isin({1, 2, 3, 4})].copy()
        funnel["after_gold_stars_ge_1"] += len(starred)
        update_counter(distributions["label_after_star_filter"], starred["ClinicalSignificance"])

        canonical = starred.loc[
            starred["ReferenceAlleleVCF"].isin(BASES)
            & starred["AlternateAlleleVCF"].isin(BASES)
        ].copy()
        funnel["after_canonical_acgt_alleles"] += len(canonical)
        candidate_ids.append(
            pd.to_numeric(canonical["VariationID"], errors="coerce")
            .dropna()
            .to_numpy(dtype=np.int64)
        )
        candidate_keys.update(exact_keys(canonical))
        true_substitutions = canonical.loc[
            canonical["ReferenceAlleleVCF"].ne(canonical["AlternateAlleleVCF"])
        ]
        funnel["after_true_substitution_ref_ne_alt"] += len(true_substitutions)
        exact_pb = canonical.loc[canonical["ClinVar_label"].isin({0, 1})]
        funnel["exact_benign_or_pathogenic_rows_pre_mane"] += len(exact_pb)
        funnel["exact_benign_rows_pre_mane"] += exact_pb["ClinVar_label"].eq(0).sum()
        funnel["exact_pathogenic_rows_pre_mane"] += exact_pb["ClinVar_label"].eq(1).sum()
        exact_pb_ids.append(
            pd.to_numeric(exact_pb["VariationID"], errors="coerce")
            .dropna()
            .to_numpy(dtype=np.int64)
        )
        exact_pb_keys.update(exact_keys(exact_pb))
        substitutions = (
            canonical["ReferenceAlleleVCF"].astype(str)
            + ">"
            + canonical["AlternateAlleleVCF"].astype(str)
        )
        update_counter(
            distributions["base_substitution_after_canonical_alleles"], substitutions
        )
        if sum(map(len, samples)) < 10:
            samples.append(canonical.head(10 - sum(map(len, samples))))

    def unique_ints(parts: list[np.ndarray]) -> np.ndarray:
        if not parts:
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate(parts))

    raw_unique_ids = unique_ints(raw_ids)
    grch38_unique_ids = unique_ints(grch38_ids)
    candidate_unique_ids = unique_ints(candidate_ids)
    exact_pb_unique_ids = unique_ints(exact_pb_ids)
    funnel["raw_unique_variation_ids"] = len(raw_unique_ids)
    funnel["grch38_unique_variation_ids"] = len(grch38_unique_ids)
    funnel["candidate_unique_variation_ids"] = len(candidate_unique_ids)
    funnel["candidate_unique_variant_keys"] = len(candidate_keys)
    funnel["candidate_duplicate_key_rows"] = (
        funnel["after_canonical_acgt_alleles"] - len(candidate_keys)
    )
    funnel["exact_pb_unique_variation_ids_pre_mane"] = len(exact_pb_unique_ids)
    funnel["exact_pb_unique_variant_keys_pre_mane"] = len(exact_pb_keys)

    return {
        "name": name,
        "path": str(path),
        "funnel": dict(funnel),
        "distributions": {key: dict(value) for key, value in distributions.items()},
        "raw_unique_ids": raw_unique_ids,
        "candidate_unique_ids": candidate_unique_ids,
        "candidate_keys": candidate_keys,
        "exact_pb_unique_ids": exact_pb_unique_ids,
        "exact_pb_keys": exact_pb_keys,
        "samples": pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(),
    }


def counter_rows(release: str, section: str, values: dict[str, int]) -> list[dict]:
    total = sum(values.values())
    return [
        {
            "release": release,
            "section": section,
            "value": value,
            "count": count,
            "fraction": count / total if total else np.nan,
        }
        for value, count in sorted(values.items(), key=lambda item: (-item[1], item[0]))
    ]


def compare_sets(old: set | np.ndarray, new: set | np.ndarray) -> dict[str, int]:
    old_set = set(old)
    new_set = set(new)
    return {
        "old": len(old_set),
        "latest": len(new_set),
        "shared": len(old_set & new_set),
        "only_old": len(old_set - new_set),
        "only_latest": len(new_set - old_set),
        "net_change": len(new_set) - len(old_set),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path)
    parser.add_argument("--latest", type=Path)
    parser.add_argument(
        "--release",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="release label and input path; repeat for two or more releases",
    )
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.release:
        release_specs = []
        for spec in args.release:
            if "=" not in spec:
                parser.error(f"invalid --release {spec!r}; expected NAME=PATH")
            name, raw_path = spec.split("=", 1)
            release_specs.append((name, Path(raw_path)))
    elif args.old and args.latest:
        release_specs = [
            ("2026-02", args.old),
            ("latest_2026-09-14", args.latest),
        ]
    else:
        parser.error("provide repeated --release NAME=PATH or both --old and --latest")
    if len(release_specs) < 2:
        parser.error("at least two releases are required")
    if len({name for name, _ in release_specs}) != len(release_specs):
        parser.error("release names must be unique")

    releases = [
        analyse_release(name, path, args.chunksize) for name, path in release_specs
    ]

    manifest = []
    for release in releases:
        path = Path(release["path"])
        manifest.append(
            {
                "release": release["name"],
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    (args.outdir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    funnel_order = [
        "raw_rows",
        "raw_unique_variation_ids",
        "grch38_rows",
        "grch38_unique_variation_ids",
        "after_exclude_exact_somatic",
        "after_basic_chromosomes",
        "after_single_nucleotide_variant",
        "after_selected_clinical_significance",
        "after_gold_stars_ge_1",
        "after_canonical_acgt_alleles",
        "after_true_substitution_ref_ne_alt",
        "candidate_unique_variation_ids",
        "candidate_unique_variant_keys",
        "candidate_duplicate_key_rows",
        "exact_benign_or_pathogenic_rows_pre_mane",
        "exact_benign_rows_pre_mane",
        "exact_pathogenic_rows_pre_mane",
        "exact_pb_unique_variation_ids_pre_mane",
        "exact_pb_unique_variant_keys_pre_mane",
    ]
    funnel_rows = []
    for release in releases:
        for metric in funnel_order:
            funnel_rows.append(
                {"release": release["name"], "metric": metric, "count": release["funnel"][metric]}
            )
    pd.DataFrame(funnel_rows).to_csv(args.outdir / "filter_funnel.csv", index=False)

    distribution_rows = []
    for release in releases:
        for section, values in release["distributions"].items():
            distribution_rows.extend(counter_rows(release["name"], section, values))
    pd.DataFrame(distribution_rows).to_csv(args.outdir / "distributions.csv", index=False)

    for release in releases:
        release["samples"].to_csv(
            args.outdir / f"sample_{release['name']}.tsv", sep="\t", index=False
        )

    comparison_fields = {
        "raw_unique_variation_ids": "raw_unique_ids",
        "candidate_unique_variation_ids": "candidate_unique_ids",
        "candidate_unique_variant_keys": "candidate_keys",
        "exact_pb_unique_variation_ids_pre_mane": "exact_pb_unique_ids",
        "exact_pb_unique_variant_keys_pre_mane": "exact_pb_keys",
    }
    comparisons = {}
    comparison_rows = []
    for old_release, new_release in itertools.combinations(releases, 2):
        comparison_name = f"{old_release['name']}_vs_{new_release['name']}"
        comparisons[comparison_name] = {}
        for metric, field in comparison_fields.items():
            counts = compare_sets(old_release[field], new_release[field])
            comparisons[comparison_name][metric] = counts
            comparison_rows.append(
                {
                    "from_release": old_release["name"],
                    "to_release": new_release["name"],
                    "metric": metric,
                    **counts,
                }
            )
    (args.outdir / "version_comparison.json").write_text(
        json.dumps(comparisons, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(comparison_rows).to_csv(args.outdir / "version_comparison.csv", index=False)

    print(pd.DataFrame(funnel_rows).pivot(index="metric", columns="release", values="count"))
    print("\nCross-release comparison")
    print(pd.DataFrame(comparison_rows).to_string(index=False))


if __name__ == "__main__":
    main()
