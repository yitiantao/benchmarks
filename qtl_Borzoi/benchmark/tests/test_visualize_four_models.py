from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from qtl_benchmark.visualize_four_models import (
    DNA_FM_NEW,
    DNA_FM_NEW_BF16,
    MODEL_ORDER,
    ModelResultPaths,
    build_dna_fm_checkpoint_deltas,
    build_dna_fm_precision_deltas,
    build_export_tables,
    build_ntv3_dna_fm_deltas,
    build_overview_table,
    load_metric_tables,
)


class VisualizeFourModelsTest(unittest.TestCase):
    def _write_model(self, root: Path, model: str) -> ModelResultPaths:
        model_root = root / model
        model_root.mkdir()
        eqtl = pd.DataFrame(
            {
                "tissue": ["Liver", "Heart_Left_Ventricle"],
                "eligible": [True, True],
                "auroc_sign": [0.6, 0.8],
                "spearmanr": [0.2, 0.4],
                "pearsonr": [0.1, 0.3],
                "auroc_class": [0.7, 0.9],
            }
        )
        eqtl_path = model_root / "eqtl.tsv"
        eqtl.to_csv(eqtl_path, sep="\t", index=False)
        task_paths = {}
        for task in ("sqtl", "paqtl", "ipaqtl"):
            table = pd.DataFrame(
                {
                    "task": [task, task],
                    "max_distance": [500, 10000],
                    "auroc_mean": [0.6, 0.7],
                    "auroc_std": [0.01, 0.02],
                    "auprc_mean": [0.61, 0.71],
                    "auprc_std": [0.02, 0.03],
                }
            )
            task_paths[task] = model_root / f"{task}.tsv"
            table.to_csv(task_paths[task], sep="\t", index=False)
        return ModelResultPaths(
            name=model,
            eqtl=eqtl_path,
            sqtl=task_paths["sqtl"],
            paqtl=task_paths["paqtl"],
            ipaqtl=task_paths["ipaqtl"],
        )

    def test_loads_all_models_and_aggregates_organs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [self._write_model(root, model) for model in MODEL_ORDER]

            eqtl, matched = load_metric_tables(paths)
            tissue, organ, matched_export = build_export_tables(eqtl, matched)

            self.assertEqual(set(tissue.model), set(MODEL_ORDER))
            self.assertEqual(set(tissue.organ_group), {"heart", "liver"})
            self.assertEqual(len(organ), 12)
            self.assertEqual(len(matched_export), 36)
            alpha_liver = organ[
                (organ.model == "AlphaGenome") & (organ.organ_group == "liver")
            ].iloc[0]
            self.assertAlmostEqual(alpha_liver.auroc_sign, 0.6)

            overview = build_overview_table(tissue, matched_export)
            self.assertEqual(
                set(overview.benchmark), {"eQTL", "sQTL", "paQTL", "iPaQTL"}
            )

    def test_ntv3_dna_fm_delta_is_dna_fm_minus_ntv3(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [self._write_model(root, model) for model in MODEL_ORDER]
            dna_eqtl = pd.read_csv(paths[-1].eqtl, sep="\t")
            dna_eqtl.loc[dna_eqtl.tissue == "Liver", "auroc_sign"] = 0.75
            dna_eqtl.to_csv(paths[-1].eqtl, sep="\t", index=False)

            eqtl, matched = load_metric_tables(paths)
            tissue, organ, matched_export = build_export_tables(eqtl, matched)
            tissue_delta, organ_delta, matched_delta = build_ntv3_dna_fm_deltas(
                tissue, organ, matched_export
            )

            liver = tissue_delta[tissue_delta.tissue == "Liver"].iloc[0]
            self.assertAlmostEqual(liver.delta_auroc_sign, 0.15)
            self.assertAlmostEqual(
                organ_delta[organ_delta.organ_group == "liver"]
                .iloc[0]
                .delta_auroc_sign,
                0.15,
            )
            self.assertTrue((matched_delta.delta_auroc_mean == 0).all())

    def test_checkpoint_delta_is_new_minus_old(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [self._write_model(root, model) for model in MODEL_ORDER]
            new_path = next(path for path in paths if path.name == DNA_FM_NEW_BF16)
            new_eqtl = pd.read_csv(new_path.eqtl, sep="\t")
            new_eqtl.loc[new_eqtl.tissue == "Liver", "auroc_sign"] = 0.55
            new_eqtl.to_csv(new_path.eqtl, sep="\t", index=False)

            eqtl, matched = load_metric_tables(paths)
            tissue, organ, matched_export = build_export_tables(eqtl, matched)
            tissue_delta, organ_delta, matched_delta = (
                build_dna_fm_checkpoint_deltas(tissue, organ, matched_export)
            )

            liver = tissue_delta[tissue_delta.tissue == "Liver"].iloc[0]
            self.assertAlmostEqual(liver.step150000_auroc_sign, 0.6)
            self.assertAlmostEqual(liver.step340000_auroc_sign, 0.55)
            self.assertAlmostEqual(liver.delta_auroc_sign, -0.05)
            self.assertTrue((matched_delta.delta_auroc_mean == 0).all())
            self.assertEqual(
                set(organ_delta.columns) & {"step150000_auroc_sign", "step340000_auroc_sign"},
                {"step150000_auroc_sign", "step340000_auroc_sign"},
            )

    def test_precision_delta_is_fp32_minus_bf16(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [self._write_model(root, model) for model in MODEL_ORDER]
            fp32_path = next(path for path in paths if path.name == DNA_FM_NEW)
            fp32_eqtl = pd.read_csv(fp32_path.eqtl, sep="\t")
            fp32_eqtl.loc[fp32_eqtl.tissue == "Liver", "auroc_sign"] = 0.75
            fp32_eqtl.to_csv(fp32_path.eqtl, sep="\t", index=False)

            eqtl, matched = load_metric_tables(paths)
            tissue, organ, matched_export = build_export_tables(eqtl, matched)
            tissue_delta, organ_delta, matched_delta = build_dna_fm_precision_deltas(
                tissue, organ, matched_export
            )

            liver = tissue_delta[tissue_delta.tissue == "Liver"].iloc[0]
            self.assertAlmostEqual(liver.bf16_auroc_sign, 0.6)
            self.assertAlmostEqual(liver.fp32_auroc_sign, 0.75)
            self.assertAlmostEqual(liver.delta_auroc_sign, 0.15)
            self.assertTrue((matched_delta.delta_auroc_mean == 0).all())
            self.assertAlmostEqual(
                organ_delta[organ_delta.organ_group == "liver"]
                .iloc[0]
                .delta_auroc_sign,
                0.15,
            )

    def test_rejects_missing_result_sets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [self._write_model(root, model) for model in MODEL_ORDER[:3]]
            with self.assertRaisesRegex(ValueError, "Expected exactly"):
                load_metric_tables(paths)


if __name__ == "__main__":
    unittest.main()
