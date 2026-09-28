# DNA Foundation Model benchmarks 与下游数据

本目录收集可直接使用的、已有公开发布版本的 DNA foundation model benchmark 和下游数据。这里没有从原始测序数据重新计算标签；下载日期为 **2026-08-27**。为保证可追溯性，下文记录了发布方、固定版本、原始链接、论文和许可状态。

> 使用提醒：代码仓库的开源许可证不一定覆盖仓库中引用或再分发的生物数据。正式发表或再分发前，请同时检查每个上游数据库/论文的引用与使用条款。人类数据仅收录公开汇总或处理结果，不包含受控访问的个体级基因型、表型或原始测序数据。

## 目录总览

| 目录 | 内容 | 本地状态 | 主要来源 |
|---|---|---:|---|
| `nucleotide_transformer_revised/` | Nucleotide Transformer 修订版 18 个分类任务；启动子、增强子、剪接位点和 10 种组蛋白标记；FNA 与 Parquet | 完整，565 MB | [Hugging Face 官方数据集](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised) |
| `dna_foundation_benchmark_2025/` | 2025 综合 benchmark：57 个序列分类任务，以及致病变异、causal QTL 和 TAD 任务 | 数据已解压，4.7 GB；含官方代码 | [数据](https://huggingface.co/datasets/hfeng3/dna_foundation_benchmark_dataset)；[代码](https://github.com/ChongWuLab/dna_foundation_benchmark) |
| `BEND/` | BEND：gene finding、enhancer annotation、chromatin accessibility、CpG methylation、histone modification、variant effects | 官方代码与官方数据下载目录 | [BEND](https://github.com/frederikkemarin/BEND) |
| `splicing_variants/GTEx_v8_sQTL/` | GTEx V8、49 个组织的显著 sQTL pairs、conditionally independent sQTL 和 LeafCutter phenotype-to-gene 映射 | 原始归档与解包文件均保留，约 1.2 GB | [GTEx Portal](https://gtexportal.org/home/downloads/adult-gtex/qtl) |
| `splicing_variants/Vex-seq/` | Vex-seq 处理结果：read counts、PSI、delta PSI、变异位置、剪接位点分数等；另存 GEO processed archive | 完整 | [作者仓库](https://github.com/scottiadamson/Vex-seq)；[GEO GSE113163](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE113163) |
| `splicing_variants/MaPSy_GSE201856/` | 已处理的 MaPSy 变异索引、效应值、参考表和 input/output FASTA（5,224 alleles） | 完整，约 20 MB | [GEO GSE201856](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE201856) |
| `splicing_variants/Spliceformer/` | 冰岛 sQTL/ClinVar 对应的官方模型、权重和评测 notebooks | 完整作者仓库 | [benniatli/Spliceformer](https://github.com/benniatli/Spliceformer) |
| `MFASS/` | Massively parallel functional assay of splicing；SNV/SRE 文库、处理结果及配套脚本/参考文件 | 原有完整作者仓库，4.3 GB | [KosuriLab/MFASS](https://github.com/KosuriLab/MFASS) |
| `clinvar/` | ClinVar 与冰岛 whole-blood sQTL benchmark；SpliceAI-10k 和 Transformer-45k delta scores | 原有 9 个文件，MD5 全部通过，133 MB | [Zenodo 14109868](https://doi.org/10.5281/zenodo.14109868) |

## 按任务分类的 benchmark 总表

下表优先采用原 benchmark 论文的评价口径；统一评测层对普通分类任务同时报告 Accuracy、macro-F1、MCC、AUROC 和 AUPRC。样本数中的 `test` 表示固定测试集大小，`variant` 表示变异记录数，`window` 表示基因组窗口数；带“参考集”的项目不是现成的普通 train/test 分类任务。

| 任务类别 | 具体测试 | 本地数据集/目录 | 主要 metric | 原始 benchmark / 论文来源 | 大约测试规模与状态 |
|---|---|---|---|---|---|
| VEP：致病变异 | pathogenic coding vs common variant；REF/ALT 表征距离或监督分类 | `dna_foundation_benchmark_2025/data/pathogenic/`；`BEND/data/variant_effects/variant_effects_disease.bed` | AUROC（主）；也可报告 AUPRC | [Feng et al., Nature Communications 2025](https://doi.org/10.1038/s41467-025-65823-8)；BEND 的 ClinVar/DeepSEA protocol：[Marin et al., ICLR 2024](https://openreview.net/forum?id=uKB4cFNQFg) | 2025 集约 39,652 variants；BEND disease 集约 295,496 variants |
| VEP：eQTL | causal eQTL vs matched non-causal；或表达效应 variant ranking | `dna_foundation_benchmark_2025/data/causal/eqtl/`；`BEND/data/variant_effects/variant_effects_expression.bed` | AUROC；REF/ALT embedding-distance ranking | Feng et al. 2025；BEND expression task 继承 [DeepSEA](https://doi.org/10.1038/nmeth.3547) protocol | 1,896 positive + 1,896 negative blood eQTL；BEND 约 105,264 variants |
| VEP：sQTL | causal sQTL vs matched negative | `dna_foundation_benchmark_2025/data/causal/sqtl/` | AUROC；REF/ALT embedding distance | [Feng et al. 2025](https://doi.org/10.1038/s41467-025-65823-8) | 540 positive + 540 negative blood sQTL，现成配对测试集 |
| VEP：RNA processing QTL | ipaQTL / paQTL causal vs matched negative | `dna_foundation_benchmark_2025/data/causal/{ipaqtl,paqtl}/` | AUROC；REF/ALT embedding distance | [Feng et al. 2025](https://doi.org/10.1038/s41467-025-65823-8) | ipaQTL 116+116；paQTL 142+142 variants |
| sQTL 汇总关联 | 跨组织 cis-sQTL effect/显著性与 fine-mapping 参考 | `splicing_variants/GTEx_v8_sQTL/` | 原生统计量为 nominal/beta-adjusted p-value、slope；模型评测需另定义 AUROC/AUPRC 或相关性 | [GTEx Consortium, Science 2020](https://doi.org/10.1126/science.aaz1776) | 49 tissues；显著 pairs、sGenes、independent sQTL 与 LeafCutter groups 均齐全；参考集，非固定 test split |
| sQTL 分类 | Icelandic whole-blood sQTL vs no-sQTL，按四类 splice delta 最大值评分 | `clinvar/*_sqtl_delta.vcf`、`clinvar/*_no_sqtl_delta.vcf` | AUPRC（原论文主口径；bootstrap）；可补 AUROC | [Jónsson et al., Communications Biology 2024](https://doi.org/10.1038/s42003-024-07298-9) | 两类 VCF 各约 40.5k/81.0k 行（含表头；实际以 adapter 对齐后的记录为准） |
| ClinVar 剪接致病性分类 | benign/likely benign vs pathogenic/likely pathogenic splice variants | `clinvar/clinvar_splice_variants.tsv`；`*_clinvar_delta.vcf` | AUPRC（主）、AUROC；原 notebook 另报阈值 accuracy | [Spliceformer / Jónsson et al. 2024](https://doi.org/10.1038/s42003-024-07298-9) | 49,475 ClinVar splice variants；严格二分类标签约 39,379（1,063 benign、38,316 pathogenic）；模型成功评分约 46,073 |
| MPRA 剪接效应回归 | Vex-seq variant-induced ΔPSI prediction | `splicing_variants/Vex-seq/` | Pearson r（原 MMSplice 比较口径）；可补 Spearman r、MAE/RMSE | [Adamson et al., Genome Biology 2018](https://doi.org/10.1186/s13059-018-1434-x)；模型比较见 [Cheng et al., Genome Biology 2019](https://doi.org/10.1186/s13059-019-1653-z) | 2,055 variants；MMSplice/CAGI split 为 957 train + 1,098 test |
| MPRA 剪接效应回归 | MaPSy mutant/wild-type allelic splicing effect | `splicing_variants/MaPSy_GSE201856/` | Pearson/Spearman r；连续效应也可报 MAE/RMSE | GSE201856 发布数据；实验范式源于 [Soemedi et al., Nature Genetics 2017](https://doi.org/10.1038/ng.3837) | 5,224 reported alleles；当前为现代人/古人类扩展数据，参考集，尚无统一固定 split |
| MPRA 剪接效应/分类 | MFASS ΔPSI 与 strong loss-of-function | `MFASS/processed_data/snv/snv_data_clean.txt` | 回归：Pearson/Spearman r；二分类：AUPRC（主）、AUROC | [Cheung et al., Molecular Cell 2019](https://doi.org/10.1016/j.molcel.2018.12.005) | 32,669 rows；strong-LOF 标签约 1,983 positive、29,048 negative，另有 1,638 NA |
| 剪接位点识别 | donor / acceptor / neither 序列分类 | `nucleotide_transformer_revised/splice_sites_*`；`dna_foundation_benchmark_2025/data/splice/` | MCC（统一主 metric）、macro-F1、Accuracy、AUROC/AUPRC | [Nucleotide Transformer, Nature Methods 2025](https://doi.org/10.1038/s41592-024-02523-z)；[Feng et al. 2025](https://doi.org/10.1038/s41467-025-65823-8) | NT 三个任务各 3,000 test sequences；2025 套件另含 acceptor、donor 和 splice-site-type tasks |
| 启动子/增强子/调控元件 | promoter、enhancer activity/type/strength、regulatory-region classification | `nucleotide_transformer_revised/{promoter_*,enhancers*}/`；2025 `prom/`、`enhancers/`、`genomic_benchmark/` | MCC、macro-F1、Accuracy、AUROC/AUPRC | Nucleotide Transformer 2025；Feng et al. 2025；后者整合 GUE 与 Genomic Benchmarks | NT 每项约 212–3,000 test sequences；2025 套件为多个固定 train/test 数据集 |
| TFBS / 染色质可及性 | human/mouse TFBS、DNase/open chromatin、多细胞类型 accessibility | 2025 `tf/`、`mouse/`、`iDHS-EL/`、`genomic_benchmark/open_chromatin_region/`；`BEND/data/chromatin_accessibility/` | MCC、macro-F1、Accuracy、AUROC/AUPRC；BEND 多标签任务按 task config | Feng et al. 2025；[BEND, ICLR 2024](https://openreview.net/forum?id=uKB4cFNQFg) | 2025 含 10 个 TFBS 和多个 accessibility 固定任务；BEND 约 2.06M windows |
| 表观遗传标记 | histone marks、CpG/5mC/6mA/4mC 分类或逐位点预测 | `nucleotide_transformer_revised/H*`；2025 `EMP/`、`iDNA_ABF/`、`deep4mc/`；BEND histone/CpG | MCC、macro-F1、Accuracy、AUROC/AUPRC；逐位点任务依 BEND config | Nucleotide Transformer 2025；Feng et al. 2025；BEND 2024 | NT 10 个任务各约 776–3,000 test sequences；BEND 约 625k histone windows、959k CpG windows |
| 基因结构预测 | exon/intron/splice-state 的逐碱基 gene finding | `BEND/data/gene_finding/` | nucleotide-level F1 / MCC（按 BEND task config） | [BEND, ICLR 2024](https://openreview.net/forum?id=uKB4cFNQFg)，标签源自 GENCODE | 约 5,978 long genomic windows，标签在配套 HDF5 |
| 增强子注释 | enhancer overlap/annotation | `BEND/data/enhancer_annotation/` | AUROC/AUPRC 或 F1（按 BEND task config） | BEND 2024，整合 ENCODE enhancer annotation | 约 286 genomic regions；小样本、长序列任务 |
| 三维基因组 | TAD boundary vs matched background，6 kb sequence classification | `dna_foundation_benchmark_2025/data/TAD/` | AUROC、MCC、F1、Accuracy | Feng et al. 2025；标签来自 Basenji Hi-C insulation release | 1,500 boundary + 1,500 background sequences |
| 物种/序列属性分类 | coding vs noncoding、human vs worm、COVID variant lineage 等 | 2025 `genomic_benchmark/{coding,human_vs_worm}/`、`virus/covid_variants/` | MCC、macro-F1、Accuracy、AUROC/AUPRC | Feng et al. 2025，整合 Genomic Benchmarks 与 GUE | 多个现成固定 train/test 分类集；精确规模可由 `unified_eval manifest` 查看 |

这里的“VEP”指 **variant effect prediction**，不是 Ensembl Variant Effect Predictor 软件本身；不过 2025 pathogenic coding 表和 BEND disease variants 中包含或使用了 Ensembl VEP consequence annotation。对于类别极不平衡的 ClinVar、MFASS 和 QTL 任务，应优先比较 AUPRC，并同时报告 AUROC；对于 ΔPSI/allelic-ratio 等连续实验值，应优先保留 Pearson/Spearman 相关性并补充误差指标。

## 1. Nucleotide Transformer revised benchmark

- 固定 revision：`851f9946252e90c665cdb3cc3eedb78f1f26197c`。
- 18 个任务：`promoter_all`、`promoter_tata`、`promoter_no_tata`、`enhancers`、`enhancers_types`、`splice_sites_all`、`splice_sites_acceptors`、`splice_sites_donors`、`H2AFZ`、`H3K27ac`、`H3K27me3`、`H3K36me3`、`H3K4me1`、`H3K4me2`、`H3K4me3`、`H3K9ac`、`H3K9me3`、`H4K20me1`。
- 所有任务使用 chromosome-held-out test set；各任务目录中同时提供 `train/test.fna` 和 `train/test.parquet`。
- 上游标签来源：ENCODE（K562 histone ChIP-seq）、SCREEN/ENCODE（enhancer）、Eukaryotic Promoter Database（promoter）、GENCODE v44（splice sites）。具体 accession 和样本数见该目录自带的 `README.md`。
- 论文：[Dalla-Torre et al., Nature Methods 2025](https://doi.org/10.1038/s41592-024-02523-z)。
- 许可：Hugging Face 数据卡未声明统一 license；应遵守 ENCODE、SCREEN、EPD 和 GENCODE 的各自条款。

## 2. DNA Foundation Models Benchmarking (2025)

- 数据固定 revision：`1a3a47ef2de823de858a2e26bad510fab513648d`；下载包 `data_processed.zip` 保留，内容解压在 `data/`。
- 官方代码 commit：`3f4c81ce066f3c47422a83466b085aac1a6be902`，位于 `code/`。
- 序列分类共 57 个 ready-made train/test 数据集：
  - GUE 类任务：human/mouse TFBS、promoter、splice site、yeast epigenetic marks、COVID variants；
  - Genomic Benchmarks：coding、human-vs-worm、enhancer、open chromatin、promoter、regulatory-region type；
  - 其他下游任务：enhancer activity/strength、5mC/6mA/4mC、DNase I、跨物种/细胞系 promoter。
- 额外遗传任务：`data/pathogenic/`（致病 coding variants）、`data/causal/{eqtl,sqtl,ipaqtl,paqtl}/`（causal QTL）以及 `data/TAD/`（6 kb TAD boundary classification；含 hg38 reference）。其中 `causal/sqtl` 是现成整理后的 sQTL variant-effect 数据，可与 GTEx 全组织汇总数据互补。
- 论文：[Feng et al., Nature Communications 2025](https://doi.org/10.1038/s41467-025-65823-8)。上游任务与原论文引用详见论文 Methods/Supplementary Information；TAD 数据来自 [Basenji Hi-C insulation release](https://console.cloud.google.com/storage/browser/basenji_hic/insulation)。
- 数据仓库声明 Apache-2.0；代码仓库 MIT。上游数据库的条款仍分别适用。

## 3. BEND

- 官方仓库 commit：`ac6e80c75e09d83cf47a7b4bcf0e44599c5706cf`。
- 数据格式主要为 BED 坐标及 HDF5 标签，任务包括 gene finding、enhancer annotation、chromatin accessibility、CpG methylation、histone modification，以及 disease/eQTL variant effects。参考序列需结合相应 genome build 使用。
- 数据由仓库自带的 `scripts/download_bend.py` 从 BEND 官方 ERDA share 下载，目录结构保持官方形式。
- 主要上游来源包括 GENCODE、ENCODE、DeepSEA/Basset 处理资源、全基因组 CpG methylation 数据及 ClinVar/eQTL variant sets；逐任务原始论文和 BibTeX 已完整保留在 `BEND/README.md`。
- 论文：[Marin et al., ICLR 2024](https://openreview.net/forum?id=uKB4cFNQFg)。代码许可 BSD-3-Clause；上游数据许可分别适用。

## 4. GTEx V8 sQTL（现成汇总结果）

路径：`splicing_variants/GTEx_v8_sQTL/`。

收录的是 GTEx 官方已经完成 LeafCutter + QTL mapping 后发布的结果，不是本地重算：

- `GTEx_Analysis_v8_sQTL/`：49 个组织；每个组织含 `*.v8.sgenes.txt.gz` 与 `*.v8.sqtl_signifpairs.txt.gz`；
- `GTEx_Analysis_v8_sQTL_independent/`：逐组织 conditionally independent sQTL；
- `GTEx_Analysis_v8_sQTL_groups/`：LeafCutter phenotype/intron cluster 到 gene 的映射；
- 三个原始 tar 包及官方 `README_eQTL_v8.txt` 同时保留，便于校验和重新解包。

官方源：[GTEx V8 QTL 下载页](https://gtexportal.org/home/downloads/adult-gtex/qtl)。本地使用的现行官方对象路径前缀为 `https://storage.googleapis.com/adult-gtex/bulk-qtl/v8/single-tissue-cis-qtl/`。主论文：[GTEx Consortium, Science 2020](https://doi.org/10.1126/science.aaz1776)。GTEx 数据使用与引用政策见 [GTEx policies](https://gtexportal.org/home/aboutGTEx#dataUseAgreement)。这里只收录 open-access summary results；约 939 GB 的 all-associations requester-pays 数据和受控个体级数据未下载。

## 5. Vex-seq

- 作者仓库 commit：`36d9c4a6e5e74def46b3cada69d3166f08c54bd5`。
- 最适合直接建模的文件包括 `processed_files/delta_PSI_values.tsv`、`processed_files/read_counts_PSI_values.tsv`、`processed_files/Vex-seq_positions.tsv` 和 `processed_files/splice_site_max_ent.tsv`。
- `GSE113163_RAW.tar` 是 GEO 页面所称的 processed supplementary archive；原始 reads 在 SRA `SRP140446`，本目录未重复下载原始 reads。
- 论文：[Adamson et al., Genome Biology 2018](https://doi.org/10.1186/s13059-018-1434-x)。来源：[作者仓库](https://github.com/scottiadamson/Vex-seq)、[NCBI GEO GSE113163](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE113163)。

## 6. MaPSy (GSE201856)

- 这是 GEO 已处理版本，包含 5,224 个现代人/尼安德特人/丹尼索瓦相关 alleles，其中发布方报告 969 个 exonic splicing mutations。
- `*_variant_ind.txt.gz` 与 `*_variant_val.txt.gz` 为变异索引和结果值，`*_table_ref.txt.gz` 为参考注释，两个 FASTA 为 assay input/output sequence；`GSE201856_RAW.tar` 也按 GEO 发布形式保留。
- 来源与实验设计：[NCBI GEO GSE201856](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE201856)，BioProject `PRJNA833320`。原始 FASTQ 未下载。

## 7. MFASS

- 原有作者仓库 commit：`9a8e4f27106be52aeb11acad27f95f5cded663a8`。
- 推荐的直接使用文件由作者标注在 `MFASS/README.md`：SNV 库的 `processed_data/snv/snv_data_clean.txt`，以及 SRE 库相应处理结果；仓库还包含 library design、read counts、参考注释和分析脚本。
- 论文：[Cheung et al., Molecular Cell 2019](https://doi.org/10.1016/j.molcel.2018.12.005)。来源：[KosuriLab/MFASS](https://github.com/KosuriLab/MFASS)。

## 8. ClinVar / Icelandic whole-blood sQTL benchmark

`clinvar/` 的 9 个文件来自 Zenodo 数据集 **“Supplementary Data for Transformers Significantly Improve Splice Site Prediction”**：

- `clinvar_splice_variants.tsv`：ClinVar splice variants 与临床标签；
- `splice_site_annotation_gtex.tsv`：GTEx V8 splice-site annotation；
- `splice_site_annotation_icelandic_whole_blood_plus_gtex.tsv`：冰岛 whole-blood + GTEx annotation；
- `*_sqtl_delta.vcf` / `*_no_sqtl_delta.vcf`：冰岛 whole-blood sQTL positive/negative 的 SpliceAI-10k 与 Transformer-45k scores；
- `*_clinvar_delta.vcf`：ClinVar benchmark scores。

所有文件的 MD5 已在 2026-08-27 与 Zenodo 文件清单核对并完全一致。来源：[Zenodo record 14109868](https://doi.org/10.5281/zenodo.14109868)，许可 CC BY 4.0。论文/数据引用信息见该 Zenodo record。

对应官方实现位于 `splicing_variants/Spliceformer/`，commit `875b6a004b13bde0718f22a887bd3a5a812a1832`。sQTL delta 与 PR-AUC 见 `Code/get_sqtl_delta_for_transformer.ipynb`，ClinVar 见 `Code/get_clinvar_delta_for_transformer.ipynb`，预训练权重在 `Results/PyTorch_Models/`。论文：[Jónsson et al., Communications Biology 2024](https://doi.org/10.1038/s42003-024-07298-9)，代码许可 MIT。

## 9. 统一评测入口与可视化

`unified_eval/` 是新增的只读评测层，不修改原始数据或官方代码。它自动发现 18 个 NT 和 57 个 2025 benchmark 分类任务，统一 CSV/FNA split、k-mer baseline、外部预测评分、JSONL 结果记录及 HTML/PNG 报告；冰岛 sQTL adapter 复用作者的 max-delta + PR-AUC 定义。使用方法见 `unified_eval/README.md`。

## 推荐使用方式

- 通用短序列分类：先用 `nucleotide_transformer_revised/` 或 `dna_foundation_benchmark_2025/data/`。
- 长序列/逐碱基任务：使用 `BEND/` 和综合包的 `TAD/`。
- 剪接变异 benchmark：实验效应用 MFASS、Vex-seq、MaPSy；群体关联用 GTEx V8 sQTL；临床变异用 `clinvar/`。
- 避免数据泄漏：同一上游资源（尤其 GUE、Genomic Benchmarks、GTEx、ClinVar）可能在多个套件中以不同预处理形式重复出现；跨 benchmark 合并训练前应按 variant、坐标、sequence 和 chromosome 去重。
