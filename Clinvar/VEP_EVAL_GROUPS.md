# VEP-eval ClinVar 功能任务：严格复现规则与统计

本页解释 `VEP_ClinVar_Benchmarking_RefSeq.ipynb` 最终生成哪些 ClinVar 功能组、
每组依据什么字段，以及三个 ClinVar 版本的实际处理结果。

实现代码：[`build_vep_eval_groups.py`](build_vep_eval_groups.py)

## 复现范围与验证

- 参考 notebook：`/data/yitian_workspace/DNA/benchmarks/VEP-eval/VEP_ClinVar_Benchmarking_RefSeq.ipynb`
- 当前目录的 notebook 副本与参考文件 SHA-256 相同：
  `ba83644d335d1afbc41f55a13651cddc6e5980c86cbe3c2da6807dbfb4d1b60f`
- MANE：GRCh38 release 1.5。
- Genome：notebook 指定的 UCSC `hg38.fa.gz`；下载文件 MD5 与 UCSC 官方值
  `1c9dcaddfa41027f17cd8f7a82c7293b` 一致。
- RefSeq strand：UCSC `ncbiRefSeq.txt.gz`。
- 参考 `VEP-eval` 仓库没有被修改。
- 优化后的 MANE 区间算法与 notebook 原始逐行扫描算法在随机 120 条候选上逐字段
  比较，gene/CDS/codon/UTR/intron/splice/RNA/other 全部一致。
- 2026-02 最终 exact B/P 为 242,132（Benign 172,617；Pathogenic 69,515），
  与参考仓库已执行的 figure notebook 完全一致。

参考资源的路径、大小和 SHA-256 见
[`processed/vep_eval/reference_manifest.json`](processed/vep_eval/reference_manifest.json)。

## 处理顺序

每个版本严格遵循以下语义：

1. `Assembly == GRCh38`。
2. `OriginSimple != somatic`。
3. `Chromosome` 仅 1–22、X、Y。
4. `Type == single nucleotide variant`。
5. `ClinicalSignificance` 仅 Benign、Likely benign、Benign/Likely benign、
   Pathogenic、Likely pathogenic、Pathogenic/Likely pathogenic。
6. 按 `ReviewStatus` 映射星级，保留至少 1 星。
7. `ReferenceAlleleVCF`、`AlternateAlleleVCF` 都必须是单个 A/C/G/T。
8. 用 UCSC hg38 检查 `PositionVCF` 的参考碱基必须等于 REF，并要求上下游各
   32,768 bp 的窗口完整。
9. 用 MANE v1.5 注释 CDS、start/stop codon、UTR、intron 和 splice 边界。
10. 从 ClinVar `Name` 中解析 HGVS 蛋白变化、显式 splice 和 RNA transcript。
11. 组合这些基础字段生成 35 个功能组、coding/noncoding 和正负链字段。
12. 主 benchmark 只保留 exact Benign（0）与 Pathogenic（1）；likely 标签保留在
    `clinvar_benchmark_all_labels.csv.gz`。

## 分组实际使用的字段

### 来自 ClinVar 原始文件

| 字段 | 用途 |
|---|---|
| `Assembly` | 只保留 GRCh38 |
| `OriginSimple` | 排除 exact somatic |
| `Chromosome` | 只保留常规染色体 |
| `Type` | 只保留 single nucleotide variant |
| `ClinicalSignificance` | 映射 Benign/Pathogenic/likely 标签 |
| `ReviewStatus` | 映射 ClinVar 星级 |
| `Name` | 解析 NM_ transcript、蛋白变化、`±1/±2` splice、`NR_` RNA |
| `PositionVCF` | GRCh38 坐标、MANE 匹配、FASTA 校验 |
| `ReferenceAlleleVCF` / `AlternateAlleleVCF` | 单碱基筛选与 REF 校验 |

### 从 `Name` 派生的字段

| 派生字段 | notebook 判定 |
|---|---|
| `ClinVarName_coding_sequence` | 能从 `(p....)` 成功解析氨基酸位置、REF 和 ALT |
| `ClinVarName_missense` | 蛋白 REF 与 ALT 都属于 20 种标准氨基酸 |
| `ClinVarName_synonymous` | 蛋白 ALT 为 `=` |
| `ClinVarName_stop_gain` | 蛋白 ALT 为 `*` |
| `ClinVarName_splice` | `Name` 匹配数字后紧跟 `+1`、`+2`、`-1` 或 `-2`，再跟 A/C/G/T |
| `ClinVarName_RNA_gene` | `Name` 包含 `NR_` |
| `ClinVarName_refseq_ids` | `Name` 中第一个 `NM_数字.版本` |

### 从 MANE v1.5 派生的字段

| 派生字段 | notebook 判定 |
|---|---|
| `MANE_coding_sequence` | 坐标同时落在同一 MANE mRNA 的 exon 和 CDS |
| `MANE_start_codon` | 落在 CDS 起始端 3 个碱基；考虑 transcript strand |
| `MANE_stop_codon` | 落在 CDS 终止端 3 个碱基；考虑 transcript strand |
| `MANE_five_prime_UTR` | 落在 exon 中、CDS 外，并处于 strand 对应的 5′ 端 |
| `MANE_three_prime_UTR` | 落在 exon 中、CDS 外，并处于 strand 对应的 3′ 端 |
| `MANE_mRNA_intron` | 落在 mRNA span 内，但不落在该 transcript 的 exon/CDS |
| `MANE_mRNA_splice` | intron 位点恰好为任一 exon 边界外的 1 或 2 bp |
| `MANE_snRNA` / `MANE_snoRNA` | 落在对应 MANE RNA feature 中 |

### 组合信号

下面用这些缩写描述最终规则：

| 缩写 | 实际字段/表达式 |
|---|---|
| `C` | `ClinVarName_coding_sequence OR MANE_coding_sequence` |
| `A` | `MANE_start_codon` |
| `M` | `ClinVarName_missense` |
| `Y` | `ClinVarName_synonymous` |
| `G` | `ClinVarName_stop_gain` |
| `L` | `ClinVarName_AAREF == '*' OR MANE_stop_codon` |
| `U5` | `MANE_five_prime_UTR` |
| `U3` | `MANE_three_prime_UTR` |
| `I` | `MANE_mRNA_intron` |
| `S` | `MANE_mRNA_splice OR ClinVarName_splice` |
| `R` | `ClinVarName_RNA_gene OR MANE_snRNA OR MANE_snoRNA` |
| `B` | `NOT A AND NOT M AND NOT Y AND NOT G AND NOT L` |

## 最终生成哪些组，以及怎么分

`¬` 表示 NOT。表中未写出的条件与该组判定无关，这一点与原 notebook 一致。

| 组 | 布尔条件 |
|---|---|
| coding | `C` |
| noncoding | `¬C` |
| start loss | `A & ¬U5 & ¬U3 & ¬I` |
| start loss + 3′UTR | `A & ¬U5 & U3 & ¬I` |
| start loss + 5′UTR | `A & U5 & ¬U3 & ¬I` |
| start loss + intron | `A & ¬U5 & ¬U3 & I` |
| missense | `¬A & M & ¬U5 & ¬U3 & ¬I` |
| missense + 3′UTR | `¬A & M & ¬U5 & U3 & ¬I` |
| missense + 5′UTR | `¬A & M & U5 & ¬U3 & ¬I` |
| missense + intron | `¬A & M & ¬U5 & ¬U3 & I` |
| synonymous | `Y & ¬U5 & ¬U3 & ¬I & ¬L` |
| synonymous + 3′UTR | `Y & ¬U5 & U3 & ¬I & ¬L` |
| synonymous + 5′UTR | `Y & U5 & ¬U3 & ¬I & ¬L` |
| synonymous + intron | `Y & ¬U5 & ¬U3 & I & ¬L` |
| stop synonymous | `Y & ¬U5 & ¬U3 & ¬I & L` |
| stop synonymous + 3′UTR | `Y & ¬U5 & U3 & ¬I & L` |
| stop gain | `G & ¬U5 & ¬U3 & ¬I` |
| stop gain + 3′UTR | `G & ¬U5 & U3 & ¬I` |
| stop gain + 5′UTR | `G & U5 & ¬U3 & ¬I` |
| stop gain + intron | `G & ¬U5 & ¬U3 & I` |
| stop loss | `¬Y & L & ¬U5 & ¬U3 & ¬I` |
| stop loss + 3′UTR | `¬Y & L & ¬U5 & U3 & ¬I` |
| stop loss + 5′UTR | `¬Y & L & U5 & ¬U3 & ¬I` |
| 5′UTR | `B & U5 & ¬U3 & ¬S & ¬I` |
| 5′UTR + 3′UTR | `B & U5 & U3 & ¬S & ¬I` |
| 5′UTR + intron | `B & U5 & ¬U3 & ¬S & I` |
| 5′UTR + splice | `B & U5 & ¬U3 & S & I` |
| 3′UTR | `B & ¬U5 & U3 & ¬S & ¬I & ¬R` |
| 3′UTR + intron | `B & ¬U5 & U3 & ¬S & I & ¬R` |
| 3′UTR + splice | `B & ¬U5 & U3 & S & I & ¬R` |
| 3′UTR + RNA gene | `B & ¬U5 & U3 & ¬S & ¬I & R` |
| splice | `B & ¬U5 & ¬U3 & S & I & ¬R` |
| intron (non-splice) | `B & ¬U5 & ¬U3 & ¬S & I & ¬R` |
| intron + RNA gene | `B & ¬U5 & ¬U3 & ¬S & I & R` |
| RNA gene | `B & ¬U5 & ¬U3 & ¬S & ¬I & R` |

另外还根据 `ClinVarName_refseq_ids` 在 UCSC `ncbiRefSeq` 中查 strand，生成
`group: +` 和 `group: -`。

## 论文/figure 重点使用的 14 个任务

notebook 后部选择以下 14 个主要组：

```text
coding, noncoding,
intron (non-splice), splice, 5'UTR, 3'UTR, RNA gene,
synonymous, start loss, stop loss, stop gain, missense,
missense + intron (non-splice), missense + 3'UTR
```

`coding/noncoding` 是覆盖全体的粗粒度二分，会与后面的详细功能任务重叠，因此
不能把下面所有行相加当作总样本数。

## 三版本处理总数

| 版本 | canonical 候选 | hg38 有效 all-label | exact B/P | REF mismatch | 窗口边界错误 |
|---|---:|---:|---:|---:|---:|
| 2025-03 | 1,284,123 | 1,284,098 | 240,351 | 5 | 20 |
| 2026-02 | 1,346,039 | 1,346,005 | 242,132 | 6 | 28 |
| 2026-09-14 | 1,437,333 | 1,437,296 | 245,566 | 8 | 29 |

## 14 个主要任务：exact Benign/Pathogenic 统计

每个单元格为 `总数 (Benign / Pathogenic)`；这里的总数是最终主 benchmark
中的 exact B/P 数量，不是该功能组的 all-label 数量。括号中的第一个数是
Benign，第二个数是 Pathogenic。

| 任务 | 2025-03 | 2026-02 | 2026-09-14 |
|---|---:|---:|---:|
| coding | 125,125 (69,352 / 55,773) | 125,931 (67,702 / 58,229) | 127,595 (67,333 / 60,262) |
| noncoding | 115,226 (104,682 / 10,544) | 116,201 (104,915 / 11,286) | 117,971 (106,031 / 11,940) |
| start loss | 770 (20 / 750) | 791 (19 / 772) | 785 (19 / 766) |
| missense | 43,504 (28,907 / 14,597) | 42,458 (27,273 / 15,185) | 42,383 (26,937 / 15,446) |
| missense + 3′UTR | 107 (56 / 51) | 100 (53 / 47) | 103 (52 / 51) |
| missense + intron | 302 (225 / 77) | 302 (224 / 78) | 294 (220 / 74) |
| synonymous | 39,640 (39,433 / 207) | 39,605 (39,397 / 208) | 39,608 (39,376 / 232) |
| stop gain | 39,870 (124 / 39,746) | 41,722 (134 / 41,588) | 43,467 (131 / 43,336) |
| stop loss | 61 (22 / 39) | 62 (22 / 40) | 62 (20 / 42) |
| 5′UTR | 2,356 (2,326 / 30) | 2,386 (2,354 / 32) | 2,433 (2,400 / 33) |
| 3′UTR | 11,713 (11,700 / 13) | 11,853 (11,840 / 13) | 12,220 (12,204 / 16) |
| splice | 9,488 (71 / 9,417) | 10,174 (72 / 10,102) | 10,771 (74 / 10,697) |
| intron (non-splice) | 86,944 (85,954 / 990) | 86,995 (85,959 / 1,036) | 87,700 (86,614 / 1,086) |
| RNA gene | 64 (58 / 6) | 75 (63 / 12) | 92 (79 / 13) |

### 为什么最新版的 missense 反而比 2025-03 少？

这不是 missense 变异总量减少，而是统计口径不同：

| missense 口径 | 2025-03 | 2026-09-14 | 变化 |
|---|---:|---:|---:|
| all-label（含 Benign、Likely benign、Pathogenic、Likely pathogenic） | 163,060 | 195,362 | +32,302 |
| exact B/P 主 benchmark | 43,504 | 42,383 | -1,121 |
| 其中 Benign | 28,907 | 26,937 | -1,970 |
| 其中 Pathogenic | 14,597 | 15,446 | +849 |
| Likely benign + Likely pathogenic | 119,556 | 152,979 | +33,423 |

因此最新版识别到的 missense 记录明显更多，但新增/保留记录中有更大比例属于
`Likely benign` 或 `Likely pathogenic`；主 benchmark 按 notebook 规则只保留
exact `Benign` 和 exact `Pathogenic`，likely 标签不进入这个总数。同时，版本更新
还会导致已有记录撤回、重分类或等位基因表示变化，所以不能把版本更新理解为简单
追加新记录。换句话说，`42,383` 是“最新版 missense 且 exact B/P”的数量，
不是“最新版所有 missense”的数量。

可以直接看出这个 benchmark 的类别先验非常强：stop gain、splice、start loss
几乎全为 Pathogenic；synonymous、UTR 和普通 intron 几乎全为 Benign。这正是
VEP-eval 论文讨论“variant-type composition 会抬高整体性能”的核心。

## 2026-02 中间特征规模

| 特征 | 数量（all-label、hg38 校验后） |
|---|---:|
| MANE CDS | 880,633 |
| MANE intron | 436,003 |
| MANE splice ±1/±2 | 35,852 |
| MANE 5′UTR | 6,435 |
| MANE 3′UTR | 23,523 |
| ClinVar HGVS missense | 191,746 |
| ClinVar HGVS synonymous | 627,690 |
| ClinVar HGVS stop gain | 62,746 |
| ClinVar `Name` 显式 splice | 35,898 |
| Union splice | 35,909 |

## 严格复现意味着保留的 notebook 行为

以下行为可能不理想，但为了与 benchmark 一致，本次没有擅自改变：

1. `OriginSimple != somatic` 会保留 `germline/somatic`。
2. 只检查 REF/ALT 都是 A/C/G/T，没有要求 `REF != ALT`。
3. `start loss` 实际表示“落在 start codon”，并未进一步检查 ALT 是否真的破坏起始密码子。
4. splice 组最终还要求 `MANE_mRNA_intron == 1`；只有 HGVS splice 信号但不在
   MANE intron 的记录不会进入纯 splice 组。
5. MANE GFF 使用 `lnc_RNA`、`antisense_RNA` 等名称，而 notebook 的循环检查
   `lncRNA`、`antisenseRNA`；严格复现后，最终 RNA union 实际只组合 `NR_`、
   MANE `snRNA` 和 `snoRNA`。
6. promoter 查询把 MANE 的 `chr1` 与 ClinVar 的 `1` 比较，因前缀不同不会命中；
   promoter 字段没有参与最终组，所以这里按 notebook 语义保留为无命中。
7. notebook 只报告 duplicate variant key 而不删除；本次三版均为 0 个重复 key。

## 输出文件

每个版本目录都包含：

| 文件 | 内容 |
|---|---|
| `clinvar_benchmark_all_labels.csv.gz` | exact + likely，含所有基础标记和 group 列 |
| `clinvar_benchmark.csv.gz` | exact B/P 主 benchmark，另含 Rule-based 类别和分数 |
| `clinvar_variants.csv.gz` | `#CHROM,POS,REF,ALT,ClinVar_label` 五列简版 |
| `filter_funnel.csv` | 每一步筛选与 hg38 校验计数 |
| `feature_counts.csv` | MANE/HGVS 中间特征计数 |
| `group_counts.csv` | 35 个功能组、正负链、四种标签计数 |

合并统计：

- [`processed/vep_eval/run_summary.csv`](processed/vep_eval/run_summary.csv)
- [`processed/vep_eval/all_feature_counts.csv`](processed/vep_eval/all_feature_counts.csv)
- [`processed/vep_eval/all_group_counts.csv`](processed/vep_eval/all_group_counts.csv)
