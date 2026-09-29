# ClinVar `variant_summary`：最简单的数据探查指南

三模型（Borzoi / NTv3 / AlphaGenome）的数据—模型—评测分层运行方法、单卡/多卡脚本和分数对齐限制见 [BENCHMARK_FRAMEWORK.md](BENCHMARK_FRAMEWORK.md)。

这个目录保存了三个 ClinVar `variant_summary` 版本，以及用于 VEP-eval benchmark
复现的统计结果。本 README 只回答三个问题：

1. 文件是什么格式？
2. 每一行和每一列是什么意思？
3. 怎样用一个很简单的脚本查看它？

## 一分钟运行

脚本只使用 Python 标准库，不需要安装 pandas：

```bash
python inspect_clinvar.py
```

默认查看最新版，并统计文件开头 100,000 行。指定其他版本：

```bash
python inspect_clinvar.py data/raw/variant_summary_2025-03.txt.gz
python inspect_clinvar.py data/raw/variant_summary_2026-02.txt.gz
```

展示 3 条示例，并扫描整个文件：

```bash
python inspect_clinvar.py data/raw/variant_summary_2026-02.txt.gz --rows 3 --scan 0
```

`--scan 0` 会解压并读取整个大文件，需要一些时间；平时快速了解格式时使用默认值
即可。

## 脚本会输出什么

`inspect_clinvar.py` 依次打印：

- 文件路径、压缩大小、格式和字段数；
- 全部 43 个字段名；
- Assembly、Type、ClinicalSignificance、ReviewStatus、OriginSimple 和 Chromosome
  的常见取值及比例；
- 缺失占比较高的字段；
- 若干条容易阅读的示例记录。

最短的核心读取代码其实只有这些：

```python
import csv
import gzip

with gzip.open("data/raw/variant_summary_2026-02.txt.gz", "rt") as f:
    rows = csv.DictReader(f, delimiter="\t")
    first_row = next(rows)
    print(first_row)
```

完整但仍然很简单的版本见 [`inspect_clinvar.py`](inspect_clinvar.py)。

## 文件格式

| 属性 | 说明 |
|---|---|
| 压缩 | gzip；文件名以 `.txt.gz` 结尾 |
| 文本格式 | UTF-8 |
| 表格格式 | TSV，即字段由 Tab `\t` 分隔 |
| 表头 | 第一行，共 43 列；第一列叫 `#AlleleID` |
| 缺失占位 | 常见为 `-`、`na` 或空字符串，不能一律当成数字 0 |
| 多值字段 | 常见使用 `|`、`;` 或 `,` 分隔，具体含义取决于字段 |
| 坐标体系 | 同时包含右对齐的 Start/Stop 表示和左对齐的 VCF 表示 |

NCBI 每周生成当前版，并在每月第一次发布附近保存一份月归档。官方说明见
[ClinVar tab-delimited README](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/README)。

## 最重要的概念：一行不等于一个唯一变异

`variant_summary` 的一行可以理解为：

> 一个简单等位基因（AlleleID）在一个参考基因组组装（Assembly）上的位置与汇总解释。

同一个 AlleleID 通常会有 GRCh37 和 GRCh38 两行。因此：

- `wc -l` 或 DataFrame 行数不是唯一变异数；
- 若想数 ClinVar 标识，通常统计唯一 `VariationID` 或 `#AlleleID`；
- 若想数基因组上的具体变异，通常统计唯一
  `(Assembly, Chromosome, PositionVCF, ReferenceAlleleVCF, AlternateAlleleVCF)`；
- VEP-eval 固定使用 GRCh38，并以 `CHROM, POS, REF, ALT` 作为 variant key。

## 四种常见 ClinVar 标识不要混淆

| 标识 | 含义 |
|---|---|
| AlleleID | ClinVar 为一个简单 allele 分配的整数标识；本文件主要按 AlleleID 组织 |
| VariationID | ClinVar variation record 的标识；一个复杂 record 可以涉及多个 allele，本文件里的映射不保证完整 |
| RCV | 某个 variation-condition 聚合记录的 accession，例如 `RCV000000012` |
| SCV | 单个提交者提交记录的 accession，例如 `SCV001451119`；多个 SCV 聚合成 ClinVar 的分类 |

如果需要完整的 AlleleID 与 VariationID 关系，应使用 NCBI 的
`variation_allele.txt.gz`，而不是假设本表是一对一映射。

## 43 个字段逐列解释

下面的解释以 NCBI 官方 README 为主，并结合当前文件表头说明。

### 变异、基因和 germline 分类

| # | 字段 | 直观解释 |
|---:|---|---|
| 1 | `#AlleleID` | ClinVar 简单 allele 的整数 ID。开头的 `#` 只是表头命名形式 |
| 2 | `Type` | 变异的大类，如 `single nucleotide variant`、`Deletion`、`Indel`、`Duplication` |
| 3 | `Name` | ClinVar 选择的首选名称，常含 RefSeq HGVS，例如 `NM_...:c... (p....)` |
| 4 | `GeneID` | NCBI Gene ID；只有一个相关基因时给整数，否则常为 `-1` |
| 5 | `GeneSymbol` | 相关基因符号，如 `BRCA1`；多基因时可能包含多个符号 |
| 6 | `HGNC_ID` | HGNC 标识，格式如 `HGNC:1100`；无法给单一基因时为 `-` |
| 7 | `ClinicalSignificance` | ClinVar 汇总得到的 germline 临床分类；可能是 Benign、Pathogenic、VUS 或冲突分类等 |
| 8 | `ClinSigSimple` | 简化标记：`1` 表示至少有一个当前提交把它归为致病/可能致病或风险等类别；`0` 表示没有；`-1` 表示完全没有临床意义值。它不等价于第 7 列 |
| 9 | `LastEvaluated` | 参与 germline 汇总分类的提交中，最近一次“最后评估日期” |

`ClinSigSimple=1` 不代表最终聚合结果一定是 Pathogenic。它不考虑提交是否提供了
判定标准和证据，而 `ClinicalSignificance` 的聚合逻辑会优先考虑有证据的提交。
做 benchmark 时通常应使用 `ClinicalSignificance` 和 `ReviewStatus`。

### 外部标识与疾病/表型

| # | 字段 | 直观解释 |
|---:|---|---|
| 10 | `RS# (dbSNP)` | dbSNP rs 数字，不带 `rs` 前缀；缺失时通常是 `-1` |
| 11 | `nsv/esv (dbVar)` | dbVar 的结构变异标识，如 nsv/esv；无值时为 `-` |
| 12 | `RCVaccession` | 与该 allele 相关的 RCV accession；多个值用 `|` 分隔 |
| 13 | `PhenotypeIDS` | 疾病/表型的数据库 ID，如 MedGen、MONDO、OMIM、Orphanet；分隔层次见下文 |
| 14 | `PhenotypeList` | 与 `PhenotypeIDS` 对应的疾病/表型名称 |

`PhenotypeIDS` 的分隔并不是一个简单列表：

- 不同 RCV 的内容由 `|` 分隔；
- 同一疾病在多个数据库中的 ID 由 `,` 分隔；
- 一个 RCV 涉及多个疾病时由 `;` 分隔；
- 条件过多时，NCBI 可能只报告条件数量。

### 来源、参考组装和坐标

| # | 字段 | 直观解释 |
|---:|---|---|
| 15 | `Origin` | 所有提交中报告的 allele origin 明细，如 germline、somatic、de novo、maternal |
| 16 | `OriginSimple` | NCBI 从 Origin 简化得到的类别，如 `germline`、`somatic`、`germline/somatic` |
| 17 | `Assembly` | 坐标所在参考组装，如 `GRCh37`、`GRCh38`；无组装时可能为 `na` |
| 18 | `ChromosomeAccession` | 定义坐标的 RefSeq accession 及版本，如 `NC_000007.14` |
| 19 | `Chromosome` | 染色体，如 `1`–`22`、`X`、`Y`；也可能是其他序列 |
| 20 | `Start` | NCBI 右对齐表示的起点，方向为 pter 到 qter |
| 21 | `Stop` | NCBI 右对齐表示的终点 |
| 22 | `ReferenceAllele` | 与 Start/Stop 对应的右对齐参考 allele |
| 23 | `AlternateAllele` | 与 Start/Stop 对应的右对齐替代 allele |
| 24 | `Cytogenetic` | 细胞遗传学带区，如 `7p22.1` |

### 证据等级、提交者和 ClinVar record

| # | 字段 | 直观解释 |
|---:|---|---|
| 25 | `ReviewStatus` | germline 聚合分类的审核状态，也是 ClinVar 星级的依据 |
| 26 | `NumberSubmitters` | 描述该变异的提交者数量 |
| 27 | `Guidelines` | 与该基因相关的 ACMG incidental/secondary finding guideline 标记；通常为 `-` |
| 28 | `TestedInGTR` | NIH Genetic Testing Registry 是否有针对该变异登记的检测，`Y`/`N` |
| 29 | `OtherIDs` | 其他数据库或知识库中的标识，多值组合字段 |
| 30 | `SubmitterCategories` | 提交来源编码：`1` 仅其他资源、`2` 其他类型来源、`3` 两者都有、`4` 无 |
| 31 | `VariationID` | ClinVar variation record ID；这里不一定列出与 AlleleID 相关的全部 VariationID |

常见 ReviewStatus 与网页星级大致对应：

- `practice guideline`：4 星；
- `reviewed by expert panel`：3 星；
- `criteria provided, multiple submitters, no conflicts`：2 星；
- `criteria provided, single submitter` 或存在冲突的有标准提交：1 星；
- `no assertion criteria provided` / `no assertion provided`：0 星。

准确规则以 NCBI 的
[review status 文档](https://www.ncbi.nlm.nih.gov/clinvar/docs/review_status/)
为准。

### VCF 规范化坐标

| # | 字段 | 直观解释 |
|---:|---|---|
| 32 | `PositionVCF` | VCF 风格的左对齐起点 |
| 33 | `ReferenceAlleleVCF` | 与 PositionVCF 对应的 VCF REF |
| 34 | `AlternateAlleleVCF` | 与 PositionVCF 对应的 VCF ALT |

做 VEP 或建立 genomic variant key 时，优先使用这三个 VCF 字段，加上
`Assembly` 和 `Chromosome`。不要把 `Start/ReferenceAllele` 与
`PositionVCF/ReferenceAlleleVCF` 混搭；indel 的左右对齐表示可能不同。

### Somatic clinical impact 与 oncogenicity

| # | 字段 | 直观解释 |
|---:|---|---|
| 35 | `SomaticClinicalImpact` | somatic clinical impact 的聚合分类 |
| 36 | `SomaticClinicalImpactLastEvaluated` | 上述 somatic impact 提交中的最近评估日期 |
| 37 | `ReviewStatusClinicalImpact` | somatic clinical impact 聚合分类的审核状态 |
| 38 | `Oncogenicity` | oncogenicity 的聚合分类 |
| 39 | `OncogenicityLastEvaluated` | oncogenicity 提交中的最近评估日期 |
| 40 | `ReviewStatusOncogenicity` | oncogenicity 聚合分类的审核状态 |
| 41 | `SCVsForAggregateGermlineClassification` | 参与 germline 聚合分类的 SCV accession 列表；通常由 `|` 分隔 |
| 42 | `SCVsForAggregateSomaticClinicalImpact` | 参与 somatic clinical impact 聚合的 SCV 列表 |
| 43 | `SCVsForAggregateOncogenicityClassification` | 参与 oncogenicity 聚合分类的 SCV 列表 |

对于纯 germline benchmark，第 35–40、42–43 列通常大面积为 `-`，这是正常的，
不是文件损坏。

## VEP-eval 实际使用哪些字段

参考 notebook 的 ClinVar 前处理主要使用：

```text
Assembly
OriginSimple
Chromosome
Type
ClinicalSignificance
ReviewStatus
Name
VariationID
PositionVCF
ReferenceAlleleVCF
AlternateAlleleVCF
```

筛选顺序是 GRCh38 → 排除 exact somatic → 1–22/X/Y → SNV → 指定良恶性标签 →
至少 1 星 → 单碱基 A/C/G/T → hg38 REF 校验，然后再结合 MANE/HGVS 生成功能类别。

## 常见陷阱

1. 不要把总行数当作变异数；同一 allele 会按 assembly 重复。
2. 不要用 `ClinSigSimple` 直接代替 `ClinicalSignificance`。
3. `OriginSimple != somatic` 会保留 `germline/somatic`；是否保留必须明确决定。
4. `REF` 和 `ALT` 都是 A/C/G/T 不等于 `REF != ALT`；当前文件里存在 A>A 等记录。
5. `ClinicalSignificance` 是聚合结果，不是某个提交者的原始结论；单个提交见 SCV。
6. 月归档与当前周更新版可能发生撤回和重分类，版本之间不是简单追加。
7. 多值字段的分隔符有层级，不要对所有字段统一 `split('|')`。

## 本目录中的文件

| 文件 | 说明 |
|---|---|
| `data/raw/variant_summary_2025-03.txt.gz` | 2025-03 月归档；作为 2025-03-23 当时可用版本 |
| `data/raw/variant_summary_2026-02.txt.gz` | VEP-eval 固定使用的归档 |
| `data/raw/variant_summary_latest_2026-09-14.txt.gz` | 下载时最新版，服务器更新时间 2026-09-14 |
| `inspect_clinvar.py` | 本 README 对应的简单探查脚本 |
| `analyze_clinvar_versions.py` | 三版本完整分块统计与比较脚本 |
| `REPORT.md` | 三版本统计结果与 VEP-eval 数据准备审查 |

输入文件的大小和 SHA-256 见 `results/input_manifest.json`。

## 按 VEP-eval 生成最终功能任务

已经严格按参考 notebook 处理三个版本，并生成 missense、synonymous、stop gain、
stop loss、splice、5′UTR、3′UTR、intron、RNA gene 等最终组。

完整的字段来源、35 个布尔条件、14 个主要评测任务和三版本统计见：

- [`VEP_EVAL_GROUPS.md`](VEP_EVAL_GROUPS.md)
- 实现代码：[`build_vep_eval_groups.py`](build_vep_eval_groups.py)
- 输出目录：[`processed/vep_eval/`](processed/vep_eval/)

注意：任务表中的格式 `总数 (B/P)` 表示 `总数 (Benign / Pathogenic)`，而且
这里的总数是最终主 benchmark 的 exact B/P 数量；它不等于该功能组的 all-label
数量（all-label 还包括 Likely benign 和 Likely pathogenic）。例如最新版
`missense` 的 `42,383 (26,937 / 15,446)` 表示 26,937 个 Benign 加 15,446
个 Pathogenic。最新版 all-label missense 实际为 195,362，之所以 exact B/P
比 2025-03 的 43,504 略少，是因为 likely 标签占比上升，而主 benchmark 排除了
likely 标签；详细对比见 [`VEP_EVAL_GROUPS.md`](VEP_EVAL_GROUPS.md) 的说明。

复跑命令：

```bash
python build_vep_eval_groups.py \
  --release 2025-03=data/raw/variant_summary_2025-03.txt.gz \
  --release 2026-02=data/raw/variant_summary_2026-02.txt.gz \
  --release 2026-09-14=data/raw/variant_summary_latest_2026-09-14.txt.gz \
  --mane data/raw/MANE.GRCh38.v1.5.refseq_genomic.gff.gz \
  --genome data/raw/hg38.fa \
  --refseq data/raw/ncbiRefSeq.txt.gz \
  --output processed/vep_eval
```

## 官方资料

- [ClinVar tab-delimited README](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/README)
- [ClinVar review status](https://www.ncbi.nlm.nih.gov/clinvar/docs/review_status/)
- [ClinVar 当前 tab-delimited 文件目录](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/)
