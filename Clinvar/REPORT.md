# ClinVar 版本对比与 VEP-eval 数据准备审查

分析日期：2026-09-20（Asia/Shanghai）

## 1. 数据与复现范围

- 参考 benchmark：`/data/yitian_workspace/DNA/benchmarks/VEP-eval`
- 参考 notebook：`VEP_ClinVar_Benchmarking_RefSeq.ipynb`
- 当前目录的 notebook 副本与参考文件 SHA-256 完全一致：
  `ba83644d335d1afbc41f55a13651cddc6e5980c86cbe3c2da6807dbfb4d1b60f`
- benchmark 固定输入：NCBI ClinVar `variant_summary_2026-02.txt.gz`
- 用户指定的 `2025-03-23` 没有对应的日级 ClinVar 快照；NCBI 可用的是
  `variant_summary_2025-03.txt.gz` 月归档，服务器 `Last-Modified` 为
  2025-03-07 14:13:45 GMT。本报告将其视为 2025-03-23 当时可获得的归档版本。
- 本次“最新版”：NCBI `variant_summary.txt.gz`，服务器 `Last-Modified` 为
  2026-09-14 13:55:03 GMT；本地用日期化文件名保存，防止以后“latest”漂移。
- 官方来源：
  [2025-03 archive](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/variant_summary_2025-03.txt.gz)、
  [2026-02 archive](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/variant_summary_2026-02.txt.gz)、
  [current variant_summary](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz)。
- 三份输入均通过 `gzip -t`。文件大小与 SHA-256 见
  `results/input_manifest.json`。

本次复刻 notebook 中从 ClinVar 原表到 MANE 注释之前/附近的筛选，并做版本
比较。没有执行完整 MANE 逐行注释及 65,537 bp hg38 序列窗口校验，因此下文的
`pre-MANE` 数量不是最终 `clinvar_benchmark.csv.gz` 的行数；完整 notebook 在参考
基因组校验后还可能删掉少量记录。

## 2. Notebook 如何准备 ClinVar benchmark

主流程如下：

1. 读取固定的 2026-02 `variant_summary`（43 列）。
2. 只保留 `Assembly == GRCh38`。
3. 排除 `OriginSimple == somatic`；注意 `germline/somatic` 仍然保留。
4. 只保留 1–22、X、Y。
5. 只保留 `Type == single nucleotide variant`。
6. 仅映射六种明确的良/恶性字符串：Benign、Likely benign、
   Benign/Likely benign、Pathogenic、Likely pathogenic、
   Pathogenic/Likely pathogenic；VUS、conflicting 等均排除。
7. 将 review status 映射为 1–4 星，保留至少 1 星。
8. 用 MANE Select GRCh38 v1.5 GFF 逐坐标标注 CDS、UTR、intron、splice、RNA 等；
   同时从 ClinVar `Name` 的 HGVS 字符串解析蛋白变化。
9. 要求 VCF REF/ALT 都是单个 A/C/G/T。
10. 用 UCSC hg38 FASTA 检查中心碱基是否等于 REF，并要求上下游各 32,768 bp
    的窗口完整。
11. 以 `#CHROM, POS, REF, ALT` 检查重复，构建 missense、synonymous、stop、
    splice、UTR、intron、RNA gene 等功能组。
12. `clinvar_benchmark_all_labels.csv.gz` 保留 exact 与 likely 标签；主 benchmark
    `clinvar_benchmark.csv.gz` 最后只保留标签 0（Benign）和 1（Pathogenic），
    likely 标签仅针对若干低样本功能组另存补充集。

## 3. 三个版本各有多少 variant

不同“variant 数”口径差别很大，因此同时报告原始行、唯一 VariationID 和唯一
GRCh38 坐标等位基因键。

| 口径 | 2025-03 | 2026-02 | 2026-09-14 latest | 2025-03 → latest |
|---|---:|---:|---:|---:|
| 原始记录行 | 6,839,999 | 8,674,145 | 9,056,310 | +2,216,311 |
| 原始唯一 VariationID | 3,452,550 | 4,370,172 | 4,562,367 | +1,109,817 |
| GRCh38 记录行 | 3,388,341 | 4,303,534 | 4,494,430 | +1,106,089 |
| GRCh38 唯一 VariationID | 3,387,587 | 4,301,221 | 4,492,073 | +1,104,486 |
| notebook 筛选后、含 exact/likely 的 canonical SNV 行 | 1,284,123 | 1,346,039 | 1,437,333 | +153,210 |
| 上述候选集唯一 VariationID | 1,283,751 | 1,345,649 | 1,436,915 | +153,164 |
| 上述候选集唯一 CHROM:POS:REF:ALT | 1,284,123 | 1,346,039 | 1,437,333 | +153,210 |
| 最终 exact Benign/Pathogenic 的 pre-MANE 唯一键 | 240,356 | 242,137 | 245,571 | +5,215 |

2025-03 与 2026-02 的含 exact/likely 候选键共有 1,274,107 个，2025-03 独有
10,016 个，2026-02 独有 71,932 个；净增 61,916。exact B/P 键共有 233,299 个，
2025-03 独有 7,057 个，2026-02 独有 8,838 个；净增仅 1,781。

2025-03 与最新版的候选键共有 1,270,974 个，2025-03 独有 13,149 个，最新版
独有 166,359 个；净增 153,210。exact B/P 键共有 228,632 个，2025-03 独有
11,724 个，最新版独有 16,939 个；净增 5,215。这说明更新不是简单追加：已有
记录也会撤回、重分类或改变坐标/等位基因表示。

exact B/P 的 pre-MANE 标签分解：

| 标签 | 2025-03 | 2026-02 | 2026-09-14 latest | 2025-03 → latest |
|---|---:|---:|---:|---:|
| Benign | 174,039 | 172,622 | 173,369 | -670 |
| Pathogenic | 66,317 | 69,515 | 72,202 | +5,885 |
| 合计 | 240,356 | 242,137 | 245,571 | +5,215 |

## 4. 筛选漏斗

| 步骤 | 2025-03 | 2026-02 | 2026-09-14 latest |
|---|---:|---:|---:|
| GRCh38 | 3,388,341 | 4,303,534 | 4,494,430 |
| 排除 exact somatic | 3,382,551 | 4,296,847 | 4,487,698 |
| 常规染色体 | 3,379,593 | 4,293,730 | 4,484,565 |
| `single nucleotide variant` | 3,096,213 | 3,985,389 | 4,157,032 |
| 六类 selected clinical significance | 1,335,970 | 1,398,129 | 1,487,246 |
| 至少 1 星 | 1,284,133 | 1,346,049 | 1,437,343 |
| REF/ALT 均为 A/C/G/T | 1,284,123 | 1,346,039 | 1,437,333 |
| 严格 REF != ALT | 1,283,632 | 1,345,548 | 1,436,843 |

## 5. 原始 variant Type 分布

统计口径与 notebook 一致：GRCh38、排除 exact somatic、常规染色体之后，尚未
限制为 SNV。

| Type | 2025-03 | 2026-02 | 2026-09-14 latest |
|---|---:|---:|---:|
| single nucleotide variant | 3,096,213 | 3,985,389 | 4,157,032 |
| Deletion | 140,074 | 156,508 | 167,408 |
| Duplication | 63,141 | 67,014 | 71,084 |
| Microsatellite | 35,724 | 37,952 | 39,135 |
| Indel | 16,293 | 17,772 | 19,279 |
| Insertion | 12,708 | 13,496 | 14,123 |
| copy number gain | 6,939 | 6,945 | 7,366 |
| copy number loss | 6,738 | 6,744 | 7,136 |
| Inversion | 1,387 | 1,441 | 1,524 |
| Variation | 354 | 442 | 443 |
| Translocation | 18 | 18 | 18 |
| Complex | 4 | 9 | 17 |

SNV 占这一阶段记录的 91.62%（2025-03）、92.82%（2026-02）和 92.70%（最新版）。需要注意，论文
中所谓 missense、splice、UTR 等“variant type”是后续由 MANE/HGVS 得到的功能
类别，不等同于此处 ClinVar `Type` 字段。

## 6. 标签、星级和碱基替换概况

至少 1 星后，2025-03 的 1,284,133 条记录中，Likely benign 占 71.14%，Benign
13.55%，Pathogenic 5.16%，Likely pathogenic 4.74%。2026-02 的 1,346,049 条中，Likely benign 占 70.96%，Benign
12.82%，Pathogenic 5.16%，Likely pathogenic 4.87%；最新版的 1,437,343 条中，
Likely benign 占 71.49%，Benign 12.06%，Pathogenic 5.02%，Likely pathogenic
5.06%。因此含 likely 的候选集增长很多，但从 2026-02 到最新版，主 benchmark
所用 exact B/P 只净增 3,434 个唯一键。

canonical 候选集中最常见的替换是 G>A 与 C>T：

| 替换 | 2025-03 | 2026-02 | 2026-09-14 latest |
|---|---:|---:|---:|
| G>A | 304,628 | 318,891 | 338,691 |
| C>T | 304,344 | 318,643 | 338,183 |
| A>G | 147,369 | 155,175 | 165,576 |
| T>C | 146,008 | 153,777 | 164,024 |

完整的 assembly、origin、chromosome、clinical significance、review status、标签和
16 种 REF>ALT 组合均在 `results/distributions.csv`。

## 7. 审查中发现的注意点

1. notebook 的 canonical allele 条件没有要求 `REF != ALT`，因此会保留 491
   （2025-03 和 2026-02）和 490（最新版）条 A>A/C>C/G>G/T>T 记录。hg38 中心 REF 校验也
   不会排除它们。如目标确为 substitution benchmark，建议在未来版本显式加上
   `REF != ALT`，但本次没有修改参考代码。
2. `OriginSimple != 'somatic'` 只排除 exact somatic，`germline/somatic` 会保留。
   这可能符合作者意图，但应在方法中明确。
3. notebook 会报告重复的 `#CHROM,POS,REF,ALT`，但不会 `drop_duplicates`。
   本次三个版本的 pre-MANE canonical 候选键均无重复行，不过完整 MANE 流程或
   后续数据版本仍应把该检查当成硬断言。
4. promoter 匹配存在染色体前缀不一致：MANE GFF 和 `MANE` 表使用 `chr1`，但
   promoter 查询比较的是不带 `chr` 的 ClinVar 值（例如 `1`），因此 promoter
   命中会是空的。当前最终分组并未实际使用 promoter 列，所以主结果可能不受
   影响，但 promoter 注释本身不可信。
5. 当前仓库里的 notebook 已清空执行输出，无法直接核对 2026-02 完整 MANE/hg38
   流程的最终行数。仓库历史里的旧输出来自更早数据/逻辑，不应当作 2026-02
   的结果。
6. MANE 标注实现对每条 ClinVar 记录反复扫描 GFF DataFrame，复杂度和运行时间
   都较高；若要完整重跑最新版，最好先做染色体区间索引，同时保持同样语义。
7. 最终 benchmark 只使用 exact Benign/Pathogenic，而 likely 标签在主文件中被
   排除。这会让“前处理候选数”和“最终 benchmark 数”相差很大，不应混用。

## 8. 产物

- `analyze_clinvar_versions.py`：可复跑的分块统计脚本。
- `results/filter_funnel.csv`：每一步筛选计数。
- `results/version_comparison.csv/json`：版本共有、独有、净变化。
- `results/distributions.csv`：各种分类完整分布。
- `results/sample_*.tsv`：三版通过 canonical 筛选后的 10 行示例。
- `results/input_manifest.json`：输入文件大小与 SHA-256。

参考 `VEP-eval` 仓库未被修改；其 `git status --short` 在分析结束时为空。
