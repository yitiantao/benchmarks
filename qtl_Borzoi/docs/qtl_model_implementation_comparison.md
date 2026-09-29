# QTL benchmark 各模型实现细节与源码审计

本文集中记录 Borzoi、AlphaGenome、NTv3 和 DNA-FM 在当前仓库中的实际实现，以及与上游源码、
教程和论文流程的关系。最后核对日期为 **2026-09-24**。

> “使用同一 benchmark”只表示输入数据、标签和最终 evaluator 一致，不表示各模型的原始输出头、
> track 粒度、score 公式或原生论文评价流程完全相同。

## 1. 结论速览

| 任务 | Borzoi | AlphaGenome | NTv3 post | DNA-FM post |
|---|---|---|---|---|
| eQTL | GTEx RNA targets；gene bins 上 SED | `RNA_SEQ`；gene exon mask LFC | RNA BigWig tracks | RNA functional tracks |
| sQTL | `nDi/span`；归一化 RNA profile 形状变化 | `SPLICE_JUNCTIONS`；junction log-count 变化 | `splice_donor`、`splice_acceptor` | `splice_donor`、`splice_acceptor` |
| paQTL | `COVR/utr3`；3' most exon PAS ratio | `POLYADENYLATION`；底层是 `RNA_SEQ` + PAS | `polyA_signal` | `polyA_signal` |
| iPaQTL | `COVR`；intronic + 3' UTR PAS | 当前复用 paQTL scorer | `polyA_signal`、`3UTR+`、`3UTR-` | `polyA_signal`、`3UTR_plus`、`3UTR_minus` |

关键结论：

- AlphaGenome 并非所有任务都直接使用普通 RNA-seq 输出：eQTL 和 paQTL/iPaQTL 使用
  `RNA_SEQ`，sQTL 使用独立的 `SPLICE_JUNCTIONS` 输出。
- NTv3 官方教程没有 QTL benchmark 或专用 QTL scorer。当前 NTv3 路线是基于官方输出 heads 的
  零样本迁移，不是官方 NTv3 QTL 评价的复现。
- 当前只有 eQTL 明确按 QTL tissue 选择模型输出。matched sQTL/paQTL/iPaQTL 输入没有 tissue
  标签；AlphaGenome 会跨输出 tissue 取最大绝对 score，NTv3/DNA-FM 则使用通用 annotation heads。
- Borzoi sQTL 的 track 选择和 evaluator 已单独隔离，只影响 Borzoi sQTL。

## 2. 共享数据与评价层

| 项目 | 当前设置 |
|---|---|
| eQTL 输入 | `data/eqtl` 下 49 个 GTEx tissues 的 positive/negative VCF 和 SuSiE causal tables |
| eQTL evaluator | `GTExEQTLDataModule` + `GTExEQTLEvaluator` |
| sQTL/paQTL/iPaQTL 输入 | `MatchedQTLDataModule`；VCF `MT` 是目标 gene，`PI` 恢复正负配对 |
| matched evaluator | seed 44，100 次、每次 80% 平衡重采样 |
| 统一预测表 | `variant_key, gene_id, tissue, score, n_tracks` |
| allele 方向 | 模型层生成 VCF `ALT-REF`；eQTL evaluator 再按 effect allele 必要时翻转 |

统一接口位于 [`interfaces.py`](../benchmark/qtl_benchmark/interfaces.py)。模型只负责把原生输出转成
统一长表，缓存和 evaluator 不导入具体模型运行时。

### 2.1 eQTL 指标

[`GTExEQTLEvaluator`](../benchmark/qtl_benchmark/evaluation/gtex_eqtl.py) 对每个 tissue 单独计算：

- `auroc_sign`：causal variant-gene 的 effect direction；
- `spearmanr` / `pearsonr`：causal effect size 与模型 score；
- `auroc_class`：每个 variant 在该 tissue 内跨 gene 取最大 `abs(score)` 后区分 pos/neg。

### 2.2 matched QTL 指标

[`MatchedQTLEvaluator`](../benchmark/qtl_benchmark/evaluation/matched_qtl.py) 的流程是：

1. 优先按 `(variant_key, MT gene)` 找预测；找不到时默认允许 variant-level fallback；
2. 若目标 gene 有多个 tissue 结果，取 `abs(score)` 最大者并保留原符号；
3. 按 `SD`（sQTL）或 `PD`（paQTL/iPaQTL）应用累积距离 cutoff；
4. 根据 negative 的 `PI` 重建一对一 positive/negative pair；
5. 每次从两类各抽取 `floor(0.8*m)`，以 `abs(score)` 计算 AUROC 和 AUPRC；
6. 重复 100 次，输出均值、标准差和样本数。

sQTL cutoff 为 10,000、2,000、500、200、100、50 bp；paQTL/iPaQTL 另含 300 bp。
`max_distance` 是 variant 到相应 splice/PAS event 的距离，不是正负 variant 之间的距离。

典型输出 schema 保持为：

```text
task  max_distance  matched_positive  matched_negative  scored_positive
scored_negative  sample_per_class  auroc_mean  auroc_std  auprc_mean  auprc_std
```

修正模型 score 或 track 选择后，这些列仍会正常输出。覆盖相同时 matched/scored 样本数通常不变，
但 AUROC/AUPRC 可能改变；“输出格式不变”不等于“指标数值不变”。

## 3. Borzoi

### 3.1 与上游代码的一致性边界

当前推理脚本来自 Borzoi/Borzoi-paper 路线，并保留核心任务参数：

| 任务 | 当前/上游关键参数 |
|---|---|
| eQTL | `SED,logSED`，gene-aware SED，GTEx targets |
| sQTL | `--span --no_untransform --stats nDi`，RNA targets |
| paQTL | `--stats COVR --utr3`，GTEx RNA targets |
| iPaQTL | `--stats COVR`，不启用 `--utr3` |

任务入口见上游 [`run_qtl_benchmark.sh`](../../../borzoi-paper/run_qtl_benchmark.sh) 和本地
[`borzoi_sed.py`](../src/borzoi_scripts/borzoi_sed.py)。本地增加的 multi-GPU shard、variant batch、
断点续跑和统一 evaluator 不改变 `nDi/COVR/SED` 的底层定义。

需要区分两种复现：四个 published replicate + RC ensemble + 上游原生 `stats.txt` 最接近论文；
`scripts/all_borzoi_qtl.sh` 当前默认只跑 `replicate_0`，不能称为四模型 ensemble。

### 3.2 `-d 0`、D1/D2 与 `nDi`

上游命令中的 `-d 0` 不是名为 “D0” 的统计量。它在
[`borzoi_bench_sqtl_folds.py`](../src/borzoi_scripts/borzoi_bench_sqtl_folds.py) 中表示
`data_head=0`，即模型的第 0 个 dataset/head。

相关统计量：

- `D1 = sum(abs(REF - ALT))`；
- `D2 = sqrt(sum((REF - ALT)^2))`；
- `nD2`：分别归一化 REF/ALT profile 后计算 L2；
- `nDi`：分别归一化 REF/ALT profile 后取最大逐位置绝对差。

sQTL 正式使用：

```text
REF_norm = (REF + pseudocount) / sum(REF + pseudocount)
ALT_norm = (ALT + pseudocount) / sum(ALT + pseudocount)
nDi      = max_position(abs(REF_norm - ALT_norm))
```

所以 `nDi` 衡量预测 RNA/splicing profile 的形状变化，不是总表达量差，也不是 AlphaGenome 的
junction pair-count statistic。

### 3.3 eQTL score 与 tissue track

```text
SED = sum(ALT) - sum(REF)
shared score = arcsinh(mean(selected GTEx-track SED))
```

[`BorzoiSEDModelAdapter`](../benchmark/qtl_benchmark/models/borzoi.py) 用 `TISSUE_KEYWORDS` 将 49 个
GTEx tissues 映射到 `brain`、`blood_vessel`、`skin` 等 broad tissue。多个精细 brain tissue 因而
共享同一组 Borzoi tracks，而不是各自拥有精确 track。

### 3.4 sQTL 的 Borzoi-only 修正

论文 sQTL 不是对 `targets_rna.txt` 中所有 RNA targets 求均值，而是读取输出的最后 **89 个 GTEx
RNA tracks**。当前 [`BorzoiQTLModelAdapter`](../benchmark/qtl_benchmark/models/borzoi.py) 通过
`sqtl_gtex_track_count=89` 独立实现，然后跨这些 tracks 求均值。

sQTL 还使用独立的
[`BorzoiSQTLEvaluator`](../benchmark/qtl_benchmark/evaluation/borzoi_sqtl.py)。它与公共 matched
evaluator 保持相同字段、cutoff、seed、重采样和 AUROC/AUPRC 契约，但代码与其他模型隔离。

### 3.5 paQTL 和 iPaQTL

- paQTL 使用 `COVR/utr3`，将 PolyADB 限制到 `3' most exon` 的连续 PAS；
- iPaQTL 使用独立推理入口和 `COVR`，保留 intronic 与 3' UTR PAS，不启用 `--utr3`；
- 两者都从 PAS 周围 coverage 的 ALT/REF ratio 构造 proximal/distal change；
- 默认 `cov_pseudo=50`、`cov_min=100`。

### 3.6 同名 channel / 同 biosample 多 experiment

Borzoi 每列由 target index/ID 唯一标识。即使多列有相同 tissue 或相似 display label，只要
accession/index 不同，仍是不同实验 channel。当前不会“同名只留第一条”，而是：

1. 按 GTEx ID、tissue keyword 或任务规定的列区间选择所有 channel；
2. replicate 文件先逐元素平均；
3. 再对所选 track 维度求均值，得到统一 scalar score；
4. `n_tracks` 记录参与聚合的列数。

因此，同一 biosample 的不同 experiment 会作为独立证据进入聚合，不会被名称覆盖。这与上游将
targets 保留为独立输出列一致；但公共 scalar evaluator 不等于论文原生把多维 targets 输入随机
森林的 classification evaluator。

## 4. AlphaGenome

### 4.1 官方 scorer 与当前映射

当前 [`AlphaGenomeModel`](../benchmark/qtl_benchmark/models/alphagenome.py) 调用安装包中的
`RECOMMENDED_VARIANT_SCORERS`：

```python
DEFAULT_TASK_SCORERS = {
    "eqtl": "RNA_SEQ",
    "sqtl": "SPLICE_JUNCTIONS",
    "paqtl": "POLYADENYLATION",
    "ipaqtl": "POLYADENYLATION",
}
```

REF/ALT 推理后，用 `tidy_scores(..., match_gene_strand=True)` 展开 gene/track 长表。错误链 channel
被去除，无链 channel 可以保留。

### 4.2 eQTL：RNA_SEQ gene-mask LFC

官方 `GeneMaskLFCScorer` 对每个 gene exon mask 和 RNA-seq track 计算：

```text
score = log(mean(ALT) + 1e-3) - log(mean(REF) + 1e-3)
```

adapter 对相同 `(variant, gene, gtex_tissue)` 的 tracks 求均值；eQTL evaluator 再精确选择当前
benchmark tissue。AlphaGenome 的 tissue 粒度因此细于 Borzoi/NTv3/DNA-FM 的 broad map。
具体公式见 AlphaGenome research 的
[`gene_mask.py`](../../../alphagenome_research/src/alphagenome_research/model/variant_scoring/gene_mask.py)。

### 4.3 sQTL：SPLICE_JUNCTIONS

官方 `SpliceJunctionScorer` 不使用普通 RNA-seq coverage：

```text
delta = log(ALT_junction_count + 1e-7)
      - log(REF_junction_count + 1e-7)
score = abs(delta)
```

它只保留与目标 gene 同链且落在 gene 内的 junction，并为每个 gene/track 选择最大变化 junction。
它也不等于 NTv3 的 donor/acceptor BED annotation head。
实现见
[`splice_junction.py`](../../../alphagenome_research/src/alphagenome_research/model/variant_scoring/splice_junction.py)。

### 4.4 paQTL：RNA_SEQ + PAS annotation

`PolyadenylationScorer` 的 `requested_output` 实际是 `RNA_SEQ`：

1. 在每个 PAS 上游 400 bp 聚合 REF/ALT RNA coverage；
2. 形成每个 PAS 的 ALT/REF ratio；
3. 枚举 proximal/distal PAS split；
4. 取最大的绝对 log2 ratio change。

AlphaGenome research 源码明确将其标为 “Implements the Borzoi statistic”。统计思想继承 Borzoi，
但 PAS mask、site 过滤、pseudocount 和 tracks 不完全相同，不能称为逐式 exactly equal。
实现见
[`polyadenylation.py`](../../../alphagenome_research/src/alphagenome_research/model/variant_scoring/polyadenylation.py)。

### 4.5 iPaQTL 限制

AlphaGenome API 和 `alphagenome_research` 中没有独立 iPaQTL scorer。当前 iPaQTL 复用 paQTL 的
`POLYADENYLATION` scorer，没有 intron-only mask 或独立 intronic PAS 公式。它是零样本复用，
不是 Borzoi iPaQTL 方法的 exactly equal 实现。

### 4.6 track、biosample 和 tissue 聚合

`tidy_scores` 保留 `track_name`、`gtex_tissue`、`ontology_curie`、`biosample_name` 和 strand。
当前处理为：

1. 可选用 `gtex_tissues` / `ontology_terms` 过滤；正式配置目前未设置；
2. 按目标 `gene_id` 过滤；
3. 按 `(gene_id, gtex_tissue)` 对不同 experiment/track 求均值；
4. 保存 `n_tracks`，不按 display name 覆盖。

eQTL 会精确匹配 tissue。sQTL/paQTL/iPaQTL 的数据层把 tissue 设为空，因而 matched evaluator 会在
该 gene 的所有 tissue 结果中取绝对值最大者。这是本项目聚合策略，不是 AlphaGenome 官方规定的
组织选择规则。

### 4.7 是否提供完整 evaluation

`alphagenome_research` 提供 variant scorer、mask、track metadata 和整理代码，但没有与本仓库输入
完全相同的 Borzoi matched-QTL 全流程 evaluator（`PI` 配对、distance cutoff、100 次 80% 重采样）。
因此官方源码定义 model-side score，本仓库定义 benchmark-side metrics。

## 5. NTv3

### 5.1 官方教程审计

[`ntv3_tutorial`](../ntv3_tutorial) 包含 quickstart、track prediction、BigWig/annotation
fine-tuning、interpretation、generative training 和 enhancer generation。未发现 eQTL、sQTL、
paQTL、iPaQTL、matched-QTL 或专用 variant scorer 教程。

### 5.2 channel 总数

100M post 和 650M post 的 channel 名称与顺序一致：

| 类型 | 数量 |
|---|---:|
| human BigWig functional tracks | 7,362 |
| BED annotation elements | 21 |
| semantic channels | 7,383 |
| raw logits / position | 7,404 |

每个 BED element 有 negative/positive 两个 logits，所以 raw logits 比 semantic channels 多 21。
完整清单见 [`ntv3_channels.txt`](../ntv3_channels.txt)。

### 5.3 post-trained QTL 实现

同一 SNV 构造 REF/ALT 两条序列并计算 `ALT - REF`。BED heads 先 softmax，再取 positive-class
probability；BigWig tracks 直接比较输出值。

| task | head | 聚合 |
|---|---|---|
| eQTL | GTEx/ENCODE RNA BigWig tracks | gene span × tissue tracks × positions 上 signed max-abs |
| sQTL | `splice_donor`, `splice_acceptor` | positions × heads 上 signed max-abs |
| paQTL | `polyA_signal` | positions × head 上 signed max-abs |
| iPaQTL | `polyA_signal`, `3UTR+`, `3UTR-` | positions × heads 上 signed max-abs |

sQTL/paQTL/iPaQTL heads 是通用 genomic annotations，不是按实验 tissue 选择的 QTL scorer，因此
不做 biosample tissue matching；也没有使用目标 junction/PAS 的指定位置，而是在有效输出区间
取最大扰动。

### 5.4 eQTL RNA track 选择

eQTL 从 7,362 个 BigWig IDs 中解析 RNA metadata，并复用 Borzoi broad tissue map：

- 49 个 benchmark tissues 中，47 个使用 GTEx RNA tracks；
- checkpoint 缺少 3 条 GTEx heart tracks，两个 Heart tissues 回退到 32 条 ENCODE heart tracks；
- 去重后选择 107 条 RNA tracks；
- 每个 tissue 在 gene span、对应 tracks 和 positions 上取 signed max-abs，而非简单平均；
- track 来源和数量写入 `model_run.json`。

实现见 [`ntv3.py`](../benchmark/qtl_benchmark/models/ntv3.py)。K562/HepG2 两条示例 track 只保留为
未提供 metadata 时的自定义 fallback，不是正式 eQTL 路线。

### 5.5 同 biosample 多 experiment / 同名 channel

NTv3 channel 以 checkpoint channel ID 和索引为身份，不以显示名称为准。metadata 匹配后会对
同一索引去重，但不同 experiment accession 仍是不同 channel，并共同进入 tissue 的 signed
max-abs 聚合。

| biosample/assay | NTv3 channel ID |
|---|---|
| K562 total RNA-seq | `ENCSR056HPM` |
| HepG2 polyA-plus RNA-seq, plus strand | `ENCSR561FEE_P` |
| HepG2 polyA-plus RNA-seq, minus strand | `ENCSR561FEE_M` |

本地产物：

- [`ntv3_rna_track_mapping.xlsx`](../ntv3_rna_track_mapping.xlsx)：RNA track、biosample 和
  experiment 对应关系 Excel；
- [`ntv3_functional_tracks_metadata.csv`](../ntv3_functional_tracks_metadata.csv)：完整 metadata；
- [`ntv3_channels.txt`](../ntv3_channels.txt)：模型 channel 索引清单。

### 5.6 pre-trained fallback

pre 模型没有 functional heads：

```text
score = log P(ALT | masked sequence) - log P(REF | masked sequence)
```

它完全不使用 track，也不是 expression/splicing/polyA 方向，应标记为 sequence-likelihood baseline。

## 6. DNA-FM

DNA-FM 与 NTv3 post 使用相同类型的 heads 和 broad tissue map，但由本地 checkpoint/canonical
manifest 提供。已核对：

- human functional tracks 为 7,362 条，与 NTv3 的 `file_id` 数量、集合和顺序一致；
- eQTL 同样选择 107 条 RNA tracks，Heart 使用 ENCODE fallback；
- annotation logits 先 softmax，再计算 ALT-REF，并做 signed max-abs；
- DNA-FM adapter 可在前向前裁剪 heads；NTv3 通常先产生全部 7,362 个 BigWig heads，因此显存和
  速度不同。

详见 [`dna_fm_qtl.md`](dna_fm_qtl.md)。

## 7. “不需要实验 track”与组织选择

“不需要实验 track”只表示 evaluation 时不读取外部真实实验 BigWig。模型输出 channel 本身仍可能
对应 GTEx/ENCODE experiment。

| 任务/模型 | 当前是否组织匹配 | 实际行为 |
|---|---|---|
| AlphaGenome eQTL | 是，精确 GTEx tissue | 同 tissue tracks 求均值 |
| Borzoi eQTL | 是，broad tissue | 匹配所有 GTEx targets 后求均值 |
| NTv3/DNA-FM eQTL | 是，broad tissue | tissue tracks × gene positions 上 signed max-abs |
| AlphaGenome sQTL/paQTL/iPaQTL | 否 | 同 gene/tissue tracks 先均值，再跨 tissues 取 max-abs |
| Borzoi sQTL | 无单个 QTL tissue 条件 | 最后 89 个 GTEx tracks 求均值 |
| Borzoi paQTL/iPaQTL | 无单个 QTL tissue 条件 | 相应 RNA targets 聚合 COVR |
| NTv3/DNA-FM sQTL/paQTL/iPaQTL | 否 | 通用 annotation heads 跨位置/head 取 max-abs |
| NTv3 pre | 不适用 | masked-base likelihood，不使用 track |

如果以后要做真正 tissue-specific 的 matched QTL，数据层必须先提供每条 QTL 的可靠 source tissue，
再把它传给 adapter。当前 `MatchedQTLDataModule` 明确把 tissue 写为空字符串，不能由 evaluator
自行推断来源组织。

## 8. “exactly 一样”应分三层判断

1. **输入一致**：同一 VCF、`MT/PI/SD/PD`、limit 和正负集合；四模型可以做到。
2. **评价一致**：公共 evaluator 的 pairing、distance cutoff、重采样和 metrics 一致。
3. **模型 statistic 一致**：输出头、track、mask 和 score 公式逐项相同；四模型没有做到，也不应
   强行做到。

具体边界：

- Borzoi 本地 statistic 与上游高度一致，但单 replicate 与四-replicate ensemble 不同；
- AlphaGenome paQTL 借鉴 Borzoi statistic，但实现不完全同式；
- AlphaGenome、Borzoi、NTv3 的 sQTL 使用三种不同中间表征；
- AlphaGenome 没有独立 iPaQTL scorer；
- NTv3 官方没有 QTL tutorial/scorer，本项目是零样本任务映射；
- Borzoi 原生随机森林 metrics 与四模型公共 scalar-score metrics 不应混用。

跨模型比较应使用公共 `metrics.tsv/metrics.json`，不要直接比较 raw score 大小，也不要把论文原生
`stats.txt` AUROC 与公共 evaluator AUROC 当作等价结果。

## 9. 代码与文档索引

- Borzoi adapter：[`models/borzoi.py`](../benchmark/qtl_benchmark/models/borzoi.py)
- Borzoi-only sQTL evaluator：[`evaluation/borzoi_sqtl.py`](../benchmark/qtl_benchmark/evaluation/borzoi_sqtl.py)
- AlphaGenome adapter：[`models/alphagenome.py`](../benchmark/qtl_benchmark/models/alphagenome.py)
- NTv3 adapter：[`models/ntv3.py`](../benchmark/qtl_benchmark/models/ntv3.py)
- DNA-FM adapter：[`models/dna_fm.py`](../benchmark/qtl_benchmark/models/dna_fm.py)
- eQTL evaluator：[`evaluation/gtex_eqtl.py`](../benchmark/qtl_benchmark/evaluation/gtex_eqtl.py)
- matched evaluator：[`evaluation/matched_qtl.py`](../benchmark/qtl_benchmark/evaluation/matched_qtl.py)
- matched 数据加载：[`data/matched_qtl.py`](../benchmark/qtl_benchmark/data/matched_qtl.py)
- 执行、batch 与 replicate：[`implementation_details.md`](implementation_details.md)
- AlphaGenome eQTL：[`alphagenome_eqtl.md`](alphagenome_eqtl.md)
- Borzoi eQTL：[`borzoi_eqtl.md`](borzoi_eqtl.md)
- NTv3：[`ntv3_qtl.md`](ntv3_qtl.md)
- DNA-FM：[`dna_fm_qtl.md`](dna_fm_qtl.md)
