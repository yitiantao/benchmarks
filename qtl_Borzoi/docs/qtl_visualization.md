# 六套 QTL 结果可视化

一键脚本固定比较以下六套 `max_2000` 结果：

1. AlphaGenome all folds
2. Borzoi replicate 0
3. NTv3 100M post-training
4. DNA-FM 100M post-training step 150000（BF16）
5. DNA-FM 100M post-training step 340000（BF16 历史运行）
6. DNA-FM 100M post-training step 340000（最新 FP32 运行；目录名 `step_0034000`）

最新 step340000 FP32 输入来自 `outputs/dna_fm_step_0034000_fp32_smoke`；旧 BF16 输入来自
`outputs/dna_fm_step_0034000_smoke`。两者使用同一 checkpoint 和 evaluator，因而可以单独审计
精度设置带来的指标差异。

在仓库根目录运行，不需要参数：

```bash
./scripts/plot_six_result_qtl.sh
```

旧命令 `./scripts/plot_five_result_qtl.sh` 和 `./scripts/plot_four_model_qtl.sh` 仍是兼容别名。
结果写入 `outputs/six_result_qtl_plots/`：

| 文件 | 内容 |
|---|---|
| `all_qtl_overview.{png,pdf}` | 六套结果在 eQTL、sQTL、paQTL、iPaQTL 上的 AUROC 总览 |
| `eqtl_by_organ.{png,pdf}` | eQTL 按 broad organ group 的六套结果、四项指标柱状图 |
| `eqtl_by_tissue.{png,pdf}` | eQTL 按全部 49 个 GTEx tissue 的六套结果柱状图 |
| `matched_qtl_by_distance.{png,pdf}` | 六套结果在 sQTL、paQTL、iPaQTL 全部匹配距离下的 AUROC/AUPRC |
| `dna_fm_checkpoint_overview.{png,pdf}` | BF16 下新旧 DNA-FM checkpoint 四任务比较 |
| `dna_fm_checkpoint_eqtl_delta_by_{organ,tissue}.{png,pdf}` | 新旧 checkpoint 的 eQTL 配对差值 |
| `dna_fm_checkpoint_matched_qtl_delta.{png,pdf}` | 新旧 checkpoint 的 matched-QTL AUROC/AUPRC 配对差值 |
| `dna_fm_step340000_precision_overview.{png,pdf}` | 同一 step340000 的 FP32 与 BF16 四任务比较 |
| `dna_fm_step340000_precision_*_delta*.{png,pdf}` | 同一 checkpoint 下 `FP32 − BF16` 配对差值 |
| `ntv3_vs_dna_fm_step340000_overview.{png,pdf}` | NTv3 与新 DNA-FM checkpoint 的四任务直接比较 |
| `ntv3_vs_dna_fm_step340000_eqtl_by_{organ,tissue}.{png,pdf}` | NTv3 与新 checkpoint 的 eQTL 并排比较 |
| `ntv3_vs_dna_fm_step340000_matched_qtl_by_distance.{png,pdf}` | NTv3 与新 checkpoint 的 matched-QTL 并排比较 |
| `ntv3_vs_dna_fm_step340000_*_delta*.{png,pdf}` | 新 checkpoint 相对 NTv3 的配对差值 |
| `eqtl_by_tissue.tsv` | 49 tissue 绘图数据 |
| `eqtl_by_organ.tsv` | broad organ 汇总绘图数据 |
| `matched_qtl_by_distance.tsv` | 其余三类 QTL 绘图数据及误差 |
| `all_qtl_overview.tsv` | 总览图中六套结果、四个任务的精确 AUROC 数值 |
| `dna_fm_checkpoint_*.tsv` | BF16 下两个 DNA-FM checkpoint 的原值及差值 |
| `dna_fm_step340000_precision_*.tsv` | step340000 的 BF16、FP32 原值及 `FP32 − BF16` 差值 |
| `ntv3_vs_dna_fm_step340000_*.tsv` | NTv3、新 DNA-FM 原值和 `step340000 − NTv3` 差值 |
| `manifest.json` | 模型、输入文件和汇总口径 |

## 指标口径

总览图中 eQTL 使用 49 个 eligible tissue 的 causal classification AUROC 宏平均；sQTL、
paQTL 和 iPaQTL 使用最大匹配距离 10 kb 的 mean AUROC。虚线表示随机水平 0.5。
距离明细图不会只截取三个阈值，而是自动展示各任务 `metrics.tsv` 中存在的全部匹配距离。

eQTL organ 图使用 `TISSUE_KEYWORDS` 将 49 个 GTEx tissue 映射到 broad organ group，再对同一
group 内的 tissue 指标做不加权宏平均。这个聚合是为了可视化，不是重新合并 variant 后计算指标；
可审计的原始 tissue 指标保留在 `eqtl_by_tissue.tsv`。

AlphaGenome 使用精确 GTEx tissue tracks；Borzoi、NTv3 和 DNA-FM 使用相同的 broad-tissue
映射。柱状图比较的是同一数据和 evaluator 产生的指标，不表示四个模型的原始 score 数值可直接
比较。模型实现差异详见 `docs/qtl_model_implementation_comparison.md`。

## 新旧 DNA-FM checkpoint 专项图

checkpoint 专项图固定在 BF16 内比较 step150000 与 step340000，避免混入精度设置变化：

```text
Δ metric = metric(step340000) - metric(step150000)
```

正值表示新 checkpoint 更高，负值表示旧 checkpoint 更高。eQTL 差值按同一 tissue 或 organ
严格配对，matched-QTL 差值按同一 task 和 matching distance 严格配对。

## step340000 精度专项图

精度专项图固定 checkpoint 为 step340000，差值统一定义为：

```text
Δ metric = metric(FP32) - metric(BF16)
```

因此它反映推理精度配置差异，不应解释成训练步数带来的变化。

## NTv3 与新 DNA-FM 专项图

专项图以两根并排柱作为主视图：同一个 organ、tissue 或 distance 下比较 NTv3 与最新的
DNA-FM step340000 FP32。另保留差值图作为辅助视图，其差值统一定义：

```text
Δ metric = metric(DNA-FM step340000 FP32) - metric(NTv3)
```

紫色正值表示新 DNA-FM 的该项公共 metric 更高，绿色负值表示 NTv3 更高。这里相减的是同一
evaluator 输出的 AUROC、相关系数或 AUPRC，而不是两个模型的 raw prediction score。NTv3 与
新 DNA-FM 虽然使用相同的 131,072 bp context、相同 broad-tissue 映射和同一测试数据，但 checkpoint、
训练过程和输出标度不同，所以 raw score 不能逐值相减来代表生物学效应差异。

## 什么是 causal effect size

eQTL 的 causal target 来自 `data/eqtl/tables/<tissue>.tsv.gz`。数据层先保留 SuSiE
`pip > 0.9` 且出现在 positive VCF 中的 `(variant, gene, tissue)`，再读取：

- `beta_posterior` 作为 `effect_size`；
- `allele1` 作为等位基因方向锚点，内部字段名为 `effect_allele`；
- SuSiE `pip` 只表示该 variant 是该信号中因果变异的后验概率，不是 effect size。

因此这里的 causal effect size 是：在指定 tissue 中，该候选因果 variant 对指定 gene 表达量的
SuSiE 后验效应系数。其绝对值描述统计模型估计的效应强弱，正负号描述 `allele1/allele2` 约定下
表达增加或降低的方向。它不是模型预测值，也不是“致病概率”，更不等于 PIP；不同数据预处理下
也不应把它直接解释为百分比表达变化。

positive/negative membership 与 effect size 是两种标签：positive 表示该变异进入 causal/QTL
集合，但 positive 的 `beta_posterior` 可以为正也可以为负。negative 对照变异不需要 causal
effect size。只有 causal `(variant, gene, tissue)` 行参与 direction AUROC、Spearman 和 Pearson。

## 什么是“对齐后的 score”

四个 adapter 首先按 benchmark VCF 方向产生有符号预测，统一记为：

```text
raw_score = model prediction for ALT - model prediction for REF
```

由于 GTEx/SuSiE coefficient 表的等位基因顺序不一定与 VCF 的 REF/ALT 顺序相同，公共 evaluator
完全复用 Borzoi 原始脚本的符号规则：

```text
aligned_score = raw_score       if VCF_REF == allele1
aligned_score = -raw_score      if VCF_REF != allele1
```

这里内部字段 `effect_allele` 实际保存原表的 `allele1`，它的作用是方向锚点；不应仅凭字段名把它
误解为“效应一定来自这个 allele”。对齐只做必要的乘 `-1`，不会缩放、标准化或校准分数。因此：

- `auroc_sign` 用 `effect_size > 0` 作为二元真值，以 `aligned_score` 排序；
- Spearman/Pearson 比较 `effect_size` 与 `aligned_score`；
- causal classification AUROC 不关心方向，使用同 tissue 内跨 gene 的 `max(abs(raw_score))`，
  所以无需做 allele 符号对齐。

“对齐”只保证预测正负号和 coefficient 的 allele ordering 一致，不会让 AlphaGenome LFC、Borzoi
arcsinh-SED、NTv3 或 DNA-FM 的 raw score 变成同一数值尺度。跨模型应该比较公共 metric，而不是
直接比较 raw/aligned score 的大小。
