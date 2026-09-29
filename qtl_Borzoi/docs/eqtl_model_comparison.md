# AlphaGenome / Borzoi eQTL 一致比较

## 为什么需要额外比较视图

两个模型的原生 eQTL 输出都保留，但它们不能直接按日志中的一串数字横向比较：

- AlphaGenome 原生输出精确的 GTEx tissue，例如 `Brain_Cortex`，并保留 gene 和 tissue；
- Borzoi 的 target label 使用较宽的关键词，例如所有 brain tissue 都匹配 `brain`；
- 旧 AlphaGenome 的 `--max-variants 2000` 是每个 tissue、每个 split 最多 2000，旧 Borzoi
  默认是所有 tissues 轮询后每个 split 全局最多 2000；现在已用显式 scope 区分；
- Borzoi 论文脚本还会分别打印 replicate/ensemble 的 `SED`、`logSED` 指标，这些是原生复现
  结果，不等同于统一 evaluator 的单一结果表。

这里的 `max-variants` 语义差异没有生物学上的必要性，而是两条历史执行路径造成的：Borzoi
论文脚本先对跨 tissue 的 merged VCF 做推理，本地 smoke/限额逻辑因而用跨 tissues 轮询后的
全局上限，直接限制昂贵的 GPU 推理总量；AlphaGenome unified data loader 则逐 tissue 读取 VCF，
原先自然地在每个 tissue、每个 split 截断。相同参数名表达了不同范围，这是接口设计上的历史
遗留，不应作为正式模型比较的默认假设。

Borzoi 的计算成本也是当时保留全局上限的实际原因：四-replicate 配置会依次运行四个 published
replicate，并为每条 REF/ALT 序列做 forward/reverse-complement ensemble。若把 eQTL 从当前
全局正负各 2000 改为每 tissue、每 split 各 2000，去重后的 variant 数会接近 AlphaGenome
当前较大 run 的规模，Borzoi 计算时间会显著增长。窗口较短并不代表整条 Borzoi pipeline 一定
更快，因为 replicate、RC、HDF5 合并/拆分和 coefficient 阶段都产生额外工作。

因此新增的比较流程不覆盖任何原输出，而是在 Borzoi 已选择的 eQTL subset 上生成三组结果：

| view | 输入样本 | tissue 语义 | 用途 |
|---|---|---|---|
| `alphagenome_native` | 共享 subset | AlphaGenome 精确 tissue | 保留 AlphaGenome 原生分辨率 |
| `alphagenome_borzoi_grouped` | 同一共享 subset | 按 Borzoi keyword 聚合 | 与 Borzoi tissue 口径对齐 |
| `borzoi_native` | 同一共享 subset | Borzoi 原生 broad target | 保留 Borzoi 原生结果 |

三组结果全部由同一个 `GTExEQTLEvaluator` 计算 `auroc_sign`、`spearmanr`、`pearsonr` 和
`auroc_class`。这组指标才是推荐的模型间横向比较入口。论文原脚本的 replicate、SED/logSED
结果仍保留在 Borzoi experiment 目录，用于复现和分析模型内部差异。

## “输入完全一样”的准确边界

在新增比较结果中，两个模型使用完全相同的 benchmark-level 输入：

- 相同的 `chrom:pos:REF:ALT` 变异集合；
- 相同的 tissue 正负例 membership；
- 相同的 causal `(variant, gene, tissue)`、effect size 和 effect allele；
- 相同的 PIP 阈值和同一份 evaluator。

比较器会先检查共享 subset 中的每个变异是否都在两个 `predictions.sqlite` 中完成。任一模型
缺少变异就停止，不会静默按不完整交集计算。`input_audit.json` 记录覆盖数、缺失 key、
membership/target 的 SHA-256，以及 `same_benchmark_inputs`。

但模型张量层面的输入不是完全相同，也不应为了“统一”而改变模型定义：

| 项目 | AlphaGenome | Borzoi |
|---|---|---|
| hg38 变异坐标和 REF/ALT | 相同共享 subset | 相同共享 subset |
| 输入序列长度 | 1,048,576 bp | 524,288 bp |
| 序列预处理/模型架构 | AlphaGenome 原生 | Borzoi 原生 |
| 输出 tracks | AlphaGenome RNA-seq metadata | Borzoi GTEx targets |
| 原生 tissue 分辨率 | 精细 GTEx tissue | keyword broad group |

所以 `same_benchmark_inputs=true` 的含义是“测试样本与标签相同”，不是“两种架构收到逐元素相同
的神经网络张量”。Borzoi 日志中出现 `alt ... matches reference genome` 时，其上游脚本会按参考
基因组修正序列构造；比较表仍使用原 benchmark 的 variant key 和 effect-allele 方向。这类模型
原生预处理差异不会被隐藏。

## tissue 对齐方法

`alphagenome_borzoi_grouped` 使用 `models/borzoi.py` 中同一份 `TISSUE_KEYWORDS` 映射。例如，
对某个 `(variant, gene)`，所有 AlphaGenome brain tissues 的分数按 `n_tracks` 加权平均，得到一个
`brain` 分数，再赋给该变异实际参加评价的 brain benchmark tissues。这样是从细粒度向粗粒度
聚合；反向从 Borzoi broad target 恢复 `Brain_Cortex` 等精细信号是不可能的。

AlphaGenome 原生分数本身不被改写。聚合后的预测与原生预测分别写入独立目录。

## 运行方式

两个模型的 `max_2000` eQTL 缓存都完成后运行：

```bash
# 正式、每 tissue 每 split 上限一致的比较
./scripts/compare_alphagenome_borzoi_eqtl.sh \
  --max-variants 2000 --limit-scope per-tissue \
  --borzoi-model-name borzoi-replicate-0

# 已有旧 global run 的计算量受控比较
./scripts/compare_alphagenome_borzoi_eqtl.sh \
  --max-variants 2000 --limit-scope global \
  --borzoi-model-name borzoi-four-replicates
```

该命令不重新运行 AlphaGenome 或 Borzoi 神经网络，也不需要 GPU。它只读取双方 unified run 的
`predictions.sqlite`，抽取共享 `(variant, tissue)`、生成 tissue 对齐视图并重新执行统一 evaluator。
已有较大的 AlphaGenome run 可以直接提供 Borzoi subset；不需要为了裁成相同样本数重新推理。

只有以下情况需要补跑模型：所选 `max_N`/数据版本发生变化、checkpoint 或模型配置发生变化，
或者 `input_audit.json` 报告某一方存在 missing variant。Borzoi 论文 HDF5 已完成但 unified
`predictions.sqlite` 尚未产生时，只需运行 unified 转换/评价阶段，不需要重做 GPU 推理。对于修改
脚本之前已经启动且仍在运行的旧 shell，稳妥做法是在其结束后手动执行上面的比较命令；该命令可
重复运行，不会修改两个模型的原预测缓存。

REF/ALT batch、524,288/1,048,576 bp 窗口以及 reverse-complement 的模型内部计算语义详见
[`implementation_details.md`](implementation_details.md#3-variant-batch-参数)。

`scripts/all_borzoi_qtl.sh` 在四类 Borzoi 任务完成后会自动执行同一比较；若不需要，可设置
`BORZOI_COMPARE_EQTL=0`。`BORZOI_ALL_QTL_MAX_VARIANTS` 可以覆盖该脚本默认的 2000。

默认输出为：

```text
outputs/alphagenome_borzoi_eqtl_comparison/per_tissue_max_2000/borzoi-replicate-0/
├── input_audit.json
├── comparison_manifest.json
├── shared_input_statistics.tsv
├── view_input_statistics.tsv
├── common_metrics.tsv
├── common_metrics_summary.tsv
├── comparison.tsv
├── summary.tsv
├── alphagenome_native/
│   ├── predictions.tsv.gz
│   ├── metrics.tsv
│   └── predictions/<tissue>_{coefficients,classification}.tsv.gz
├── alphagenome_borzoi_grouped/
│   └── ...
└── borzoi_native/
    └── ...
```

- `common_metrics.tsv`：所有 view 使用同一 evaluator 得到的同 schema 长表，是主要公共比较入口；
- 其中主要同粒度比较是 `alphagenome_borzoi_grouped` 对 `borzoi_native`；
  `alphagenome_native` 作为保留精细组织分辨率的辅助公共 view；
- `common_metrics_summary.tsv`：只在所有 view 都 eligible 的共同 tissues 上计算 macro mean；
- `comparison.tsv` / `summary.tsv`：前两者的兼容宽表/汇总文件；
- 各 view 的 `metrics.tsv`：保留完整逐 tissue 指标和样本/覆盖计数；
- `shared_input_statistics.tsv`：逐 tissue 和全局的正负例、唯一 variants、targets、genes 与哈希；
- `view_input_statistics.tsv`：为每个 view 写出同一输入统计，便于自动断言完全一致；
- `input_audit.json`：判断本次结果能否声称测试输入一致；缺失 variant 会直接终止；
- `comparison_manifest.json`：记录 scope、N、公共指标列、输入哈希及原生指标保留说明。

小样本中 causal 行数不超过默认阈值 32 时，指标仍为 `NaN`。这是 evaluator 的防误读规则，
不是模型运行失败；正式横向比较应使用足够大的 subset，并同时检查 `n_score_found` 和
`n_class_score_found`。

## 已实现的限额语义

正式 benchmark 的关键不是一定选择“每 tissue 上限”或“全局上限”中的某一个，而是两个模型
必须消费同一份冻结后的 variant/membership/target manifest。当前已把两个概念拆开：

- `per-tissue`：每 tissue、每 split 最多 N，适合组织均衡的正式 benchmark；
- `global`：跨 tissues 轮询后每 split 最多 N，适合控制总 GPU 成本的 smoke/快速实验。

`run_eqtl_benchmark.sh` 使用 `--limit-scope`，总 QTL 入口使用 `--eqtl-limit-scope`。旧默认仍为
`global`，保持已有命令兼容；`all_borzoi_qtl.sh` 作为正式入口显式选择 `per-tissue`。输出 tag 分别
是 `max_N` 和 `per_tissue_max_N`，不会误复用同名旧 HDF5。subset manifest 写入 scope、实际唯一
variant 数和逐 tissue 计数，比较阶段再写 membership/target hash。

当前 Borzoi global subset 的比较结果仍然有效，因为新增比较器已经让 AlphaGenome 在完全相同
的 subset 上重新评价。若后续选择 `per-tissue`，现有 AlphaGenome 大 run 通常可以直接抽取已有
分数；Borzoi 上游 HDF5 与整份输入 VCF/行索引绑定，expanded subset 应使用新输出目录重新运行
Borzoi 推理，不能把新增 variants 原地追加进当前 HDF5。当前 global run 不需要删除或重跑，仍可
作为计算量受控的共享 subset 结果保留。

## 2026-09-19 修改记录

- 新增 `benchmark/qtl_benchmark/compare_eqtl_models.py`；
- 新增 `scripts/compare_alphagenome_borzoi_eqtl.sh`；
- `scripts/all_borzoi_qtl.sh` 在 Borzoi 全部任务完成后自动生成共享 eQTL 比较结果；
- 比较器直接从 SQLite 按共享 `(variant, tissue)` 读取，避免加载约 20 GB 的 AlphaGenome
  原始缓存或 1.4 GB 的压缩导出表；
- 新增 AlphaGenome broad-tissue 分组视图、输入覆盖审计、逐 tissue 横向表和共同 eligible
  tissue 的汇总表；
- 新增 `global/per-tissue` 显式 scope；正式 Borzoi all-QTL 入口改用与 AlphaGenome 一致的
  per-tissue/per-split 上限，并使用独立输出 tag；
- 新增公共指标长表、公共汇总表、逐 tissue 输入统计和 comparison manifest；
- Borzoi eQTL/sQTL SED、eQTL SAD、paQTL 和 iPaQTL 的推理日志改为与 AlphaGenome 相同的
  `completed/total rows=... batch_elapsed=...s` batch 口径，同时保留 replicate/split/job 标识；
- 真实 `max=2000` 验证中，两条 loader 都得到 56,024 个唯一 variants、93,258 条
  memberships、50,761 条 targets，逐 tissue 统计完全相同；全局 membership SHA-256 为
  `1835b6450a4b8847b02449cea60ee32ed0691eb1df76a57df46cc2baf83cd68a`，target SHA-256 为
  `660d316c327443578ccf6a4f8d048e1827327d8c2c08d641d30e114df34f220a`；
- 原生 AlphaGenome、Borzoi 产物及 Borzoi replicate/SED/logSED 指标均保持不变。
