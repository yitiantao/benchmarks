# AlphaGenome 接入 Borzoi GTEx eQTL benchmark：实现说明

## 目标和边界

这里复用 Borzoi benchmark 的数据划分和四项评价指标，但用 AlphaGenome 的本地
`all_folds` checkpoint 产生 expression variant effect。实现的是 gene-aware、
tissue-aware 路线，不是只在 `pos_merge.vcf`/`neg_merge.vcf` 上产生一个匿名标量。

原始实现的关键代码是：

- `src/borzoi_scripts/borzoi_sed.py`：对 REF/ALT 预测按 gene 区域聚合，产生 SED/logSED；
- `src/borzoi_scripts/borzoi_bench_gtex_folds_sed.py`：合并、拆分 tissue 结果；
- `src/borzoi_scripts/borzoi_gtex_coef_sed.py`：连接 `(variant, gene)` 和 SuSiE 表并计算指标。

AlphaGenome 对应实现分成三个可替换组件：

- `benchmark/qtl_benchmark/models/alphagenome.py`：模型层；
- `benchmark/qtl_benchmark/data/gtex_eqtl.py`：共享数据层；
- `benchmark/qtl_benchmark/evaluation/gtex_eqtl.py`：共享评价层。

`benchmark/qtl_benchmark/alphagenome_eqtl.py` 只保留旧 CLI 和 helper 的兼容包装。完整框架见
[`framework.md`](framework.md)。

## 模型层

`AlphaGenomeModel.setup_task()` 使用：

```python
dna_model.create(local_checkpoint, organism_settings=..., device=gpu)
```

FASTA、GTF Feather 和由 GTF 生成的 splice-site starts/ends Feather 都从本地读取，避免在
推理阶段下载参考文件。每个变异调用官方推荐的
`RECOMMENDED_VARIANT_SCORERS["RNA_SEQ"]`。这是 gene-mask log-fold-change scorer；
`score_variant_table()` 返回完整 tidy 表，至少包含：

```text
gene_id, gene_strand, gtex_tissue, track_name, track_strand, raw_score
```

模型层不决定 tissue benchmark 的聚合方式。保留的 `predict_batch()` 会把表压成单个
variant score，只供通用 QTL 接口使用。

## Benchmark 层

对每个选择的 GTEx tissue：

1. 读取 `<tissue>_pos.vcf(.gz)` 和 `<tissue>_neg.vcf(.gz)`；
2. 从 `tables/<tissue>.tsv.gz` 选择 `pip > 0.9` 的 causal `(variant, gene)`；
3. 对每个唯一 `(chrom, pos, REF, ALT)` 只运行一次 AlphaGenome；
4. 在相同 `gene_id + gtex_tissue` 内对 RNA tracks 求均值；
5. 将均值写进 SQLite 缓存；
6. 从缓存产生 coefficient 和 classification 两张评价表。

使用 AlphaGenome metadata 的精确 `gtex_tissue` 名称，而不是 Borzoi target label 的宽泛
关键词。例如 `Brain_Cortex` 使用 AlphaGenome 的 `Brain_Cortex` track；原 Borzoi 脚本因
其训练 target 命名方式，对所有 brain tissues 使用关键词 `brain`。这一区别是两个模型
输出体系之间必要且显式的映射，不改变数据正负集或指标定义。本地 metadata 已验证包含
benchmark 的全部 49 个 tissues。

### Coefficient 指标

对 causal 表中的每个 `(variant, gene)` 取同 tissue、同 gene 的 AlphaGenome LFC。找不到
完整 gene 时按原 Borzoi evaluator 记为 0。AlphaGenome 分数方向是 ALT-REF；若 VCF REF
不等于 SuSiE 表的 `allele1`，分数乘以 -1。随后计算：

- `auroc_sign`：`beta_posterior > 0` 对 signed prediction；
- `spearmanr`：`beta_posterior` 与 prediction 的秩相关；
- `pearsonr`：二者的 Pearson 相关。

AlphaGenome scorer 已经输出 gene-level log fold change，不再施加 Borzoi raw SED 路线的
`arcsinh`。否则会把两个模型不同定义的 effect score 重复变换。

### Classification 指标

每个变异先在同 tissue 内跨 genes 取最大绝对值，再用 positive/negative VCF 标签计算
`auroc_class`。没有任何完整 gene score 的变异与 Borzoi SED 的行为一致，不进入该项
分类评价。

与原 evaluator 相同，只有 causal coefficient 行数严格大于 `--min-variants`（默认 32）
时才报告该 tissue 的四项指标。smoke test 仍输出该 tissue 行，但 `eligible=false`，指标
为 NaN，便于检查管线而不误读小样本结果。

## 缓存和可恢复性

`predictions.sqlite` 包含：

- `variants`：完成状态、坐标、alleles 和单条耗时；
- `predictions`：`variant_key, gene_id, tissue, score, n_tracks`；
- `metadata`：模型类和 JSON 配置的 SHA-256 identity。

每条变异成功后立即事务提交。因此任务被调度器中止时，重跑同一命令会跳过完成项。
配置 identity 不同却复用同一缓存时会直接报错，防止把不同 checkpoint 或 sequence length
的结果混在一起。

## 资源与验证状态

当前已完成的验证包括：

- 本地 checkpoint 的 Orbax 必需文件和总大小检查；
- JAX 识别 NVIDIA A100 80 GB；
- AlphaGenome RNA-seq metadata 有 768 tracks、54 个 GTEx tissues，覆盖本 benchmark 49 个；
- GENCODE 41 GTF 转换为 0-based Feather（1,945,667 行），并生成 splice-site starts/ends；
- 模拟 model table 的端到端测试，覆盖 track 均值、gene join、allele 翻转、四项指标和缓存；
- prediction/evaluation CLI 的无模型阶段测试。

注意 XLA 首次编译同时需要较多主存。`run_alphagenome_local.sh` 默认在 cgroup 可用主存小于
10 GiB 时提前退出，以避免只留下不清晰的 `Killed`/exit 137；可用
`ALPHAGENOME_MIN_FREE_GIB=0` 显式跳过预检。

当前 Codex 执行进程处于 16 GiB cgroup，且共享进程已占约 13 GiB。真实 checkpoint restore
在该限制中被 OOM killer 以 exit 137 终止；这是主存限制，不是 checkpoint、CUDA 或 GPU
探测失败。应在主存余量更大的普通 shell/job 中执行 `scripts/run_alphagenome_local.sh`。

## 2026-09-19：缺失 gene 元数据导致的 KeyError 修复

### 现象

eQTL 分片推理在 AlphaGenome 的 exon gene-mask 提取阶段失败，例如：

```text
KeyError: "['ENSG00000006194.10'] not in index"
```

错误发生在 AlphaGenome 生成 tidy score 之前，因此 benchmark 后处理阶段已有的 Ensembl
版本号归一化（如 `.10` → 无版本 ID）无法解决它。

### 根因

输入文件 `gencode41_basic_nort.gtf` 是筛选后的 GENCODE basic 注释。它保留了 16 个
gene_id 的 transcript/exon 行，却没有保留相应的 `Feature == "gene"` 行。转换出的 Feather
因而也包含这 16 个孤儿 gene_id。AlphaGenome 先从 transcript TSS 找到 gene_id，再通过
gene 行查询 `gene_name`、`gene_type`、strand 和坐标；严格的 Pandas `.loc` 查询因此抛出
`KeyError`。`ENSG00000006194.10`（ZNF263）是这 16 个 ID 之一。

### 修改

- `scripts/prepare_alphagenome_reference.py` 新增 `repair_missing_gene_rows()`；
- 转换时检测 transcript/exon 中存在、gene 表中缺失的 gene_id；
- 从同一 gene_id 的注释重建唯一 gene 行，坐标取全部保留注释的最小 `Start` 和最大
  `End`，并保留 chromosome、strand、gene name/type 等 gene-level 元数据；
- 新增单元测试，覆盖孤儿 gene 行的重建与 transcript 字段清理；
- 重新生成 `reference/alphagenome/gencode41_basic_nort.gtf.feather`，新增 16 行，总行数从
  1,945,651 变为 1,945,667；splice-site Feather 未变化；
- 同步更新 `SHA256SUMS`。

### 验证

- 所有 transcript gene_id 都能在 gene 索引中找到：缺失数由 16 变为 0；
- AlphaGenome `RNA_SEQ` 实际使用的 `GeneMaskType.EXONS` 路径可成功提取
  `ENSG00000006194.10 / ZNF263`；
- `benchmark.tests.test_prepare_alphagenome_reference` 与
  `benchmark.tests.test_alphagenome_eqtl` 共 8 个测试通过；
- 修复后 GTF Feather SHA-256：
  `73058b070c01e1a10a94aed67eff756794985a67cd9b4afdfd23783ec1cb561d`。

原预测命令无需增加参数。失败的 shard 直接按原命令重跑即可；SQLite prediction store
会保留并跳过此前已成功提交的变异。

## 2026-09-19：与 Borzoi 的共享输入比较视图

AlphaGenome 的 `max_2000` 是每 tissue、每 split 上限，而 Borzoi 的 `max_2000` 是跨 tissues
轮询后的每 split 全局上限，因此两个原生 run 不能仅凭相同目录标签认定测试变异完全相同。
新增比较器改用 Borzoi subset 作为共享测试集，并要求其中所有变异都已存在于 AlphaGenome 和
Borzoi 缓存；现有较大的 AlphaGenome run 可以直接提供共享子集分数，无需重新推理。

后续正式 Borzoi 入口已经改为显式 `per-tissue` scope，与这里的 AlphaGenome 限额一致；旧
global run 使用不同 tag 继续保留。比较器仍会用逐 tissue 计数和 membership/target SHA-256
再次验证，而不会只相信命令行参数相同。

比较结果同时保留 AlphaGenome 精细 GTEx tissue 指标，并增加按 Borzoi tissue keyword 聚合的
一组指标，原预测和原指标不受影响。运行命令、输入一致性的边界和输出文件定义见
[`eqtl_model_comparison.md`](eqtl_model_comparison.md)。
