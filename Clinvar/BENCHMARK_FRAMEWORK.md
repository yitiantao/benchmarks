# ClinVar 三层评测框架

这套代码只处理当前目录已构建的 GRCh38、exact Benign/Pathogenic SNV benchmark。原始 ClinVar 到功能组的生成仍由 `build_vep_eval_groups.py` 负责；本框架不重新推断分组。

## 层与接口

| 层 | 文件 | 责任 |
|---|---|---|
| 数据 | `clinvar_benchmark/data.py` | 按版本和 `group: ...` 列选择任务、去重、读取标签与 hg38、构造 REF/ALT 窗口并核对参考碱基 |
| 模型 | `clinvar_benchmark/models.py` | 分别加载本地 Borzoi、NTv3、AlphaGenome；一条变异返回命名原始分数 |
| 评测 | `clinvar_benchmark/evaluation.py` | 统一逐变异 CSV、覆盖率、错误计数、AUROC/AUPRC；不导入任何模型库 |
| 编排 | `clinvar_benchmark/cli.py` | 参数、稳定哈希分片、运行、汇总和同键配对比较 |

`score` 阶段的通用数据流是 `Variant -> Reference.window (需要窗口的模型) -> model.score() -> 行级预测`。模型不决定样本、标签或评测指标。`evaluate` 阶段只读行级预测，不加载模型或 GPU。

## 一键运行

默认使用 2026-02、全部任务数据和已提供的模型权重。首四卡可以直接这样用：

```bash
bash scripts/run_single_gpu.sh --model ntv3 --task missense --gpu 0 --limit 100
bash scripts/run_multi_gpu.sh --model borzoi --task splice --gpus 0,1,2,3 --limit 100
bash scripts/run_multi_gpu.sh --model alphagenome --task "5'UTR" --gpus 0,1,2,3 --version 2026-09-14
```

`--limit 0`（默认）代表完整任务；先用 `--limit 1` 或 `100` 做端到端 smoke test。允许的任务名直接来自数据列，可列出：

```bash
python -m clinvar_benchmark.cli tasks --version 2026-02
```

例如 `all`、`coding`、`noncoding`、`missense`、`splice`、`intron (non-splice)`、`5'UTR`、`3'UTR`、`RNA gene`，也支持所有组合组以及 `+`、`-`。任务名有空格或撇号时用 shell 引号。单卡脚本调用对应 conda 环境；多卡脚本给每个进程指定一张可见卡，进程内均用 `cuda:0`，因此 AlphaGenome 本地 device 0 也是正确的。可改 `--gpus` 列表，不写时默认 `0,1,2,3`。已有分片文件不会被静默覆盖；要重新评分时加 `--force`。

运行时先显示“筛选任务变异”和“加载模型”阶段；筛选完成即打印所用变异总数及 Benign/Pathogenic 数。推理阶段显示已完成/总变异数、百分比、预计剩余时间，以及实时 `ok/B/P/skip` 数。单卡使用原地刷新的进度条（另显示速度）；多卡为了避免四个进程抢终端行，每分片定期输出一行进度条。结束后打印选中数、成功评分数（分别列出 Benign/Pathogenic）及 `no_score/error`；`metrics.json` 汇总全任务的选中和成功评分类别数。大任务为了给进度条提供精确总数，会先在每个进程中读一遍任务变异列表，因此推理前会占用一定内存。

`--limit` 是按数据文件顺序取前 N 条，仅用于 smoke test，**不是随机/分层抽样**；前几十条可能全属同一标签，所以对应 AUROC/AUPRC 会是 `null`，不能据此判断模型表现。

输出默认在 `runs/<version>/<task>_<hash>/<model>/<limit>/`：每个分片一个 `shard-...csv.gz`，评测结果为 `metrics.json`。CSV 每行保留 `variant_key, label, score_name, raw_score, score, status, error, extra_scores_json` 等字段。`score` 列按“越高越像致病相关扰动”的方向翻转，但**不是概率**。单卡/多卡使用相同字段、同一任务键和确定性分片，评测会检查重复键、缺失键和标签不一致。

三模型都跑完后，对共同成功评分的变异做严格同键比较：

```bash
/root/anaconda3/envs/ntv3/bin/python -m clinvar_benchmark.cli compare \
  --version 2026-02 --task missense --limit 100 --num-shards 4
```

这里的 `--num-shards` 必须与三模型输出时一致；若模型使用不同分片数，需要分别重跑到同一分片布局。`paired_comparison.json` 记录共同样本数和三个模型在该共同集合上的 AUROC/AUPRC。各自的 `metrics.json` 则按各自成功评分集合计算，不能直接当成配对比较。无分数的变异不会被填成 0。

## 中断和损坏分片的恢复

同一任务/模型/版本/`--limit`/分片数组合不要同时启动两份作业；新版本会用同分片文件锁立即阻止并发重写，并为每次写盘使用唯一临时文件。若运行中断或 gzip 损坏，沿用原参数并加 `--resume`：

```bash
bash scripts/run_multi_gpu.sh --model borzoi --task splice --version 2026-02 \
  --gpus 0,1,2,3 --resume
```

恢复会逐行校验模型、版本、任务、变异键、标签、分数方向及有限数值，仅复用有效的 `status=ok` 行；无法读取的压缩尾部会被舍弃并补算。原文件先复制成同目录的 `*.backup-*`，完成后才原子替换最终分片。恢复进度条只计待补算变异，避免复用行瞬间推进导致 ETA 虚低；每个分片结束时显示 `reused` 和 `newly_attempted`。若某分片已完整，则不加载模型。`--resume` 与覆盖重跑的 `--force` 互斥。严重压缩损坏时，损坏点之后的行通常不能恢复，因此仍可能需要补算较多变异。

## 模型定义及对齐边界

| 模型 | 本地资源 | 默认原始分数 | 解释与主要不等价处 |
|---|---|---|---|
| NTv3 | 官方 `NTv3_100M_post` safetensors 和已缓存的官方 remote code | REF 输入上 ALT-REF 的位置 logit 差 `position_llr` | 对齐 VEP-eval notebook 的位置 LLR 定义，`score=-position_llr`；但参考 notebook 是 **650M pre/post、131072 bp**，这里是用户提供的 **100M post、32768 bp**，绝不能标成同一模型结果。另存 `seq_pllr`、可用时 `log2fc_max`。位置分数并非校准的致病概率。 |
| AlphaGenome | 本地 `all_folds` Orbax，AlphaGenome Research 与本地 Feather/FASTA | 推荐 scorer 的所有输出中最大绝对 `raw_score` | 已测本地样本返回 raw，但没有参考 API notebook 所用的 `quantile_score`；因此当前只能对齐“跨输出取最大绝对值”的聚合形式，**不能复现 notebook 的 quantile 数值或排序**。使用 1 MB 窗口；不同输出的 raw 尺度也不统一，最大值可能被高幅值输出主导。若日后本地校准数据可用，应另加命名分数，不能覆盖此列。无 scorer 行记为 `no_score`。 |
| Borzoi | replicate 0 H5、官方架构参数、human target 表 | RNA 轨道所有位置/track 的最大绝对 ALT-REF 变化 | VEP-eval **没有 Borzoi notebook**；这是新定义的调控效应 proxy，不是训练好的 ClinVar 致病分数。仅 replicate 0，没有四 replicate ensemble、反向互补/shift ensemble；不同轨道尺度和窗口大小会影响最大值。 |

三模型分数数值大小不可互相比较，只能在同一标签任务、同一变异集合上比较排序指标。尤其 Borzoi RNA 输出偏向表达机制，AlphaGenome 涵盖多个机制，NTv3 是碱基语言模型 logit；对 coding、splice、UTR 等任务的能力覆盖本就不同。ClinVar 临床分类也不是纯功能效应真值，功能组/星级、疾病和基因分布会造成组成偏差。AUROC/AP 均在 `status=ok` 的行上算，AP 对正例比例敏感；报告同时给出覆盖率和正负例数。无双类别的任务指标为 `null`。

## 当前验证状态与风险

- 已运行不需要 GPU 的单元测试、任务列读取、模型环境中的 CLI 帮助以及 bash 语法检查。
- 沙箱内无法连接 NVIDIA 驱动；在获准的沙箱外命令中已确认 0–3 号 A100 可用。三模型均已对 `2026-02 / missense / --limit 1` 完成真实单卡推理且 `status=ok`，共同键比较也通过。NTv3 已用 0–3 号卡完成 `--limit 12` 多卡端到端验证（4 分片、12/12 成功）；Borzoi/AlphaGenome 的多卡及完整 benchmark 尚未实测，不应把单条或小样本结果视为性能验证。
- AlphaGenome 本地加载是重型 JAX 初始化，多卡是每卡独立进程/副本；可用显存和首次编译时间取决于节点。NTv3 `post` 两次前向及 Borzoi 524288 bp 双等位基因前向也可能耗时。失败行会保留错误原因；如果整个分片全报错，命令以非零状态退出。
- 当前 Borzoi 的 RNA 最大值是全轨道/全 bins 汇总，没有使用 MANE 基因或组织条件。想让某类任务更有生物学针对性，应新增显式模型分数名，并分别评测，不能悄悄改写现有 `rna_max_abs_delta` 语义。
