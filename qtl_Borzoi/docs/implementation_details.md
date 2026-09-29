# Benchmark 执行机制与数据语义

本文记录当前仓库相对上游 Borzoi paper 脚本做的本地化改造。命令示例见
[`gpu_execution.md`](gpu_execution.md)，eQTL 专用流程见 [`borzoi_eqtl.md`](borzoi_eqtl.md)。

## 1. 环境和本地路径

- Borzoi 默认 Conda 环境是 `borzoi_py310`，AlphaGenome 默认环境是 `alphagenome`。
- 本机上游源码默认位置是 `/data/yitian_workspace/DNA/borzoi` 和
  `/data/yitian_workspace/DNA/borzoi-paper`；运行时使用本仓库 `src/` 中固定的代码副本，不要求
  原目录一直存在。
- `setup_qtl_benchmark.sh` 用 Conda/Bioconda 安装二进制科学计算包和 `pybedtools`，再以
  `--no-deps` editable 方式安装本地 Baskerville/Borzoi。这样避免 pip 隔离构建缺少
  `setuptools`/`Cython`、`setuptools_scm` 不兼容，以及 NumPy/Pandas ABI 混装。
- TensorFlow C++ 启动信息默认以 `TF_CPP_MIN_LOG_LEVEL=3` 隐藏。cuFFT/cuBLAS factory 重复注册和
  TF-TRT 未找到通常是每个 worker 初始化 TensorFlow 时的日志，不代表多卡失效；如需诊断可设置
  `BORZOI_TF_CPP_MIN_LOG_LEVEL=0`。

## 2. 单卡、多卡和 replicate 屏障

多卡采用“一张物理 GPU 一个 Python 进程”。父进程把 `CUDA_VISIBLE_DEVICES` 分别设为单个物理
GPU，因此每个子进程内部只看到一张卡。VCF 按记录序号切成连续 shard，不按染色体切分。

对四个 published replicate，执行次序是：所有 GPU 共同计算 replicate 0 的不同数据 shard，
等待全部成功并合并；再共同计算 replicate 1，依次到 replicate 3。调度器以 `fNcN` job 名识别
replicate，并在 replicate 之间放置 barrier，不会采用“一张卡长期负责一个 replicate”的方式。

每个 worker 写 `merge_pos/jobN/sed.h5` 或 `merge_neg/jobN/sed.h5`。合并器按 shard 顺序拼接 SNP
字段和分数，并给 gene 行中的 `si` 加上累计 SNP offset；先写 `sed.h5.tmp`，完成后原子替换正式
`sed.h5`。只有 worker 正常退出才创建 `jobN/.complete`。断点续跑要求 HDF5 和完成标记同时存在，
从而不会复用异常退出留下的空文件。

worker 读取 launcher 生成的 `options.pkl` 时，会用当前命令行 parser 的默认值补齐旧 pickle 中
缺失的新字段，同时保留 pickle 已有值。这样代码更新期间已排队的旧任务也不会因为新增参数出现
`AttributeError`；重新启动顶层命令时，launcher 会把本次参数重新写入 pickle。

AlphaGenome 同样按 variant shard 并行。不同 shard 写共享的 WAL SQLite cache，数据库初始化设置
60 秒 busy timeout，并对 `database is locked` 做有限重试；所有 shard 成功后只导出和评估一次。

## 3. variant batch 参数

Borzoi 的 `--variant-batch-size N` 表示每次模型前向计算 `N` 个变异。每个变异保持
`REF, ALT` 相邻，所以实际 TensorFlow sequence batch 是 `2*N`：

```text
N=1: [variant0_REF, variant0_ALT]

N=3: [variant0_REF, variant0_ALT,
      variant1_REF, variant1_ALT,
      variant2_REF, variant2_ALT]
```

因此 `N=1` 不是只预测一个 allele，而是选中一个具体 variant，在同一个以该变异为中心的
524,288 bp 窗口内分别构造 VCF REF background 和 ALT sequence，再由下游 SED 等统计量计算
ALT−REF effect。AlphaGenome 对一个 variant 也比较 REF/ALT，但当前配置使用 1,048,576 bp
窗口；两者的 benchmark variant 可以相同，模型实际接收的序列张量长度仍然不同。

参数覆盖 eQTL SED/SAD、sQTL、paQTL 和 iPaQTL，并通过单卡、多卡一键脚本原样传递。默认值为
`1`，也可用环境变量 `BORZOI_VARIANT_BATCH_SIZE` 设置。多卡 shard 和单 worker 内的 batch 是两层
独立机制：四卡且 `N=2` 表示四个进程各自一次处理 2 个变异，而不是四卡共同执行一个 TensorFlow
batch。

batch 只改变一次前向的输入数量，不改变 VCF 顺序、REF/ALT 配对、HDF5 schema、replicate
ensemble 或评价逻辑。最后不足 `N` 个变异时使用较小的尾 batch。Borzoi 序列很长且 `--rc` 会增加
推理开销，建议使用 `1` 或 `2`；OOM 时把参数调小即可。

当前 benchmark 默认传递 `--rc`。`SeqNN.build_ensemble()` 会对 sequence batch 中的每条 REF/ALT
序列分别计算 forward 和 reverse-complement 方向，将反向输出恢复到原方向后取平均。因此
`N=1` 的外层基础 batch 仍是两条序列 `[REF, ALT]`，但模型图内部会为二者各执行两个方向；这会
增加计算量，不会增加 benchmark 中的 variant 数，也不会改变最终分数的 ALT−REF 定义。默认
`--shifts=0`，没有额外 shift ensemble。

四-replicate ensemble 配置会顺序执行四个 published replicate，之后再做 ensemble、
tissue HDF5 拆分和 coefficient 评价。因此“一个 variant 使用 524,288 bp 窗口”不能直接推出
Borzoi 会比使用 1,048,576 bp 窗口的 AlphaGenome 更快：Borzoi 对每条基础 REF/ALT 输入还包含
RC 两个方向和四个独立 replicate，且旧论文格式有额外磁盘 I/O。`variant-batch-size` 只改善单次
调用的吞吐，不会减少这些 replicate/RC 计算。

`scripts/all_borzoi_qtl.sh` 为控制正式 per-tissue 大集合的耗时，默认只运行
`models/replicate_0/model0_best.h5`。因此该入口的默认计算量不再包含四倍 replicate；RC、REF/ALT
和 HDF5 I/O 仍然存在。单模型结果名称为 `borzoi-replicate-0`，不能标记或解释为四模型 ensemble。
默认输出使用独立的 `outputs/borzoi_replicate_0_all_qtl`，因为上游 experiment/HDF5 路径不包含
model name；若与四模型 run 共用 output root 和 scope tag，断点续跑可能错误复用已有 ensemble
产物。只有确认目录全新时才应通过 `BORZOI_ALL_QTL_OUTPUT_DIR` 改到其他位置。

### 3.1 统一的 batch 进度日志

AlphaGenome 的日志例如：

```text
[predict/alphagenome-all-folds] 6850/6859 rows=6270 batch_elapsed=3.7s
```

这里的分母是该 GPU shard 本次尚待处理的唯一 variants；`batch_elapsed` 是整个 variant batch
（例如 N=2）的耗时，`rows` 是本批产生的 gene/tissue 预测行数，不是 variant 数，也不是模型
调用数。八张卡会各自打印一套接近总唯一 variant 数八分之一的进度。

Borzoi 的四条推理路径（eQTL/sQTL SED、eQTL SAD、paQTL、iPaQTL）现在也按同一核心字段输出：

```text
[predict/borzoi-replicate-0/f0c0/neg/job3] 52/4968 rows=5 batch_elapsed=1.6s
```

两边统一后，`52/4968` 都表示当前 GPU shard 已完成/总 variant 数；每处理完一个
`variant-batch-size` 才打印一行，尾 batch 可以小于配置值。`batch_elapsed` 是这一个 batch 从构造
REF/ALT 输入、模型前向到结果整理和写入的实测 wall-clock，不再是 tqdm 的滚动 ETA。

Borzoi 前缀额外保留 `replicate/split/job`，因为多个 GPU、正负集合和 replicate 的日志会交错；
这些后缀只用于定位 worker，不改变核心统计口径。顶层 `--model-name` 通过
`BORZOI_MODEL_NAME` 传给 worker，因此单模型正式入口会显示 `borzoi-replicate-0`。

`rows` 都表示本批实际写出的逻辑结果行，但两个模型的底层 schema 不同，数值本身不能用来比较
模型速度或输出“细致程度”：AlphaGenome 的长表会展开 annotation/assay/tissue；Borzoi SED 的
一行通常是一个 variant-gene（paQTL/iPaQTL 还包括 PAS 行），多条 tissue/track 分数存放在该行的
数组列中。SAD 的一行对应一个 variant。可直接比较的是 variant 进度和 batch wall-clock；结果质量
应继续使用统一评价层的 shared metrics。

正式 `per_tissue_max_2000` 输入包含 39,780 条 negative merged records；除以 8 个 shard 后约为
4,973，所以日志中每卡约 `4966–4972` 的分母是预期值（少量无可评分 gene 的记录会在推理前
过滤）。positive merged records 为 17,248；Borzoi 分 split 推理，若同一坐标同时出现在两个
split，仍可能在两个 HDF5 流程中各处理一次。AlphaGenome unified runner 则先按
`chrom:pos:REF:ALT` 跨 tissues/splits 去重，每个唯一 variant 只写一次 SQLite cache。

单行 `batch_elapsed` 现在可以作为相同 batch 口径的即时吞吐观察，但完整速度比较仍应使用完整阶段
wall-clock 和最终处理的逻辑 variant 数，例如 runner 输出的 `[performance]`，并分别报告
replicate、RC、GPU 数和 scope。已启动的 Python worker 已加载旧代码，不会在进程中途切换格式；
后续新启动的 worker 才会采用新格式。日志改动不改变预测值，因此已有有效产物无需为指标正确性
重跑；只有需要一份从头到尾口径统一的日志时才需要重跑。

当前配置第一层卷积长度为 524,288、通道数为 512。`N=4` 时 REF/ALT sequence batch 为 8，
第一层输出元素数为 `8 × 524288 × 512 = 2,147,483,648 = 2^31`。TensorFlow GPU launch config
使用有符号 32 位 work-element 计数，因而会报
`work_element_count >= 0 (-2147483648 vs. 0)` 并以 `SIGABRT` 退出。这不是普通 warning，也不是
多卡调度或显存问题。该模型的理论硬上限为 `N=3`，启动脚本和 worker 都会在加载 GPU 模型前
校验；推荐值仍是 `2`。

```bash
BORZOI_GPU_IDS=0,1,2,3 \
./scripts/run_borzoi_multi_gpu.sh \
  --tasks sqtl \
  --max-variants 10 \
  --variant-batch-size 2 \
  --model-dir models \
  --model-name borzoi-four-replicates \
  --conda-env borzoi_py310 \
  --data-dir data \
  --output-dir outputs/my_borzoi_sqtl
```

AlphaGenome unified runner 也接受 `--variant-batch-size N`，但语义不是 Borzoi 的 tensor batch。
本地 AlphaGenome `DnaModel.score_variants` 会建立至多 N 个线程，每个线程独立调用一次本地 JAX
`score_variant`。默认 `1`；设置 `2` 前应先用少量 variant 观察显存，因为并发调用可能显著增加
激活峰值。该参数只影响调度吞吐，不改变 score、variant 顺序、SQLite cache identity 或最终评价。

## 4. 染色体、训练 fold 和当前评价范围

本仓库从官方数据整理出的合并 QTL VCF 覆盖 `chr1`–`chr22` 和 `chrX`，不只有个别染色体；当前
没有 `chrY`/`chrM`。`--max-variants` 是按记录挑选小子集，可能改变小样本中实际出现的染色体，
正式全量输入不做按染色体过滤。当前统一指标也是跨输入变异整体计算，并未额外输出逐染色体指标。

四个 published 权重来源是上游 `f3c0`–`f3c3`：它们是同一个 fold 3 划分上的四个 cross/model
replicate，训练/验证/测试区间相同，不是四个不同 test fold。setup 为兼容本地 launcher，把它们
映射到 `f0c0`–`f0c3` 目录。官方模型说明中 fold 3 是 test、fold 4 是 validation；这里的 genomic
fold 是平衡后的 syntenic contig component 分组，不等同于“一条染色体一个 fold”。

当前包中没有官方完整 `sequences_human.bed.gz` 区间表，benchmark 因而会评分所选 QTL VCF 中的
全部变异，不会根据模型的 train/validation/test fold 再过滤。也就是说，当前结果可复现官方 QTL
任务输入和四 replicate ensemble，但不能仅凭运行目录名声称它是“只在 fold 3 held-out 区间”的
评价。若要做严格 held-out-only 分析，需要先加入官方 sequence 区间及 fold 标签，再按变异窗口与
区间的关系过滤，并单独输出样本清单和指标。

## 5. 输出兼容性

上游 fold launcher、统计名称和 HDF5 字段保持不变：sQTL 使用 `nDi/span`，paQTL 使用
`COVR/utr3`，iPaQTL 使用 `COVR`，eQTL 保留 SED/SAD 路线。多卡合并与 variant batch 都发生在
这些统计之前，因此统一评价层仍读取相同产物。更换模型或 batch 设置做正式对比时建议使用新的
`--output-dir`；已有完整输出会被断点续跑逻辑复用，而不会仅因 batch 参数变化自动重算。
