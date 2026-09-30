# qtl_Borzoi：可扩展的统一 QTL benchmark

这个目录统一支持 Borzoi 论文的 eQTL、sQTL、paQTL 和 iPaQTL benchmark，并可让
AlphaGenome、Borzoi 与后续模型复用同一份数据划分、预测 schema 和评估代码。运行时不依赖
原来的 `borzoi-paper` 或 `borzoi` 目录。

## 目录结构

```text
qtl_Borzoi/
├── benchmark/qtl_benchmark/   # 统一数据、模型 adapter、评价与调度框架
├── configs/                   # Borzoi 参数和 target 表
├── data/{eqtl,sqtl,paqtl,ipaqtl}/ # 四类 QTL 的正负与合并 VCF
├── docs/paper_eqtl/           # 论文仓库原始 eQTL shell/notebook
├── licenses/                  # 三个上游项目的许可证
├── models/replicate_0..3/     # 四个已发表 Borzoi 模型
├── reference/hg38/            # hg38 FASTA、GTF 和 PolyADB
├── scripts/                   # 安装、检查和运行入口
├── src/borzoi/                # Borzoi Python 包源码
├── src/borzoi_scripts/        # Borzoi 上游命令行脚本
├── src/baskerville/           # Baskerville Python 包源码
├── environment.yml            # 固定 Python/TensorFlow 运行环境
├── SOURCE_VERSIONS.txt        # 上游 commit 和文件来源
└── SHA256SUMS                 # 打包时所有静态文件的校验和
```

运行结果统一写入 `outputs/`，不会修改 `data/`、`models/` 或 `reference/`。

## 当前四模型的权重存放路径

下面对应本仓库四个零参数 `all_*_qtl.sh` 入口。六套结果图还会同时读取历史 DNA-FM
step150000 BF16 和 step340000 BF16 结果，并与最新 step340000 FP32 结果并列。路径是**模型权重**，
不是 `outputs/` 下的 benchmark 预测或指标；除特别说明外均以本仓库根目录为基准。

| 模型 | 当前配置使用的权重位置 | 配置/入口 |
|---|---|---|
| AlphaGenome（all folds） | `/data/yitian_workspace/DNA/alphagenome_models/all_folds/`（Orbax checkpoint 目录） | `configs/alphagenome_local.json`；`scripts/all_alphagenome_qtl.sh` |
| Borzoi（replicate 0） | `/data/yitian_workspace/DNA/benchmarks/qtl_Borzoi/models/replicate_0/model0_best.h5` | `scripts/all_borzoi_qtl.sh` 默认值；可用 `BORZOI_ALL_QTL_MODEL` 覆盖 |
| NTv3（官方 100M post） | `/data/yitian_workspace/DNA/benchmarks/qtl_Borzoi/.benchmark_deps/ntv3_hf/models--InstaDeepAI--NTv3_100M_post/snapshots/b7292f0bd1b5b004d28561783a1890e1ec6f94fd/model.safetensors` | `configs/ntv3_100m_post.json`；`scripts/all_ntv3_qtl.sh` |
| DNA-FM（本地 100M post，optimizer step 340000） | `/data/yitian_workspace/DNA/benchmarks/step_0034000/model.safetensors` | `configs/dna_fm_100m_post_step_0034000.json`；`scripts/run_dna_fm_step_0034000.sh`；`scripts/all_dna_fm_qtl.sh` |

NTv3 在配置中以 Hugging Face ID `InstaDeepAI/NTv3_100M_post`、固定 revision
`b7292f0bd1b5b004d28561783a1890e1ec6f94fd` 和本地 `cache_dir=.benchmark_deps/ntv3_hf`
加载；表中的 snapshot 文件是缓存内的符号链接，不是另外训练出的权重。DNA-FM 的模型代码和
训练配置则分别固定在同一训练 run 的 `code/` 与 `config.yaml`，具体路径见
`configs/dna_fm_100m_post_step_0034000.json`。checkpoint 目录名为 `step_0034000`，其中
`trainer_state.json` 记录的 `optimizer_step` 为 340000；本文按训练状态称为 step 340000。

Borzoi eQTL 执行链、与论文脚本的对应关系和裁剪语义详见
[`docs/borzoi_eqtl.md`](docs/borzoi_eqtl.md)。
多卡调度、REF/ALT batch、断点续跑、染色体覆盖及 fold 语义集中记录在
[`docs/implementation_details.md`](docs/implementation_details.md)。
NTv3 的 gated 权重、pre/post 两条打分路线、独立环境及运行限制见
[`docs/ntv3_qtl.md`](docs/ntv3_qtl.md)。
AlphaGenome 与 Borzoi eQTL 的共享 subset、tissue 粒度对齐、输入审计和统一指标比较见
[`docs/eqtl_model_comparison.md`](docs/eqtl_model_comparison.md)。
AlphaGenome、Borzoi、NTv3，以及 DNA-FM 新旧 checkpoint/精度运行的四类 QTL 图见
[`docs/qtl_visualization.md`](docs/qtl_visualization.md)；当前命令为
`./scripts/plot_six_result_qtl.sh`。

## 统一框架（推荐入口）

四个模型现在统一以 multi-GPU 脚本作为公开 benchmark 入口；单卡同样使用该入口，只需传
`--gpus 0`：

| 模型 | 当前推荐 benchmark 脚本 | 单卡写法 |
|---|---|---|
| Borzoi | `scripts/run_borzoi_multi_gpu.sh` | `./scripts/run_borzoi_multi_gpu.sh --gpus 0 ...` |
| AlphaGenome | `scripts/run_alphagenome_multi_gpu.sh` | `./scripts/run_alphagenome_multi_gpu.sh --gpus 0 ...` |
| NTv3 | `scripts/run_ntv3_multi_gpu.sh` | `./scripts/run_ntv3_multi_gpu.sh --gpus 0 ...` |
| DNA-FM | `scripts/run_dna_fm_multi_gpu.sh` | `./scripts/run_dna_fm_multi_gpu.sh --gpus 0 ...` |

`run_dna_fm_step_0034000.sh` 是当前 DNA-FM checkpoint 的 smoke-test 参数预设，内部仍调用
`run_dna_fm_multi_gpu.sh`。`all_*_qtl.sh` 是正式 workload 参数预设。原来的模型级
`run_*_qtl_benchmark.sh` 已被相应 multi-GPU 脚本吸收，不再作为独立入口。

当前 benchmark 的主路径已经拆成三个稳定接口：

```text
DataModule（读取、筛选、划分）
        → ModelAdapter（模型推理，输出 variant/gene/tissue 长表）
        → Evaluator（统一指标）
```

AlphaGenome、Borzoi 和未来模型共享 `BenchmarkDataset`、`PredictionStore`；eQTL 使用
`GTExEQTLEvaluator`，其余三类任务使用同一个 `MatchedQTLEvaluator`。详细的数据 schema、
目录职责和新模型接入步骤见
[`docs/framework.md`](docs/framework.md)。

统一 CLI 示例：

```bash
# AlphaGenome：数据准备、模型推理、统一评价
PYTHONPATH=benchmark:src python -m qtl_benchmark.cli \
  --config configs/benchmark_alphagenome_smoke.json

# 将现有 Borzoi SED 产物送入同一预测 schema 和 evaluator
PYTHONPATH=benchmark:src python -m qtl_benchmark.cli \
  --config configs/benchmark_borzoi_smoke.json

# AlphaGenome 的 sQTL/paQTL/iPaQTL
for task in sqtl paqtl ipaqtl; do
  PYTHONPATH=benchmark:src python -m qtl_benchmark.cli \
    --config "configs/benchmark_alphagenome_${task}.json"
done
```

`--predict-only` 和 `--evaluate-only` 可以拆开两个阶段，`--force` 会清除当前模型缓存后重跑。
旧的 `run_eqtl_benchmark.sh` 与 `run_alphagenome_local.sh` 继续保留，用于兼容原有运行方式。

### 一键脚本：指定 task 和模型

一条命令会依次完成真实数据裁剪、模型推理、标准预测转换和统一评价。默认只跑 `eqtl`，
不会自动启动四个大任务。

Borzoi 指定单个模型文件：

```bash
BORZOI_CONDA_ENV=borzoi_py310 \
./scripts/run_borzoi_multi_gpu.sh --gpus 0 \
  --tasks sqtl \
  --max-variants 10 \
  --variant-batch-size 2 \
  --model models/replicate_0/model0_best.h5 \
  --model-name borzoi-replicate-0 \
  --conda-env borzoi_py310 \
  --data-dir data \
  --output-dir outputs/my_borzoi_sqtl
```

指定模型目录时，脚本会递归寻找 `model0_best.h5`/`model_best.h5`，找到的模型组成 ensemble：

```bash
BORZOI_CONDA_ENV=borzoi_py310 \
./scripts/run_borzoi_multi_gpu.sh --gpus 0 \
  --tasks paqtl \
  --max-variants 100 \
  --variant-batch-size 2 \
  --model-dir models \
  --model-name borzoi-four-replicate-ensemble \
  --conda-env borzoi_py310 \
  --data-dir data \
  --output-dir outputs/my_borzoi_paqtl
```

也可以重复传递 `--model` 精确指定 ensemble；只有明确写 `--tasks all` 才运行四类任务：

```bash
./scripts/run_borzoi_multi_gpu.sh --gpus 0 \
  --tasks ipaqtl \
  --max-variants 0 \
  --model models/replicate_0/model0_best.h5 \
  --model models/replicate_1/model0_best.h5 \
  --model-name borzoi-replicates-0-1 \
  --conda-env borzoi_py310 \
  --data-dir data \
  --output-dir outputs/my_borzoi_ipaqtl_replicates_0_1
```

AlphaGenome 使用 checkpoint 目录：

```bash
ALPHAGENOME_CONDA_ENV=alphagenome \
./scripts/run_alphagenome_multi_gpu.sh --gpus 0 \
  --tasks ipaqtl \
  --max-variants 10 \
  --model-config configs/alphagenome_local.json \
  --checkpoint-path /data/yitian_workspace/DNA/alphagenome_models/all_folds \
  --model-name alphagenome-all-folds \
  --data-dir data \
  --output-dir outputs/my_alphagenome_ipaqtl
```

单卡和多卡使用同一个入口。以下命令都可以从项目根目录直接复制运行。

Borzoi 单卡：

```bash
BORZOI_CONDA_ENV=borzoi_py310 \
./scripts/run_borzoi_multi_gpu.sh --gpus 0 \
  --tasks sqtl \
  --max-variants 10 \
  --variant-batch-size 1 \
  --model models/replicate_0/model0_best.h5 \
  --model-name borzoi-replicate-0 \
  --conda-env borzoi_py310 \
  --data-dir data \
  --output-dir outputs/my_borzoi_sqtl
```

Borzoi 8卡、四个 published replicate：

```bash
BORZOI_CONDA_ENV=borzoi_py310 \
BORZOI_GPU_IDS=0,1,2,3,4,5,6,7 \
./scripts/run_borzoi_multi_gpu.sh \
  --tasks sqtl \
  --max-variants 2000 \
  --variant-batch-size 2 \
  --model-dir models \
  --model-name borzoi-four-replicates \
  --conda-env borzoi_py310 \
  --data-dir data \
  --output-dir outputs/my_borzoi_sqtl_four_replicates
```

AlphaGenome 单卡：

```bash
ALPHAGENOME_CONDA_ENV=alphagenome \
./scripts/run_alphagenome_multi_gpu.sh --gpus 0 \
  --tasks sqtl \
  --max-variants 10 \
  --variant-batch-size 1 \
  --model-config configs/alphagenome_local.json \
  --checkpoint-path /data/yitian_workspace/DNA/alphagenome_models/all_folds \
  --model-name alphagenome-all-folds \
  --data-dir data \
  --output-dir outputs/my_alphagenome_sqtl
```

AlphaGenome 四卡：

```bash
ALPHAGENOME_CONDA_ENV=alphagenome \
./scripts/run_alphagenome_multi_gpu.sh \
  --gpus 0,1,2,3,4,5,6,7 \
  --tasks sqtl \
  --max-variants 2000 \
  --variant-batch-size 2 \
  --model-config configs/alphagenome_local.json \
  --checkpoint-path /data/yitian_workspace/DNA/alphagenome_models/all_folds \
  --model-name alphagenome-all-folds \
  --data-dir data \
  --output-dir outputs/my_alphagenome_sqtl

ALPHAGENOME_CONDA_ENV=alphagenome \
./scripts/run_alphagenome_multi_gpu.sh \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-variants 2000 \
  --variant-batch-size 2 \
  --model-config configs/alphagenome_local.json \
  --checkpoint-path /data/yitian_workspace/DNA/alphagenome_models/all_folds \
  --model-name alphagenome-all-folds \
  --data-dir data \
  --output-dir outputs/my_alphagenome_sqtl
```

NTv3 使用独立的 `ntv3` 环境。先接受 Hugging Face 模型条款并下载三种参数规模的 QTL
尝试套件（8M pre、100M post、650M post）：

```bash
./scripts/setup_ntv3_benchmark.sh ntv3
conda run -n ntv3 hf auth login
conda run -n ntv3 hf auth whoami
NTV3_CONDA_ENV=ntv3 \
./scripts/download_ntv3_models.sh --suite qtl
```

NTv3 单卡跑通四类任务：

```bash
NTV3_CONDA_ENV=ntv3 \
./scripts/run_ntv3_multi_gpu.sh --gpus 0 \
  --tasks all \
  --max-variants 1 \
  --batch-size 1 \
  --model-config configs/ntv3_100m_post.json \
  --data-dir data \
  --output-dir outputs/ntv3_100m_post_smoke
```

NTv3 四卡按 variant 分片：

```bash
NTV3_CONDA_ENV=ntv3 NTV3_GPU_IDS=0,1,2,3 \
./scripts/run_ntv3_multi_gpu.sh \
  --tasks all \
  --max-variants 10 \
  --batch-size 1 \
  --model-config configs/ntv3_650m_post.json \
  --data-dir data \
  --output-dir outputs/ntv3_650m_post_all_qtl
```

NTv3 100M post-training 的零参数正式入口使用 0–7 号 GPU、四类 QTL、每任务
limit 2000，并把 context 固定为与 DNA-FM 一致的 131,072 bp：

```bash
./scripts/all_ntv3_qtl.sh
```

NTv3 官方没有这四类 QTL 的专用 scorer。post 模型使用 task-related functional/BED heads 的
`ALT-REF`，pre 模型使用 masked-base `log P(ALT)-log P(REF)` fallback。正式 post 配置的
eQTL 使用 checkpoint 内 GTEx RNA tracks 并复用 Borzoi broad-tissue 映射；两个 Heart tissues
显式回退到 ENCODE heart tracks。详细 head、参数规模、显存和科学解释边界必须以
[`docs/ntv3_qtl.md`](docs/ntv3_qtl.md) 为准。

本地 PyTorch NTv3 post-training 模型以 `DNA-FM` 名称接入。它复用 AlphaGenome 的数据与
统一 evaluator，eQTL 优先使用 checkpoint 内的 86 条 GTEx RNA tracks（Heart 组织
显式回退到 ENCODE RNA tracks），其他任务使用 annotation heads 的 `ALT-REF`。单卡
smoke test：

```bash
./scripts/run_dna_fm_multi_gpu.sh --gpus 0 \
  --tasks all --max-variants 1 --batch-size 1 \
  --model-config configs/dna_fm_100m_post_step_0034000.json \
  --output-dir outputs/dna_fm_smoke
```

模型重建、checkpoint 固定、tissue 映射及分数定义见
[`docs/dna_fm_qtl.md`](docs/dna_fm_qtl.md)。

当前 step340000 DNA-FM 配置默认使用 FP32，与官方 NTv3 post 的默认推理精度一致；BF16
仅作为显式的速度/显存优化选项，不用于默认跨模型比较。

与 `all_alphagenome_qtl.sh` 对齐的 DNA-FM 零参数一键脚本会使用 0–7 号 GPU，依次跑
eQTL、sQTL、paQTL 和 iPaQTL，每个任务的 limit 为 2000：

```bash
./scripts/all_dna_fm_qtl.sh
```

多卡模式是一张卡一个模型进程，但各卡加载的是同一个 Borzoi replicate，并负责不同的
variant 数据分片；全部 shard 合并后才进入下一个 replicate。AlphaGenome 同样按 variant
分片并在全部 shard 完成后统一评估。参数、资源占用和断点续跑说明见
[`docs/gpu_execution.md`](docs/gpu_execution.md)。

`--tasks` 接受 `eqtl`、`sqtl`、`paqtl`、`ipaqtl`、逗号分隔列表或 `all`。
`--max-variants 0` 保持全量语义。Borzoi eQTL 用 `--eqtl-limit-scope global|per-tissue`
显式选择限额范围；正式的 `all_borzoi_qtl.sh` 使用 `per-tissue`，与 AlphaGenome 的每 tissue、
每 split 上限一致，旧 `global` 模式保留给快速实验。sQTL/paQTL/iPaQTL 会先选最多 N 个
不同 negative，再加入这些 negative 的 `PI` 所指向的 positive；每个 negative 都保留配对对象，
但多个 negative 可以共用同一个 positive，所以去重后的 positive 数量不保证等于 N。

`--variant-batch-size N` 同时可用于 Borzoi 和 AlphaGenome，但两者实现不同。Borzoi 会把 N 个
变异组成一次真正的 TensorFlow 前向，即 `2*N` 条相邻的 REF/ALT 序列；默认 `1`，也可设置
`BORZOI_VARIANT_BATCH_SIZE`。Borzoi 当前硬上限是 `3`，`4` 会让第一层卷积达到 `2^31` 个元素并
触发 GPU launch 计数溢出。AlphaGenome 使用本地模型公开的 `score_variants` 方法，以 N 个线程
并发执行独立的本地 JAX `score_variant` 调用，并不是一个堆叠 tensor batch；默认同样是 `1`，
建议确认显存余量后再尝试 `2`。两者都与多卡数据分片独立，不改变预测、缓存或评价口径。

Borzoi 的 task 设置与原脚本一致：sQTL=`nDi/span`、paQTL=`COVR/utr3`、
iPaQTL=`COVR`。Borzoi、AlphaGenome、NTv3 和以后接入统一 `BenchmarkRunner` 的模型都会在
`<output-dir>/logs/` 的运行日志中输出完整指标：`[metrics]` 行记录列名和结果文件，随后每个
tissue 或 distance cutoff 各有一条不截断的 `[metric]` JSON 行。该 JSON 的 `values` 包含
`metrics.tsv` 当前行的全部字段，包括 AUROC/AUPRC、Spearman/Pearson、标准差、样本数、
命中计数和 eligibility；未定义的指标写为 `null`。机器可读结果同时提供 TSV 和 JSON：

```text
<output-dir>/unified/<all|max_N>/<model-name>/<task>-<all|max_N>/metrics.tsv
<output-dir>/unified/<all|max_N>/<model-name>/<task>-<all|max_N>/metrics.json
<alpha-output-dir>/<all|max_N>/<model-name>/<task>-<all|max_N>/metrics.tsv
<alpha-output-dir>/<all|max_N>/<model-name>/<task>-<all|max_N>/metrics.json
```

当前 evaluator 不报告需要人为分类阈值的普通 accuracy：sQTL/paQTL/iPaQTL 使用阈值无关的
AUROC/AUPRC，eQTL 使用方向 AUROC、因果 AUROC、Spearman 和 Pearson。这样不会为不同模型
任意指定不公平的 score cutoff；日志会完整打印 evaluator 实际计算出的所有指标。

只准备输入、不加载模型：

```bash
./scripts/run_borzoi_multi_gpu.sh --gpus 0 --tasks sqtl --max-variants 10 --prepare-only
```

### 数据规模、正负配对和染色体覆盖

以下数字来自当前 `data/` 中的官方全量数据，不包含 `outputs/` 下由 `--max-variants` 生成的
裁剪副本。计数时将 VCF 的非 header 行视为记录；“tissue 出现次数”会重复计算在多个 tissue
出现的同一 variant，“merged 数量”则是跨 tissue 独立去重后真正送入模型推理的 variant 数。

#### 文件构成和磁盘占用

四类 QTL 都包含 49 个 tissue。每个 tissue 各有 positive/negative VCF，再加
`pos_merge.vcf.gz` 和 `neg_merge.vcf.gz`，所以每类都有 100 个 benchmark VCF。

| task | tissue 数 | benchmark VCF 数 | 额外文件 | 目录大小 |
|---|---:|---:|---:|---:|
| eQTL | 49 | 100 | 49 个 SuSiE TSV + 49 个 table VCF | 约 952 MB |
| sQTL | 49 | 100 | 0 | 约 1.8 MB |
| paQTL | 49 | 100 | 0 | 约 684 KB |
| iPaQTL | 49 | 100 | 0 | 约 636 KB |

eQTL 体积明显更大，主要来自 `tables/` 中的完整 SuSiE variant-gene 表，而不是 benchmark
正负 VCF 本身。

#### 全量记录数

每个 tissue 内的 positive 和 negative 数量严格相等；下表中的 tissue 正/负总数是 49 个
tissue 的记录数之和。merged VCF 分别对 positive 和 negative 做跨 tissue 去重，因此不再要求
两边数量相等。

| task | tissue positive 总出现次数 | tissue negative 总出现次数 | merged unique positive | merged unique negative | 每 tissue positive：最小 / 中位数 / 最大 |
|---|---:|---:|---:|---:|---:|
| eQTL | 48,961 | 48,961 | 17,925 | 41,563 | 54 / 776 / 2,740 |
| sQTL | 21,514 | 21,514 | 4,217 | 16,396 | 54 / 376 / 1,047 |
| paQTL | 6,493 | 6,493 | 1,128 | 4,314 | 13 / 117 / 312 |
| iPaQTL | 5,218 | 5,218 | 1,077 | 3,843 | 9 / 90 / 262 |

merged VCF 中每个 task 的 variant ID 和 `(chrom, pos, ref, alt)` 都没有重复。正例在不同
tissue 之间的共享程度远高于负例，这也是 merged positive 明显少于 merged negative 的主要原因：

| task | positive tissue 出现次数 / merged | negative tissue 出现次数 / merged |
|---|---:|---:|
| eQTL | 2.73 | 1.18 |
| sQTL | 5.10 | 1.31 |
| paQTL | 5.76 | 1.51 |
| iPaQTL | 4.84 | 1.36 |

换句话说，正负样本是在每个 tissue 内匹配的；merged 文件的职责是避免同一个序列重复推理，
不是一个要求全局类别平衡的评价表。模型先对 merged variant 计算一次，再由上游 fold launcher
拆回各 tissue 进行评价。

#### 为什么 merged negative 和 positive 数量不一致

原因分为三层：

1. 同一个 positive QTL 往往在多个 tissue 复用，跨 tissue 去重后只保留一条；negative control
   的跨 tissue 重用较少，因此 merged negative 更多。
2. sQTL、paQTL、iPaQTL 的每条 negative 都通过 INFO 字段中的 `PI` 指向配对 positive。
   tissue 内是一对一，但合并后可能有多个不同 negative 指向同一个 positive ID，所以
   “negative 数量”等于 N 不意味着“unique positive 数量”也等于 N。
3. eQTL 标签本身是 tissue-specific：有 1,279 个 variant 会在某个 tissue 是 positive、在另一个
   tissue 是 negative。因此它们会同时出现在全局 `pos_merge` 和 `neg_merge`；同一个 tissue 内
   仍然不存在 positive/negative 重叠。

数据完整性检查结果如下：

- 四类 QTL 的 49 个 tissue 全部满足 positive 数量等于 negative 数量；
- 每个 tissue 内没有重复 positive、重复 negative 或正负坐标重叠；
- sQTL、paQTL、iPaQTL 的每条 negative `PI` 都能在同 tissue positive VCF 中找到；
- tissue 内 `PI` 是一对一的，merged 层面的多对一来自跨 tissue 合并去重；
- benchmark merged VCF 中的记录全部是双等位 SNV。

#### `--max-variants` 裁剪后的实际正负数量

非 eQTL 当前按 merged negative 的顺序选择 N 个不同 negative，然后把它们的 `PI` 放入一个
positive ID 集合。集合会去重，所以 N 较大时会出现 unique positive 少于 negative 的情况：

| task | N=10（pos/neg） | N=100（pos/neg） | N=1000（pos/neg） | N=2000（pos/neg） |
|---|---:|---:|---:|---:|
| sQTL | 10 / 10 | 100 / 100 | 1,000 / 1,000 | 1,371 / 2,000 |
| paQTL | 10 / 10 | 92 / 100 | 494 / 1,000 | 722 / 2,000 |
| iPaQTL | 10 / 10 | 100 / 100 | 522 / 1,000 | 787 / 2,000 |

因此日志中的 `Prepared variants: positive=1371 negative=2000` 并不是文件缺失，而是 2,000 个
sQTL negative 在 merged 层面只对应 1,371 个不同 positive。每个 negative 的配对 positive
仍然都包含在裁剪结果中。需要“unique positive 也严格等于 N”的一对一子集时，必须改成在选择
negative 时同时禁止重复 `PI`；当前实现见
[`benchmark/qtl_benchmark/subset_vcfs.py`](benchmark/qtl_benchmark/subset_vcfs.py)。

#### 是否覆盖全染色体

是，但这里的“全染色体”指当前人类 benchmark 使用的 `chr1`–`chr22` 和 `chrX`：

- eQTL、sQTL、paQTL、iPaQTL 的 merged positive 和 merged negative 都覆盖全部 23 条染色体；
- 当前数据没有 `chrY` 和线粒体 `chrM`；
- benchmark merged VCF 不按染色体过滤，多卡也只是按 VCF 记录序号分 shard，不是一张卡负责
  一条染色体；
- `--max-variants` 小子集按记录选择，因此很小的 smoke test 不保证仍出现全部 23 条染色体；
- 当前统一指标跨所有选中 variant 计算，没有额外输出逐染色体 AUROC/AUPRC。

每类数据量最少的 tissue 都是 Kidney Cortex：eQTL/sQTL/paQTL/iPaQTL 分别为
54、54、13、9 对。数据量最多的 tissue 分别为 Nerve Tibial（eQTL，2,740 对）、
Thyroid（sQTL，1,047 对）、Nerve Tibial（paQTL，312 对）和 Nerve Tibial
（iPaQTL，262 对）。

#### eQTL SuSiE 表和 VCF 格式说明

49 份 eQTL SuSiE TSV 共包含 12,312,896 条 variant-gene 记录；每 tissue 中位数为 222,056，
最少是 Kidney Cortex 的 27,298 条，最多是 Thyroid 的 574,899 条。对应的 49 个 table VCF
共有 9,959,140 条 variant 记录。SuSiE 原始表中可以包含 indel，但进入当前 benchmark
positive/negative merged VCF 的记录都是 SNV。

上游生成的 100 个 eQTL benchmark VCF 只有 `##fileformat` 元信息，没有标准 `#CHROM` header；
Borzoi 当前的逐行读取器可以正常处理。sQTL、paQTL 和 iPaQTL 的 300 个 VCF 都有完整 INFO 定义
和 `#CHROM` header。若要对 eQTL 文件使用要求严格 VCF header 的外部工具（例如部分
`bcftools` 子命令），应先补齐 header；这不影响仓库现有 benchmark。

#### eQTL REF/ALT 为什么会与 hg38 相反

这不是 hg38 FASTA 模板选错，也不是坐标版本或 liftOver 错误。eQTL benchmark VCF 有意把
ALT 统一为 SuSiE 表中的 `minor_allele`，而没有要求第 4 列 REF 始终等于 hg38 reference allele。
对 49 个 tissue 的 97,922 条 positive/negative 记录与 SuSiE 表进行全量交叉检查后：

- 97,922 / 97,922 条均满足 `VCF ALT == minor_allele`；
- 76,404 次（78.03%）VCF REF 匹配 hg38；
- 21,518 次（21.97%）VCF ALT 匹配 hg38，这些记录的 REF/ALT 全部是 variant ID 所编码等位
  基因的精确交换；
- 没有任何 eQTL 记录出现 hg38 同时不匹配 REF 和 ALT；
- sQTL、paQTL、iPaQTL 的 30,975 条 merged 记录全部由 VCF REF 匹配 hg38。

因此 21,518 条并非可忽略的少量坏记录，跳过会系统性删除约 22% 的 eQTL 数据。Borzoi 原始
`snp_seq1` 以及本项目 NTv3 adapter 都会保留 benchmark 的 minor-allele 方向：当 FASTA 匹配
VCF ALT 时，先构造 VCF REF background，再计算 VCF `ALT-REF`。完整、可复现的审计命令是：

```bash
conda run -n ntv3 python scripts/audit_qtl_references.py
```

汇总位于 `reports/qtl_reference_audit/summary.json`。逐条清单位于
`reports/qtl_reference_audit/exceptions.tsv`，其中包含相对/绝对文件路径、文件内 0-based record
index、1-based record number、实际文件行号、染色体坐标、variant ID、VCF REF/ALT、FASTA 碱基，
以及 VCF 等位基因相对 variant ID 是同向还是精确交换。

### 四类 QTL 的数据、模型分数和统一评价口径

本节描述正式对比时真正进入统一框架的数据与公式。需要先区分三层对象：

1. `variants` 是按 `chrom:pos:REF:ALT` 去重后需要模型推理的序列输入；
2. `memberships` 记录同一 variant 属于哪个 tissue 的 positive/negative 集合，同一 variant 可以有
   多条 membership；
3. `targets` 是只提供给 evaluator 的标签，例如 eQTL 的 gene、effect size 和 effect allele。

这里的 `positive` / `negative` 不是医学意义上的“患病/健康”，也不是病人分组。它们是针对
variant effect 的 benchmark 标签：

- eQTL 中，`positive` 表示该 tissue 的 SuSiE/QTL 数据把这个 variant 标为有因果信号的一侧，
  `negative` 是配套的非因果/对照 variant。eQTL 的 causal effect size、effect allele 和
  positive/negative membership 是分开的信息；`positive` 不等于 effect size 必须为正，effect
  的正负方向由 `effect_size > 0` 单独用于 `auroc_sign`。
- sQTL、paQTL 和 iPaQTL 中，`positive` 表示真实 QTL variant，`negative` 表示与之匹配的
  negative control。每条 negative 的 VCF INFO `PI` 指向它配对的 positive variant；配对依据是
  统计设计中的 matched variant 关系，不是临床病例对照关系。

因此统一 evaluator 的分类 AUROC 问题是“模型能否区分 QTL positive 与 matched negative”，而不是
“模型能否区分疾病与非疾病”。方向相关的 eQTL 指标另外问“预测效应方向是否与 coefficient 一致”。

#### 当前四类 QTL 是否都是单核苷酸变异

QTL 名称描述的是变异关联的**分子表型**，不是变异本身的类型：

| task | 本 benchmark 所问的问题 |
|---|---|
| eQTL | 变异是否影响目标 gene 的表达量，以及表达增加/降低方向 |
| sQTL | 变异是否影响 RNA 剪接模式或 splice junction 使用 |
| paQTL | 变异是否影响 3' 端 alternative polyadenylation / PAS 使用 |
| iPaQTL | 变异是否影响 intronic polyadenylation / intronic PAS 使用 |

从一般生物学定义看，这四类 QTL 都可能由 SNV、indel 或其他变异造成，并不天然限定为 SNP。
但对本仓库 `data/{eqtl,sqtl,paqtl,ipaqtl}` 下当前打包的全部 positive/negative、逐 tissue 和
merged VCF 做实际审计后，每条记录都满足：

```text
len(REF) == 1
len(ALT) == 1
REF, ALT ∈ {A, C, G, T}
```

因此**本次四模型 benchmark 实际测的全部都是双等位 SNV（通常也称 SNP）**；当前数据中没有
insertion、deletion、MNV 或 multi-allelic 记录。这里的结论是当前数据集属性，不应扩展解释为
模型只能预测 SNV，也不应扩展解释为这四类 QTL 在真实人群中只有 SNV。

模型 adapter 只能看到推理输入，不会看到 positive/negative 标签；所有模型的结果最终都转换为
统一长表 `(variant_key, gene_id, tissue, score, n_tracks)`，再交给同一个 evaluator。因而“公共
可比”指测试样本、标签、配对规则和 metric 实现相同，并不要求 AlphaGenome 和 Borzoi 使用相同
的生物学 scorer；两者的原生 scorer 本来就是模型能力的一部分。

#### 当前 `max-variants=2000` 的实际数据量

下表对应 `all_alphagenome_qtl.sh` 和当前 `all_borzoi_qtl.sh` 的正式设置。eQTL 使用
`per-tissue`，其余三类使用 matched-pairs：

| task | 2000 的准确含义 | selected positive | selected negative | 去重后推理 variants | 统一评价标签 |
|---|---|---:|---:|---:|---|
| eQTL | 每 tissue、每 split 最多 2000 | 17,248 个 split-unique | 39,780 个 split-unique | 56,024 | 93,258 memberships；50,761 个规范化 causal targets |
| sQTL | 前 2000 个 negative 加其 `PI` 配对 positive | 1,371 | 2,000 | 3,371 | 1,371 个可用 unique positive ID 与对应 negative |
| paQTL | 同上 | 722 | 2,000 | 2,722 | 722 个可用 unique positive ID 与对应 negative |
| iPaQTL | 同上 | 787 | 2,000 | 2,787 | 787 个可用 unique positive ID 与对应 negative |

eQTL 的 49 个 tissues 各有 46,629 条 positive 和 46,629 条 negative membership；同一基因组
variant 可以出现在多个 tissues，也可以在不同 tissue 分别充当 positive 和 negative。17,248 与
39,780 相加得到 57,028 个 split records，但跨 split 再去重后是 56,024 个模型输入。这就是
AlphaGenome 每卡显示 `7003 × 8 = 56024`，而 Borzoi negative shard 约为 4,973、随后还有
positive shard 约为 2,156 的原因。

sQTL/paQTL/iPaQTL 的 `2000` 不是“正负各取 2000 条互不重复记录”。流程先选 2,000 个不同
negative，再根据每条 negative 的 `PI` 找 positive；多个 negative 可以跨 tissue 指向同一个
positive，所以 positive ID 去重后分别只剩 1,371、722、787。统一 evaluator 随后会再强制恢复
一对一配对，不会把没有配对对象的记录加入 AUROC/AUPRC。全量数据量见前面的“全量记录数”表。

#### `pos_merge` / `neg_merge` 到底如何 merge

当前打包的 merge 文件实际等价于对 49 个 tissue 文件按 split 分别求并集，再在各自 split 内按
`variant_id` 去重：

```text
所有 <tissue>_pos.vcf ──并集、pos 内去重──> pos_merge.vcf
所有 <tissue>_neg.vcf ──并集、neg 内去重──> neg_merge.vcf
```

positive 和 negative 是两条独立 merge 路线；不会在 merge 时把两边放在一起裁决一个“全局标签”，
也不会因为某个 ID 已经出现在 `pos_merge` 就自动从 `neg_merge` 删除。每个 merged ID 保留一条实际
来自 source tissue VCF 的代表记录。对于 sQTL/paQTL/iPaQTL，如果同一 ID 在多个 tissue 的
`MT`、transcript/group 或 `PI` 不同，merged VCF 也只能保留其中一条代表 INFO；这是上游
merged benchmark 的既定语义，所有模型在公共比较中读取同一份 merged 文件。

逐 tissue 原始文件被刻意构造成平衡集合，因此每个 tissue 都有
`n_positive == n_negative`。但同一个真实 QTL positive 常在多个 tissues 重复，negative controls
通常更分散，所以 split 内去重后 positive 缩减得更多：

| task | merge 前 pos/neg 行数 | pos_merge | neg_merge |
|---|---:|---:|---:|
| eQTL | 48,961 / 48,961 | 17,925 | 41,563 |
| sQTL | 21,514 / 21,514 | 4,217 | 16,396 |
| paQTL | 6,493 / 6,493 | 1,128 | 4,314 |
| iPaQTL | 5,218 / 5,218 | 1,077 | 3,843 |

这些是全量 source VCF 的文件级计数，不是前表正式 `max-variants=2000` subset 的计数。

eQTL 的标签本身是 tissue-specific。同一个 SNV 完全可以在 tissue A 有显著/因果表达关联而在
tissue B 被放入 negative control；当前全量 eQTL 中有 1,290 个 variant ID 同时出现在某个 tissue
的 positive 和另一个 tissue 的 negative，但逐 tissue 检查时 positive/negative 位点交集均为 0。
统一框架不会把它压成矛盾的全局标签，而是保留两条 membership：

```text
(variant X, tissue A, positive)
(variant X, tissue B, negative)
```

`GTExEQTLEvaluator` 对每个 tissue 独立取该 tissue 的预测并计算指标，所以 X 在 A 与 B 的角色互不
覆盖。eQTL unified data/evaluation 直接保留逐 tissue memberships；merged VCF 主要让昂贵的模型
推理对跨 tissue 重复变异复用结果。可视化中的 broad organ 是对已经算好的 tissue metrics 做宏平均，
也不会把不同 tissue 的原始正负标签先混在一起。

sQTL、paQTL 和 iPaQTL 的当前全量数据则没有任何 variant ID 同时跨 split 出现在 positive 与
negative。它们的 unified data module 读取 merged VCF，并在距离过滤后使用 negative 的代表
`PI` 指回一个 merged positive，最后重新形成严格一对一评价集合。因此 merge 后文件数量不相等，
不代表最终 AUROC/AUPRC 使用不平衡类别；重采样阶段仍从两类各取相同数量。

只要两边使用同一 `data/`、相同 N 和相同 scope，AlphaGenome 与 Borzoi 的 benchmark-level
variant/label 数据就是同一份。eQTL 比较器还会检查 variant 覆盖、membership hash 和 target
hash，任一模型缺失 variant 都会停止。模型实际张量仍有固有差异：AlphaGenome 使用
1,048,576 bp，Borzoi 使用 524,288 bp；这不属于数据集划分差异。

#### eQTL：表达方向、效应相关性和因果分类

模型分数的实现不同：

- AlphaGenome 使用推荐的 `RNA_SEQ` / `GeneMaskLFCScorer`。对每个 gene exon mask 和 RNA-seq
  track 计算
  `log(mean(ALT) + 1e-3) - log(mean(REF) + 1e-3)`，然后 adapter 对相同
  `(variant, gene, GTEx tissue)` 的 tracks 求均值。它已经是 log fold-change，不再加 Borzoi 的
  `arcsinh`。
- Borzoi 在目标 gene 的 bins 上先求和，原生 `SED = sum(ALT) - sum(REF)`；统一 adapter 选择
  tissue keyword 对应的 GTEx tracks，先求 track 均值，再使用 `arcsinh(SED)`。`logSED =
  log2(sum(ALT)+1)-log2(sum(REF)+1)` 及各 replicate 的原生指标仍保留，但公共 evaluator 默认读取
  ensemble/single-run 的 `SED`。
- NTv3 100M/650M post 使用官方 checkpoint 中的 GTEx RNA BigWig heads，按与 Borzoi 相同的
  broad keyword 分组；DNA-FM 使用本地 post-training checkpoint 中对应的同源 tracks。两者都在
  gene span、对应 tissue tracks 和输出位置上对 `ALT-REF` 取 signed max-abs。官方
  checkpoint 缺少 GTEx heart heads，两者的 Heart 均显式回退到 ENCODE heart RNA tracks。
- AlphaGenome 有精确 GTEx tissue metadata；Borzoi target 是 `brain`、`blood_vessel` 等较宽的
  keyword；NTv3 post 和 DNA-FM 也是这个 broad 粒度。正式横向比较的主视图把
  AlphaGenome 向 Borzoi broad group 聚合，并同时保留 AlphaGenome native 细粒度视图；不能从
  broad 输出反向恢复精确 brain tissue。

四个模型在四类 QTL 上的完整实现差异见
[`docs/qtl_model_implementation_comparison.md`](docs/qtl_model_implementation_comparison.md)。

`GTExEQTLEvaluator` 对每个 tissue 独立计算以下字段：

| 字段 | 具体计算 |
|---|---|
| `auroc_sign` | 仅 causal `(variant,gene,tissue)`；标签为 `effect_size > 0`，预测为按 effect allele 对齐符号后的 score |
| `spearmanr` | causal effect size 与对齐后 score 的秩相关 |
| `pearsonr` | causal effect size 与对齐后 score 的线性相关 |
| `auroc_class` | membership 的 pos/neg 标签；每个 `(variant,tissue)` 跨 genes 取 `max(abs(score))` 后算 AUROC |
| `n` | 该 tissue 的 causal target 行数 |
| `n_score_found` | causal target 中实际找到预测的行数；未找到的 coefficient score 填 0 |
| `n_class_pos/neg` | 找到至少一个可用 gene score 后参加分类的正/负例数 |
| `n_class_score_found` | 分类阶段实际有分数的总数 |
| `eligible` | `n > 32`；否则四个主要指标写 NaN，防止误读极小样本 |

方向对齐完全由公共 evaluator 完成：内部 `effect_allele` 字段实际读取 SuSiE 表的 `allele1`；
当 benchmark VCF REF 不等于 `allele1` 时，ALT−REF score 乘以 -1，否则保持不变。这个过程只
统一等位基因方向，不缩放或校准模型分数。分类只关心效应强度，因此使用绝对值。完整的
causal effect size 与 aligned score 定义见
[`docs/qtl_visualization.md`](docs/qtl_visualization.md#什么是-causal-effect-size)。推荐比较输出是
`common_metrics.tsv` 中的 `alphagenome_borzoi_grouped` 对 `borzoi_native`；Borzoi 原脚本单独
打印的多组 `Sign AUROC/SpearmanR/Class AUROC` 属于原生复现输出，不是另一批公共 metric。

#### sQTL：剪接 junction 变化

两种模型都输出非方向性的剪接扰动强度，但中间统计量不同：

- AlphaGenome 使用 `SPLICE_JUNCTIONS` / `SpliceJunctionScorer`。先对 junction pair count 做
  `log(x + 1e-7)`，计算 `abs(log(ALT+1e-7)-log(REF+1e-7))`，只保留落在同链目标 gene 内的
  junction；推荐 scorer 为每个 track 找最大变化 junction。adapter 对同一 gene/tissue 的 tidy
  rows 求均值，公共 evaluator 再为 VCF `MT` 目标 gene 选择绝对值最大的 tissue score。
- Borzoi 使用论文路径的 `nDi/span`。先在 gene span 内把 REF、ALT profile 分别加 pseudocount 并
  归一化为总和 1，再计算各位置 `abs(REF_norm-ALT_norm)`，`nDi` 取位置最大值；adapter 对 Borzoi
  output tracks 求均值。这里比较的是预测剪接 profile 形状变化，不是 AlphaGenome junction
  log-count 公式，所以原始 score 数值尺度不能直接比较。

统一 evaluator 不比较 score 的绝对尺度，而是在每个距离阈值上评价 positive 是否比其 matched
negative 得到更大的 `abs(score)`。sQTL 阈值依次为 10,000、2,000、500、200、100、50 bp。

#### paQTL：3' UTR polyadenylation

- AlphaGenome 使用 `POLYADENYLATION` / `PolyadenylationScorer` 和 RNA-seq 输出。在每个 PAS
  上游 400 bp 聚合 REF/ALT coverage，形成各 PAS 的 ALT/REF ratio；枚举 proximal/distal PAS
  切分，取最大的绝对 log2 coverage fold-change。当前同一个推荐 scorer 同时用于 paQTL 和
  iPaQTL，没有按 task 再切换 scorer 类。
- Borzoi 使用 `COVR/utr3`，先把 PolyADB 限制到 `3' most exon` 且保留从 distal 端连续的 sites。
  对每个 PAS 周围的 Borzoi bins 计算带 pseudocount 的 ALT/REF coverage ratio，枚举上下游切分并
  取最大的 `abs(log2(mean_downstream/mean_upstream))`；正式默认 `cov_pseudo=50`、
  `cov_min=100`。adapter 再对 output tracks 求均值。

两者都在测 alternative polyadenylation 的强度，但 PAS mask 宽度、site 过滤、pseudocount 和模型
输出轨道不同，因此 scorer 不完全同式。公共 evaluator 的距离阈值为 10,000、2,000、500、300、
200、100、50 bp。

#### iPaQTL：intronic polyadenylation

- AlphaGenome 仍使用上面的 `POLYADENYLATION` 推荐 scorer；本地 adapter 当前没有额外的
  intron-only mask。iPaQTL 与 paQTL 的区别首先来自输入 variant/target gene 数据集，而不是换成
  另一套 AlphaGenome scorer。
- Borzoi 使用独立的 `borzoi_sed_ipaqtl_cov.py` 和 `COVR`。它保留 PolyADB 的 intronic 与
  3' UTR sites，通过独立的 iPaQTL 数据和 gene/PAS 映射计算 coverage-ratio change；不像 paQTL
  正式命令那样启用 `--utr3` 只保留 3' most exon。

iPaQTL 使用与 paQTL 相同的公共距离阈值和 matched evaluator。这个实现差异必须保留在解释中：
公共指标算法相同，并不代表两种模型的 PAS 定义完全相同。

#### matched QTL 中的 `match distance` 是什么

图中的 `match distance` / `max_distance` 不是 positive 与 negative 之间的基因组距离，也不是
模型输入 context，更不是两条记录允许相差多远的配对容差。它是 source VCF 已为**每条 variant
自身**记录的、variant 到相关分子事件的距离，单位为 bp：

- sQTL 从 INFO `SD` 读取 splice distance，即 variant 到相关 splice event/site 的距离；
- paQTL 和 iPaQTL 从 INFO `PD` 读取 PAS distance，即 variant 到相关 polyadenylation site 的距离；
- INFO `PI` 才负责指定一条 negative 应与哪条 positive 配对。

例如 `max_distance=500` 时，evaluator 先分别保留 `positive.distance <= 500` 和
`negative.distance <= 500` 的记录，再根据 negative 的 `PI` 重建严格 positive/negative pair。
因此只有 pair 的双方都通过 500 bp cutoff 时，这一对才参加该行指标。cutoff 是累积子集：

```text
50 bp ⊂ 100 bp ⊂ 200 bp ⊂ 500 bp ⊂ 2 kb ⊂ 10 kb
```

paQTL/iPaQTL 另外报告 300 bp，sQTL 不含 300 bp。分多个 cutoff 的目的是观察模型区分 QTL 与
matched control 的能力是否随 variant 远离 splice/PAS event 而变化，同时确保正负例采用相同的
距离过滤。图中误差线来自固定 seed 下 100 次、每次 80% 配对样本重采样的指标标准差，不是
distance 的测量误差。

#### sQTL/paQTL/iPaQTL 的公共 metric 公式

三类任务共同使用 `MatchedQTLEvaluator`：

1. 优先按 `(variant_key, VCF MT gene)` 找预测；同一目标 gene 有多条 tissue/track 结果时取
   `abs(score)` 最大的那条，同时保留原 score。目标 gene 没有结果时，当前默认允许退回到该
   variant 所有预测中绝对值最大的 score，并在明细表标记 `score_source=variant`。
2. 去掉完全没有 score 的 variants，并在每个 `max_distance` 下同时过滤 positive 和 negative。
3. 根据 negative 的 `PI` 重新组成严格一对一 pair；同一 positive 被多个 negatives 引用时只保留
   第一对。
4. 令 `m = min(n_positive, n_negative)`，每次从两类各无放回抽取
   `floor(0.8*m)` 个样本，以 `abs(score)` 为预测值，计算 AUROC 和 average precision（AUPRC）。
5. 使用固定 seed 44 重复 100 次，报告 `auroc_mean/std`、`auprc_mean/std`、
   `sample_per_class`、`matched_positive/negative` 和 `scored_positive/negative`。

由于每次抽样正负数量相等，AUPRC 的随机基线约为 0.5。这里没有人为 score cutoff，因此不输出
accuracy。AUROC/AUPRC 的代码对所有模型完全相同；模型之间不同的仅是上游如何从 REF/ALT 输出
构造 score。

Borzoi 论文原生的 sQTL/paQTL/iPaQTL `stats.txt` 仍然保留：当 HDF5 有多个 output targets 时，
它使用这些 targets 作为多维特征训练随机森林并做 8-fold classification；这与上面把结果归一成
单个公共 score 后进行 matched resampling 的 evaluator 不同。原生结果用于复现 Borzoi，跨模型
比较应使用统一目录中的 `metrics.tsv/metrics.json`，不要拿原生随机森林 AUROC 直接对比
AlphaGenome 的公共 AUROC。

NTv3 没有针对这四类 QTL 的官方 scorer，属于额外的零样本迁移基线；其 head 选择与公式单独记录
在 [`docs/ntv3_qtl.md`](docs/ntv3_qtl.md)，不应把它的模型分数解释成 AlphaGenome 或 Borzoi
原生 statistic。

## 1. 创建环境

需要先有 Conda 或 Mamba：

```bash
cd /data/yitian_workspace/DNA/benchmarks/qtl_Borzoi
./scripts/setup_environment.sh
```

默认环境名是 `borzoi_py310`（与原 `borzoi-paper` 脚本一致）。自定义环境名：

```bash
./scripts/setup_environment.sh my_qtl_env
```

源码已经放在本目录中，不需要重新 clone Borzoi 或 Baskerville。脚本通过 `PYTHONPATH` 直接加载 `src/` 下的代码。

## 2. 检查文件、环境和 GPU

不依赖 Python 环境的文件检查：

```bash
./scripts/verify_package.py
```

完整 SHA256 校验会再次读取约 8 GB 文件，因此更慢：

```bash
./scripts/verify_package.py --deep
```

在 GPU 计算节点检查 Conda 依赖和 TensorFlow GPU：

```bash
./scripts/run_eqtl_benchmark.sh --check
```

登录节点没有 GPU 时，文件检查仍可通过，但 `--check` 会明确报告没有 TensorFlow GPU；正式推理应在 GPU 节点运行。

## 3. 一键运行

论文的 gene-aware eQTL SED benchmark：

```bash
./scripts/run_eqtl_benchmark.sh
```

上面的默认行为是：

- 使用全部 eQTL（`--max-variants 0`）；
- 使用四个模型 replicate 并进行 ensemble；
- 运行 `SED` 和 `logSED`；
- 自动计算每个 GTEx tissue 的方向 AUROC、因果分类 AUROC、Spearman 和 Pearson；
- 把结果写入 `outputs/borzoi_eqtl/`。

同时运行论文中的 SED 与 SAD 两条 eQTL 路线：

```bash
./scripts/run_eqtl_benchmark.sh --modes all
```

仅运行 SAD：

```bash
./scripts/run_eqtl_benchmark.sh --modes sad
```

推荐先用一个模型和小数据做端到端 smoke test：

```bash
./scripts/run_eqtl_benchmark.sh \
  --modes sed \
  --max-variants 10 \
  --model models/replicate_0/model0_best.h5 \
  --output-dir outputs/borzoi_smoke
```

`--max-variants N` 表示从全部 tissue 中轮询选取全局最多 N 个唯一正例、N 个唯一负例，
并把 SuSiE coefficient 表同步裁剪到这些正例。它不是“每个 tissue 各 N 个”，因此可精确控制
smoke test 的推理量，也不会把没有推理的全量变异当成 0 分参与评价。小子集主要验证
“数据读取 → 模型推理 → HDF5 汇总 → 评价脚本”是否连通。SED 的 tissue 评价默认要求每个
tissue 超过 32 个阳性变异，所以很小的子集可能产生空的 tissue 指标；正式结果必须使用
`--max-variants 0`。

如果之前只想准备解压后的 VCF：

```bash
./scripts/run_eqtl_benchmark.sh --prepare-only --max-variants 100
```

裁剪后的数据和清单位于
`outputs/borzoi_eqtl/inputs/max_100/eqtl_pip90/`；其中 `subset_manifest.json` 记录实际正负
例数、每个 tissue 的记录数和 coefficient 表行数。目录名保留 `eqtl_pip90` 是因为论文原始
coefficient 脚本会从目录名读取 PIP 阈值。

## 4. 结果位置

全量 SED ensemble 的主要结果：

```text
outputs/borzoi_eqtl/experiment/all/ensemble/eqtl_sed/coef-SED/metrics.tsv
outputs/borzoi_eqtl/experiment/all/ensemble/eqtl_sed/coef-logSED/metrics.tsv
```

`metrics.tsv` 包含：

- `auroc_sign`：预测效应方向与 GTEx coefficient 符号的一致性；
- `spearmanr`：预测分数与 coefficient 的 Spearman 相关；
- `pearsonr`：预测分数与 coefficient 的 Pearson 相关；
- `auroc_class`：阳性 eQTL 与匹配阴性变异的分类 AUROC；
- `n`：该 tissue 参与方向/相关性评价的变异数。

SAD 的 coefficient 指标同样位于 `ensemble/eqtl_sad/coef-*/metrics.tsv`；因果分类的输出以 `stats.txt` 保存。启动脚本结束时会打印发现的全部 `metrics.tsv` 和 `stats.txt`。

## 5. 更换模型

指定一个模型：

```bash
./scripts/run_eqtl_benchmark.sh \
  --model models/replicate_0/model0_best.h5 \
  --conda-env borzoi_py310 \
  --data-dir data/eqtl \
  --output-dir outputs/my_borzoi_eqtl_replicate_0
```

多次使用 `--model` 可以组成 ensemble：

```bash
./scripts/run_eqtl_benchmark.sh \
  --model models/replicate_0/model0_best.h5 \
  --model models/replicate_1/model0_best.h5 \
  --conda-env borzoi_py310 \
  --data-dir data/eqtl \
  --output-dir outputs/my_borzoi_eqtl_ensemble_0_1
```

模型必须兼容本包的 `configs/params_pred.json` 和 target 索引。如果模型结构或输出 target 不同，应同时提供对应 `--params`、`--targets-gtex` 或 `--targets-human`。

## 6. 兼容标量接口

下面的命令保留给旧版“每个 variant 一个标量”流程。新代码应使用前面的统一框架；
自定义标量模型继承 `QTLModel` 后，也仍可先只产生预测：

```bash
PYTHONPATH=benchmark:src python -m qtl_benchmark.predict_model \
  --model-class my_models.eqtl:MyModel \
  --model-config my_model.json \
  --tasks eqtl \
  --data-dir data \
  --output-dir outputs/custom_models
```

再单独评价：

```bash
PYTHONPATH=benchmark:src python -m qtl_benchmark.evaluate_predictions \
  --prediction-dir outputs/custom_models/<model-name> \
  --tasks eqtl \
  --data-dir data
```

这个兼容接口只评价标准化的标量因果/非因果分数。统一接口保留
`variant-gene-tissue` 维度，并由任务对应的 Evaluator 统一计算指标。

## 7. 常用参数

```text
--modes sed|sad|sed,sad|all  选择 eQTL 路线，默认 sed
--max-variants N             全局正/负集合各最多 N 个变异；0 表示全量，默认 0
--variant-batch-size N        每次前向的变异数（实际为 2*N 条 REF/ALT 序列），默认 1
--model FILE                 指定模型，可重复
--model-dir DIR              在目录中寻找 model0_best.h5/model_best.h5
--params FILE                模型参数 JSON
--conda-env NAME             Conda 环境名，默认 borzoi_py310
--data-dir DIR               eQTL 数据目录
--output-dir DIR             输出目录
--prepare-only               只准备 VCF
--check                      检查依赖、文件和 GPU，不运行推理
```

不要让不同模型共用同一个 `--output-dir`，否则脚本会阻止把不同模型的中间结果混在一起。

## 8. 用本地 AlphaGenome 跑 Borzoi eQTL benchmark

这里不是把合并 VCF 简化成“每个 variant 一个分数”。实现直接对照
`src/borzoi_scripts/borzoi_sed.py` 和 `borzoi_gtex_coef_sed.py`，保留 Borzoi 评价需要的
`(variant, gene, tissue)` 关系：

```text
每个 tissue 的 pos/neg VCF + tables/<tissue>.tsv.gz
                        │
                        ▼
AlphaGenomeModel.score_variant_table
（本地 checkpoint；RNA_SEQ gene-mask LFC）
                        │
                        ▼
同一 gene/tissue 的 AlphaGenome tracks 求均值
                        │
                        ▼
predictions.sqlite（每个 REF/ALT 只推理一次，可断点续跑）
                        │
          ┌─────────────┴──────────────┐
          ▼                            ▼
variant-gene coefficient          variant max-|score|
方向 AUROC / Spearman / Pearson    正负样本 AUROC
```

模型接口与 benchmark 仍然分离：

- `benchmark/qtl_benchmark/models/alphagenome.py` 只负责加载模型和产生完整 gene/track 表；
- `benchmark/qtl_benchmark/data/gtex_eqtl.py` 负责数据读取、筛选和划分；
- `benchmark/qtl_benchmark/interfaces.py` 与 `predictions.py` 负责聚合和通用缓存；
- `benchmark/qtl_benchmark/evaluation/gtex_eqtl.py` 负责统一指标；
- `benchmark/qtl_benchmark/alphagenome_eqtl.py` 仅保留旧命令兼容入口；
- `scripts/run_alphagenome_local.sh` 只是本地环境包装入口。

方向处理与 Borzoi 原脚本一致：预测原本是 ALT-REF；当 VCF REF 与 GTEx 表中的 `allele1`
不一致时翻转符号。分类分数则先在同 tissue 内对重复 tracks 求均值，再跨 genes 取最大
绝对值。AlphaGenome 已输出 gene-level log fold change，因此不再套 Borzoi SED 的
`arcsinh` 变换。

### 8.1 下载和检查模型权重

官方权重来自 Kaggle 或 Hugging Face，二者都要先在网页接受 AlphaGenome 的非商业模型
条款。下载脚本只下载 checkpoint，不会顺便初始化 JAX 模型。

Hugging Face：

```bash
conda run -n alphagenome hf auth login
conda run -n alphagenome python scripts/download_alphagenome_weights.py \
  --source huggingface \
  --output-dir /data/yitian_workspace/DNA/alphagenome_models/all_folds
```

Kaggle：

```bash
# 先配置 ~/.kaggle/kaggle.json 或 KAGGLE_API_TOKEN
conda run -n alphagenome python scripts/download_alphagenome_weights.py \
  --source kaggle \
  --output-dir /data/yitian_workspace/DNA/alphagenome_models/all_folds
```

当前本地 checkpoint 可以离线检查：

```bash
conda run -n alphagenome python scripts/download_alphagenome_weights.py \
  --check-only \
  --output-dir /data/yitian_workspace/DNA/alphagenome_models/all_folds
```

脚本检查 Orbax 的 `_CHECKPOINT_METADATA`、`_METADATA` 和 `manifest.ocdbt`，并打印总大小。

### 8.2 环境和参考文件

默认配置对应：

```text
AlphaGenome 源码：  /data/yitian_workspace/DNA/alphagenome_research
all_folds 权重：    /data/yitian_workspace/DNA/alphagenome_models/all_folds
Conda 环境：        alphagenome
```

检查 JAX GPU：

```bash
env -u LD_LIBRARY_PATH conda run -n alphagenome python -c \
  'import jax; print(jax.devices("gpu"))'
```

若环境还没有 GPU JAX wheel：

```bash
conda run -n alphagenome python -m pip install 'jax[cuda12]==0.11.1'
```

AlphaGenome 的 gene scorer 读取 Feather GTF，variant 推理还需要 splice-site starts/ends
Feather；paQTL/iPaQTL 还需要 PolyADB Feather。转换脚本会一次生成这些本地文件：

```bash
conda run -n alphagenome python scripts/prepare_alphagenome_reference.py
```

输出为 `reference/alphagenome/gencode41_basic_nort.gtf.feather`、两个
`gencode41_basic_nort.splice_sites_*.feather` 和 `polyadb_human_v3.feather`。

### 8.3 最小本地测试

默认用 `Liver` 的 1 个正例和 1 个负例、`2^17` 输入长度。选择 Liver 是为了让首个
causal gene 完整落在 128 kb 窗口内，从而让 smoke 同时覆盖 coefficient join：

```bash
./scripts/run_alphagenome_local.sh
```

首次运行会加载 Orbax checkpoint 并触发 XLA 编译，明显慢于后续变异。输出为：

```text
outputs/alphagenome_eqtl/alphagenome-local-smoke/borzoi_eqtl/
├── scores.sqlite
├── benchmark_run.json
├── model_run.json
├── metrics.tsv
└── predictions/
    ├── Liver_coefficients.tsv.gz
    └── Liver_classification.tsv.gz
```

smoke 样本少于原脚本的 32 条阈值，`eligible=false` 且指标为 NaN 是预期行为；它验证的是
“本地权重 → GPU → gene/tissue 分数 → Borzoi 评价表”整条链路。

XLA 首次编译还需要较多主存，不只是 GPU 显存。启动脚本默认要求容器至少有 10 GiB 可用
主存，并会在不足时提前提示；VS Code/Pylance 可能单独占用数 GiB。确认资源足够时可设置
`ALPHAGENOME_MIN_FREE_GIB=0` 跳过这项预检。

### 8.4 正式运行与断点续跑

`configs/alphagenome_smoke.json` 使用 `2^17`；正式配置
`configs/alphagenome_local.json` 使用 AlphaGenome 完整的 `2^20`（1 Mb）输入。

先测一个 tissue 的 10+10 个变异：

```bash
./scripts/run_alphagenome_local.sh \
  --model-config configs/alphagenome_local.json \
  --tissue Whole_Blood \
  --max-variants 10
```

所有 49 个 tissues、全量数据：

```bash
./scripts/run_alphagenome_local.sh \
  --model-config configs/alphagenome_local.json \
  --all-tissues \
  --max-variants 0
```

缓存键包含 chromosome、position、REF 和 ALT。一个变异出现在多个 tissues 时只推理一次；
中断后原命令重跑会跳过 `scores.sqlite` 中已完成的变异。需要重算当前命令选中的变异时加
`--force-rescore`。

也可分开执行：

```bash
# 只推理并更新缓存
./scripts/run_alphagenome_local.sh --tissue Liver --max-variants 100 --score-only

# 不加载模型，只用缓存重算评价
./scripts/run_alphagenome_local.sh --tissue Liver --max-variants 100 --evaluate-only
```

正式全量共有 17,925 个合并正例和 41,563 个合并负例，1 Mb 推理耗时会很长，应该先从
小子集确认显存和主存余量。`scores.sqlite` 不保存庞大的逐 track 原始张量，只保存每个
gene/tissue 的 track 均值。

### 8.5 配置项

JSON 中主要字段：

- `research_source`：`alphagenome_research` checkout；
- `checkpoint_path`：本地 Orbax checkpoint；
- `fasta_path`、`gtf_feather_path`：本地 hg38 资源；
- `sequence_length`：`2^17`、`2^18`、`2^19` 或 `2^20`；
- `device`：JAX GPU 序号；
- `gtex_tissues`、`ontology_terms`：可选 track 过滤，完整 benchmark 通常留空。

相对路径以配置中的 `root_dir` 为基准。包装脚本会切换到本包根目录。

原来的通用 `QTLModel.predict_batch` 标量入口仍然保留，可用于跨模型的 variant-level
AUROC/AUPRC；但它不再被描述为 Borzoi 的 tissue-aware coefficient benchmark。

逐步的数据连接、聚合公式、缓存 schema 和与 Borzoi 原脚本的对应关系见
`docs/alphagenome_eqtl.md`。
