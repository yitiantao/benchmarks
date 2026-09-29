# Borzoi eQTL benchmark 实现说明

## 保留的论文实现

模型推理和指标计算仍由 `src/borzoi_scripts/` 中的论文原脚本完成，没有重新实现 Borzoi
打分公式：

- `borzoi_bench_gtex_folds_sed.py` / `borzoi_bench_gtex_folds_sad.py`：创建任务、按 tissue
  拆分结果、做 replicate ensemble，并启动评价；
- `borzoi_sed.py` / `borzoi_sad.py`：加载 `model0_best.h5`，提取 hg38 序列并执行
  REF/ALT 推理；
- `borzoi_gtex_coef_sed.py` / `borzoi_gtex_coef_sad.py`：计算方向 AUROC、相关系数和正负例
  分类 AUROC；
- `configs/params_pred.json`、target 表、hg38 FASTA/GTF 和四个 published replicate 都由本包
  固定提供。

默认仍启用论文路径的 reverse complement（`--rc`）和 untransform（`-u`）。

## 本地运行层

入口是 `scripts/run_eqtl_benchmark.sh`。论文 launcher 原来通过 Slurm 提交子任务；本包的
`benchmark/qtl_benchmark/slurm.py` 保留同样的 `Job`/`multi_run` 接口，但在当前计算节点上
逐个运行任务。逐个运行是有意的：每个 TensorFlow 子进程独占 GPU，避免多个模型进程同时
占满显存。

`benchmark/qtl_benchmark/run_borzoi_script.py` 只提供两项兼容处理：

1. 把上游生成的脚本名转发到打包后的 `src/borzoi_scripts/`；
2. 当 coefficient 脚本请求 `.tsv` 时，透明读取包内的 `.tsv.gz` SuSiE 表。

数据流如下：

```text
压缩 tissue/merged VCF + SuSiE tables
              │ subset_vcfs.py
              ▼
inputs/max_N/eqtl_pip90
              │ 原版 borzoi_sed.py / borzoi_sad.py
              ▼
每个 replicate 的 REF/ALT HDF5
              │ 原版 fold launcher
              ▼
replicate ensemble + tissue 拆分
              │ 原版 coefficient scripts
              ▼
metrics.tsv / stats.txt
```

输入目录必须以 `eqtl_pip90` 结尾，因为论文 coefficient 脚本会从目录名解析 PIP 阈值。

## 精确控制测试数据量

```bash
./scripts/run_eqtl_benchmark.sh \
  --modes sed \
  --max-variants 10 \
  --model models/replicate_0/model0_best.h5 \
  --output-dir outputs/borzoi_smoke
```

`--max-variants N` 必须与 `--limit-scope` 一起解释。默认 `global` 在所有 tissue 间轮询，分别
选择最多 N 个唯一正例和 N 个唯一负例；`N=10` 最多推理 20 个变异，适合 smoke test。
`per-tissue` 则保留每个 tissue、每个 split 的前 N 条记录，与 AlphaGenome 的限额语义一致：

```bash
./scripts/run_eqtl_benchmark.sh \
  --max-variants 2000 --limit-scope per-tissue
```

这个“全局上限”最初用于控制四 replicate、reverse-complement Borzoi 推理的总成本，并非模型
或 eQTL 指标要求。AlphaGenome 历史入口的同名参数采用“每 tissue、每 split 上限”，因此两个
原生 run 的相同 `max_N` 标签不代表相同样本集合。正式跨模型比较应使用共享 subset；后续参数
scope 的统一实现见 [`eqtl_model_comparison.md`](eqtl_model_comparison.md)。两种 scope 分别写入
`max_N` 和 `per_tissue_max_N`，防止旧 global HDF5 被正式 per-tissue run 误用。

有限子集下，SuSiE 表也只保留被选中的正例。否则原评价脚本会把未推理的全量因果变异补成
0 分，使小样本指标失真。实际选择结果写在：

```text
outputs/.../inputs/max_N/eqtl_pip90/subset_manifest.json
```

`--max-variants 0` 表示不裁剪 VCF，并直接使用原始全量 SuSiE 表，是正式复现模式。

## 建议的运行顺序

```bash
# 0. 一次性创建/更新环境（以下两个路径也是脚本针对本机布局的默认值）
./scripts/setup_qtl_benchmark.sh \
  --borzoi-dir /data/yitian_workspace/DNA/borzoi \
  --borzoi-paper-dir /data/yitian_workspace/DNA/borzoi-paper \
  --no-download-assets

# 1. 静态文件、依赖和 TensorFlow GPU
./scripts/run_eqtl_benchmark.sh --check

# 2. 只准备 10+10 个输入，不加载模型
./scripts/run_eqtl_benchmark.sh --prepare-only --max-variants 10

# 3. 单模型 smoke test
./scripts/run_eqtl_benchmark.sh \
  --modes sed --max-variants 10 --variant-batch-size 2 \
  --model models/replicate_0/model0_best.h5 \
  --output-dir outputs/borzoi_smoke

# 4. 四模型、全量 SED benchmark
./scripts/run_eqtl_benchmark.sh --modes sed --max-variants 0
```

`setup_qtl_benchmark.sh` 以当前 benchmark 仓库为根目录读取 `environment.yml`，默认环境名为
`borzoi_py310`，优先使用 `mamba`、否则使用 `conda`。若需要保持环境不变而只做检查，可运行
`./scripts/setup_qtl_benchmark.sh --check`。路径也可通过 `BORZOI_DIR`、
`BORZOI_PAPER_DIR`、`BASKERVILLE_DIR` 和 `BORZOI_CONDA_ENV` 环境变量覆盖。

`pybedtools` 由 Bioconda 安装而不是 pip 源码构建。若一次环境更新曾在 pip 阶段失败，可直接
重新执行同一条 setup 命令；Conda 会复用已完成的包并补齐剩余依赖，不需要删除环境。

Baskerville 和 Borzoi 的 editable 安装使用环境中固定的 `setuptools_scm` 8.x，并设置各自的
pretend version。安装命令带有 `--no-deps`，所有运行依赖由前一步 `environment.yml` 统一安装；
其中 `pyranges=0.0.129` 使用 Bioconda 预编译包，避免 pip 在本机用 Cython 构建
`sorted-nearest`。这样在计算节点没有 `git` 命令、或 pip 镜像提供了不兼容的最新隔离构建
依赖时，也不会影响本地源码安装；pretend version 只用于包 metadata，不影响模型代码或权重。

NumPy、Pandas、SciPy、scikit-learn、h5py 等含二进制扩展的科学计算包全部由 Conda 安装，
避免 Conda/PyPI 多次中断更新后留下同名包的多套 metadata 和不兼容 ABI。如果已有环境已经出现
`numpy.core.multiarray failed to import`，应使用新的环境名重新 setup；仅执行 `pip check` 不能
发现这种文件级混装。

相同命令和输出目录可以续跑：原 fold launcher 会复用已经写完的 `sed.h5`/`sad.h5` 和
`metrics.tsv`。如果更换模型，应使用新的 `--output-dir`；启动脚本也会检查 staged model
软链接，阻止不同模型结果混用。

`--variant-batch-size N` 表示一次模型调用处理 N 个变异，即 `2*N` 条 REF/ALT 序列；默认
为 `1`。SED 和 SAD 都支持该参数，单卡和多卡入口会继续向下传递。实现细节和显存建议见
[`implementation_details.md`](implementation_details.md)。

具体地，`--variant-batch-size 1` 的基础 sequence batch 是同一个 variant 的
`[REF, ALT]` 两条 524,288 bp 序列，而不是只计算一个 allele。默认 `--rc` 还会在模型图内部对
每条序列计算 forward/reverse-complement 并取平均，最终 SED 方向仍是 ALT−REF。

小于 33 个 tissue 阳性样本时，原 coefficient 评价门槛会令相应指标为 `NaN`。这不影响
smoke test 判断推理/HDF5/评价链路是否跑通，但不能作为模型效果结论。

## 2026-09-19：全 QTL 启动与跨模型一致比较

`scripts/all_borzoi_qtl.sh` 现在用一条 `--tasks all` 调用依次完成 eQTL、sQTL、paQTL 和
iPaQTL，默认使用 8 张 GPU、单个 `replicate_0`、`max-variants=2000` 和
`variant-batch-size=2`；eQTL 显式使用 `per-tissue` scope，与 AlphaGenome 输入统计一致。可用
`BORZOI_ALL_QTL_MODEL`、`BORZOI_ALL_QTL_MODEL_NAME` 和 `BORZOI_ALL_QTL_MAX_VARIANTS` 分别覆盖
checkpoint、结果名称和样本数；`BORZOI_ALL_QTL_OUTPUT_DIR` 可覆盖默认独立输出根目录
`outputs/borzoi_replicate_0_all_qtl`。若显式把 `BORZOI_ALL_QTL_MODEL` 改成其他单个 checkpoint，仍然
只运行一个模型；需要四模型 ensemble 时应直接调用通用 runner 并传 `--model-dir models`。统一评价读取 eQTL
裁剪目录，若目录中没有 tissue VCF 会给出明确错误，不再只显示 Pandas 的
`No objects to concatenate`。

当相同 tag 的 AlphaGenome eQTL 缓存存在时，该脚本最后会自动运行
`scripts/compare_alphagenome_borzoi_eqtl.sh`。它保留 Borzoi 原生 replicate、ensemble、SED 和
logSED 结果，同时额外生成三组同数据、同 evaluator 的可比结果。可设置
`BORZOI_COMPARE_EQTL=0` 跳过。详见
[`eqtl_model_comparison.md`](eqtl_model_comparison.md)。
