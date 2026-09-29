# NTv3 benchmark 数据准备工作记录

## 并行版目录说明

本目录是 `NTv3_scripts/` 的完整独立副本；原目录保持已验证的单进程实现。仅本目录的
`prepare_tensorstore.py` 和 `prepare_annotations.py` 增加 `--workers`/`--prefetch`。
Functional 由 worker 并行读 block、主进程顺序写 store 并更新断点；annotation 也改为
worker 只栅格化有界 block、主进程顺序写 store。不要让多个 annotation worker 直接向
网络文件系统的同一 Zarr 目录写几十万个小 chunk。README 记录了内存/文件句柄约束、
运行命令和一致性测试。

## 项目目标

原始数据位于：

```text
/data/yitian_workspace/DNA/benchmarks/NTv3_benchmark_dataset
```

本次工作将它整理成可用于 NTv3 风格训练、验证和测试的数据管线，目标包括：

- 严格沿用官方 `splits.bed` 的 train/val/test 基因组区域；
- 固定验证和测试坐标，保证不同模型之间可重复比较；
- 避免复制多份 FASTA 和 BigWig；
- 使用 TensorStore 和 Grain 支持大规模 functional tracks；
- 区分真实零信号与 BigWig 未覆盖位置；
- 为新增物种、模态和轨道预留扩展空间。

## 原始数据含义

```text
NTv3_benchmark_dataset/<species>/
├── genome.fasta
├── splits.bed
├── functional_tracks/*.bigwig
└── genome_annotation/*.bed
```

- `genome.fasta`：参考基因组 DNA 序列。
- `splits.bed`：官方 train/val/test 区域，使用 0-based、半开坐标。
- `functional_tracks/*.bigwig`：实验信号轨道，不是 DNA 序列。
- `genome_annotation/*.bed`：exon、intron、start codon 等离散注释。
- `benchmark_metadata.tsv`：track ID、物种、assembly、mean、std 和 assay。

BigWig 通常包含同一参考组装的多条染色体。多个 BigWig 是同一基因组坐标上的不同实验通道，不是首尾拼接成参考基因组。

当前 functional-track 数量：

| 物种 | BigWig 数量 |
|---|---:|
| human | 34 |
| tomato | 20 |
| chicken | 14 |
| rice | 14 |
| maize | 8 |
| arabidopsis | 4 |
| cattle | 0 |

Cattle 目前只有 genome annotation，没有 functional BigWig。

## 官方划分和采样结论

### Split 是区域，不是固定样本清单

官方发布了 train/val/test 基因组范围，但没有发布唯一的逐窗口训练或测试坐标表。

公开 fine-tuning notebook 是简化实现：先等概率选择一行 split region，再在其中随机选择窗口。示例使用 32,768 bp 输入、1,000 个 validation 样本和 10,000 个 test 样本。这不等同于论文完整 post-training 采样。

### 论文 post-training 采样

论文 4.3.3 `Genomic Sequence Selection` 规定：

- train 使用 `0.1% × sequence_length` 的 stride；
- validation/test 使用等于 `sequence_length` 的 stride；
- 训练时让同一碱基出现在多个上下文中；
- validation/test 让每个位置只评估一次。

对于 32,768 bp 输入：

```text
train stride ≈ 33 bp
validation stride = 32,768 bp
test stride = 32,768 bp
```

训练窗口数量非常大，因此没有生成巨大的 train Parquet。`NTv3WindowSource` 根据样本 index 实时计算窗口。

当前配置为：

```text
input length = 32,768 bp
keep_target_center_fraction = 0.375
target length = 12,288 bp
left context = 10,240 bp
right context = 10,240 bp
```

整个输入窗口必须完整落入一个官方 split region，不能只检查中央 target。

### Post-training 长度扩展

NTv3 post-training 不是全程直接使用 1 Mb：

1. 第一阶段约 1.2T tokens，混合 16、32、65、131 kb；131 kb 权重约 70%。
2. 第二阶段约 54B tokens，混合 16、32、65、131、262、524 kb 和 1 Mb；1 Mb 权重约 40%，其他长度各约 10%。

当前管线先实现 32 kb benchmark，但 `window_size` 可配置，后续可以增加混合长度调度。

## 已实现的 TensorStore

转换脚本：`prepare_tensorstore.py`。

数据布局按照论文：

```text
每个物种一个 TensorStore
signal dtype = float16
compressor = zstd level 3
chunk shape = 8192 bp × 该物种全部 tracks
```

输出结构：

```text
NTv3_prepared/tensorstore/<species>/
├── signal/
├── coverage/
└── metadata.json
```

实现细节：

- 染色体沿 TensorStore 第一维拼接；
- 每条染色体起点按 8192 bp 对齐；
- `metadata.json` 保存染色体长度和 global offset；
- track 顺序保存在 metadata 中；
- `signal` 缺失位置填 0；
- `coverage` 保存 uint8 mask，区分真实 0 与缺失；
- 转换进度写入 `progress.json`，支持断点续跑；
- `--species` 支持逐物种转换；
- `--overwrite` 用于重新构建；
- `--limit-bp` 只用于 smoke test。

原始值超过 float16 最大有限值 65,504 时会显式 clip，并在 metadata 中记录 `float16_overflow_values_clipped`，避免静默产生 `inf`。

## 已实现的 Manifest

构建脚本：`build_manifests.py`。

正式论文模式命令：

```bash
python NTv3_scripts_parallel/build_manifests.py \
  NTv3_benchmark_dataset \
  NTv3_prepared/manifests/32k_paper \
  --window-size 32768 \
  --target-fraction 0.375 \
  --paper-eval
```

已生成目录：

```text
NTv3_prepared/manifests/32k_paper/
├── official_regions.parquet
├── manifest.json
├── human.val.parquet
├── human.test.parquet
└── 其他物种的 val/test Parquet
```

固定窗口数量：

| 物种 | Validation | Test |
|---|---:|---:|
| arabidopsis | 181 | 211 |
| cattle | 1,672 | 1,682 |
| chicken | 1,572 | 1,571 |
| human | 10,534 | 10,531 |
| maize | 1,777 | 1,777 |
| rice | 604 | 600 |
| tomato | 1,568 | 1,568 |

Parquet 字段包括：

```text
sample_id, species, assembly, split, chrom,
input_start, input_end, target_start, target_end,
region_id, sampling_policy, seed
```

不带 `--paper-eval` 时可以生成与公开 tutorial 类似的固定随机 val/test 数据，并支持 `official_region_uniform` 和 `coordinate_uniform` 两种随机策略。

## 已实现的 Grain loader

加载代码：`ntv3_data.py`。

主要接口：

```python
NTv3WindowSource
make_grain_loader
```

训练示例：

```python
from NTv3_scripts.ntv3_data import NTv3WindowSource, make_grain_loader

source = NTv3WindowSource(
    "NTv3_prepared/tensorstore",
    "human",
    regions="NTv3_prepared/manifests/32k_paper/official_regions.parquet",
    split="train",
    window_size=32_768,
    target_fraction=0.375,
    sampling_policy="paper_stride",
    assay="ATAC-seq",
    normalize=True,
    use_mask=True,
)

loader = make_grain_loader(
    source,
    batch_size=2,
    shuffle=True,
    seed=0,
    workers=8,
)
```

固定测试集：

```python
test_source = NTv3WindowSource(
    "NTv3_prepared/tensorstore",
    "human",
    manifest="NTv3_prepared/manifests/32k_paper/human.test.parquet",
    assay="ATAC-seq",
)
```

单条样本返回：

```python
{
    "sample_id": str,
    "species": str,
    "chrom": str,
    "input_start": int,
    "input_end": int,
    "sequence": str,
    "targets": float32[target_bp, tracks],
    "target_mask": bool[target_bp, tracks],
    "track_ids": str[tracks],
}
```

TensorStore 和 FASTA 使用 worker-local lazy handles，打开的文件句柄不会被序列化给 Grain worker。

## Functional track scaling

loader 按论文 4.3.4 实现：

```python
scaled = targets / track_mean

if RNA-seq:
    scaled = scaled ** 0.75

scaled = where(
    scaled > 10,
    2 * sqrt(scaled * 10) - 10,
    scaled,
)
```

coverage mask 应用于 loss 和评价指标，避免将没有数据的位置当作真实零信号。

## 已完成验证

已经使用 Arabidopsis 完成真实小规模端到端测试：

```text
BigWig
→ float16/zstd TensorStore
→ 染色体/global 坐标映射
→ indexed FASTA
→ 32,768 bp sequence
→ 12,288 × 4 targets
→ coverage mask
→ target scaling
→ Grain 2-worker multiprocessing
→ batch
```

测试结果：

```text
sequence length = 32768
targets shape = (12288, 4)
target_mask shape = (12288, 4)
Grain batch targets shape = (1, 12288, 4)
```

所有 `NTv3_scripts_parallel/*.py` 已通过 Python 语法检查。

## 已安装依赖

当前环境已安装：

```text
tensorstore 0.1.85
grain 0.2.18
pyfaidx 0.9.0.4
pyarrow 25.0.1
pyBigWig
numpy
```

依赖声明见 `requirements.txt`。

## 尚未执行

没有启动完整约 42 GB 原始 benchmark 的全量 TensorStore 转换。目前仅在 `/tmp` 中做了小规模转换验证，正式 manifest 已经生成。

建议先逐物种执行并观察实际磁盘占用：

```bash
python NTv3_scripts_parallel/prepare_tensorstore.py \
  NTv3_benchmark_dataset NTv3_prepared/tensorstore \
  --species arabidopsis

python NTv3_scripts_parallel/prepare_tensorstore.py \
  NTv3_benchmark_dataset NTv3_prepared/tensorstore \
  --species human
```

之后再转换全部 functional-track 物种：

```bash
python NTv3_scripts_parallel/prepare_tensorstore.py \
  NTv3_benchmark_dataset NTv3_prepared/tensorstore
```

## 后续接手注意事项

1. 不要修改原始 `splits.bed`，它是官方 split 来源。
2. 不要复制 train/test FASTA 或 BigWig；隔离应通过坐标实现。
3. 固定 val/test manifest 后不要在不同模型间重新生成。
4. 整个输入窗口必须落入 split region，不能只检查中央 target。
5. 新增轨道后要重建对应物种 TensorStore，因为 chunk 第二维覆盖全部 tracks。
6. track 顺序必须使用 `metadata.json`，不能依赖文件系统遍历顺序。
7. 不同物种的 track 数不同，多物种训练应按物种组成 batch。
8. 论文物种权重为 human 45%、其他 functional species 各 5%、annotation-only species 各 1%；当前尚未实现跨物种 batch scheduler。
9. Annotation 已由 `prepare_annotations.py` 转成 uint8 多标签 TensorStore；cattle 可用 `load_functional=False, load_annotations=True` 加载。
10. 两类转换器默认只写 `splits.bed` 引用的 contig；不要无意中加入 `--all-contigs`，否则会产生大量无用碎片。
11. 完整复现论文还需要混合长度调度、跨物种权重以及 annotation/MLM 联合 loss。

## Genome annotation 实现

- `prepare_annotations.py` 读取每个物种的 `genome_annotation/*.bed`，生成每物种一个
  `uint8 [genome_bp, annotation_elements]` TensorStore；压缩、8192 bp 分块和拼接基因组
  坐标布局与 functional store 保持一致。
- annotation 是逐碱基、逐元素的独立二元标签，因此 exon、intron、splice acceptor、
  start codon 可以在同一碱基重叠；没有 BED 命中的位置写为 0（背景），不存在 BigWig
  那种 missing-value coverage mask。
- 原始 BED 保留 strand，但公开模型输出按 annotation element 设二分类 logits、没有单独
  strand 维；转换器因此把正负链折叠为同一个 element 标签。
- `ntv3_data.py` 支持 annotation-only 和 functional+annotation 联合读取，分别适用于
  cattle，以及 human/tomato 等同时具备两类监督的数据。
- 论文方法部分对 annotation fine-tuning/model selection 使用加权 focal loss
  (`gamma=2.0`)，指标为 MCC；数据加载器只负责返回未经损失函数变换的 `uint8` 标签。
- TensorStore 的磁盘数组、metadata JSON 字段以及 `NTv3WindowSource` 返回 record 的完整
  schema 已记录在 `README.md` 的 `On-disk TensorStore schema` 一节；后续更改字段时需要
  同步更新该节。
- README 的 `What the manifests are for` 解释了 manifest 只保存坐标和实验 provenance，
  不复制序列或监督数组；`official_regions.parquet` 用于动态 train，物种级 val/test
  Parquet 用于固定且可复现的评估，并以完整 input window 落入单一 split 来避免泄漏。
- Functional store 不做跨物种统一列宽：每物种数组为
  `[该物种 padded genome bp, 该物种实际 track 数]`，模态由 `metadata.json` 中每列的
  `assay` 标注。缺失模态不创建零列；已有 track 的局部坐标缺失才由同 shape 的
  `coverage=0` 表达。因此默认应使用物种内同构 batch，跨物种训练由 scheduler 交替
  species-homogeneous batches；若将来必须混 batch，需要额外的全局列 schema 和有效列 mask。
- Loader 现在随 `targets` 返回同序的 `track_ids`、`track_assays`、`track_indices`；其中
  `targets[:, j]` 的模态就是 `track_assays[j]`，原始 store 列号是 `track_indices[j]`。
- 修复 cattle annotation 空 store：其 split/BED 使用 `chr1`，FASTA 使用 `1`。转换器现以
  split 名作为公开 `name`，保存真实 `fasta_name`，支持保守的 `chr` 前缀别名；loader
  用 `fasta_name` 取序列。零 contig、零 BED 匹配及已有 `shape[0]==0` 均直接报错。
- 转换器现统一按 FASTA 顺序拼接染色体。Loader 对 joint functional+annotation 分别使用
  两份 metadata 的 offset，只强制公共 chromosome 名称/size 一致，因此也能安全读取
  tomato 这类旧 store 中 `1,2,...` 与 `1,10,11,2,...` 的不同物理排列。

## 相关文件

- `README.md`：用户使用说明；
- `prepare_tensorstore.py`：BigWig 转换器；
- `prepare_annotations.py`：BED annotation 转换器；
- `build_manifests.py`：split 和 manifest 构建；
- `ntv3_data.py`：TensorStore + Grain loader；
- `check_prepared_data.py`：验证工具；
- `requirements.txt`：Python 依赖；
- `../NTv3_prepared/manifests/32k_paper/`：已生成的固定 val/test 坐标。
