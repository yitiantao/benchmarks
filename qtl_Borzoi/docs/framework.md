# QTL benchmark 统一框架

## 设计目标

框架把一次 benchmark 明确拆成三个彼此独立的组件：

```text
DataModule                 ModelAdapter                 Evaluator
读取/筛选/划分数据  ─────▶  只产生标准预测长表  ─────▶  只读取标签并计算指标
        │                         │                         │
        └── BenchmarkDataset      └── PredictionStore      └── metrics.tsv
```

统一调度由 `BenchmarkRunner` 完成。Runner 只向模型传递不含 label 的 `InferenceDataset`，
Evaluator 也不加载
TensorFlow/JAX 模型。这样一份测试集可喂给 AlphaGenome、Borzoi 或未来模型，最终指标由同一份
代码产生。

## 目录职责

```text
benchmark/qtl_benchmark/
├── core.py                    # 数据集、运行上下文、预测 schema
├── data/
│   ├── base.py                # DataModule 接口
│   ├── gtex_eqtl.py           # GTEx VCF + SuSiE label
│   └── matched_qtl.py         # sQTL/paQTL/iPaQTL 配对、距离与基因
├── models/
│   ├── alphagenome.py         # AlphaGenome 模型实现
│   ├── borzoi.py              # Borzoi HDF5 -> 标准预测 adapter
│   └── precomputed.py         # 标量预测兼容 adapter
├── evaluation/
│   ├── gtex_eqtl.py           # tissue-aware eQTL 指标
│   └── matched_qtl.py         # 匹配式 AUROC/AUPRC 与距离阈值
├── interfaces.py              # ModelAdapter / Evaluator 接口
├── predictions.py             # 通用、可恢复的 SQLite 预测仓库
├── runner.py                  # 唯一编排入口
├── config.py                  # JSON 组件装配
└── cli.py                     # 统一 CLI
```

`alphagenome_eqtl.py` 现在只是旧命令兼容层；核心逻辑不再放在该文件中。

## 共享数据契约

`BenchmarkDataset` 同时保存三类信息：

- `variants`：按 `chrom:pos:ref:alt` 去重后的模型推理输入；
- `memberships`：每个 variant 属于哪个 tissue、正例还是负例；
- `targets`：仅评测使用的 `(variant, gene, tissue, effect_size, effect_allele)`。

进入模型前，Runner 会从中生成仅含唯一 variants 和所选 tissues 的 `InferenceDataset`，因此
模型 adapter 无法读取正负标签或 effect label。

模型输出统一为长表：

```text
variant_key  gene_id  tissue  score  n_tracks
```

这里不能把 gene/tissue 过早压缩成一个 variant 标量。eQTL 方向评测需要精确的
`variant-gene-tissue` 分数；sQTL/paQTL/iPaQTL 则优先匹配 VCF 中 `MT` 指定的 gene，
再按 `PI` 配对并在论文的距离阈值上评价。没有 gene 维度的旧模型可以启用 variant fallback。

## 统一运行

AlphaGenome smoke 配置：

```bash
PYTHONPATH=benchmark:src python -m qtl_benchmark.cli \
  --config configs/benchmark_alphagenome_smoke.json
```

AlphaGenome 的一键 unified runner 还接受 `--variant-batch-size N`。它使用本地模型
`score_variants` 的线程并发接口，不经过远程服务，也不改变统一预测 schema 或评价指标。

Borzoi 已生成的 SED 产物可直接进入同一数据与评测流程：

```bash
PYTHONPATH=benchmark:src python -m qtl_benchmark.cli \
  --config configs/benchmark_borzoi_smoke.json
```

四类任务的模型输出对应如下：

| task | AlphaGenome scorer | Borzoi statistic |
|---|---|---|
| eQTL | `RNA_SEQ` | tissue-aware `SED` |
| sQTL | `SPLICE_JUNCTIONS` | `nDi` + gene span |
| paQTL | `POLYADENYLATION` | `COVR` + 3' UTR |
| iPaQTL | `POLYADENYLATION` | `COVR` |

非 eQTL 可直接选用 `configs/benchmark_<model>_<task>.json`。Borzoi 神经网络产物由
`scripts/run_borzoi_multi_gpu.sh` 生成，AlphaGenome 则由统一 CLI 直接推理。
单卡/多卡的一键入口和 GPU 分配语义见 [`gpu_execution.md`](gpu_execution.md)。

可独立执行或复用阶段：

```bash
# 只推理，写 predictions.sqlite 和 predictions.tsv.gz
python -m qtl_benchmark.cli --config CONFIG.json --predict-only

# 不加载模型，只统一评价已有缓存
python -m qtl_benchmark.cli --config CONFIG.json --evaluate-only

# 清除该 run 的模型预测缓存并重新预测
python -m qtl_benchmark.cli --config CONFIG.json --force
```

每次输出固定为：

```text
<output_dir>/<model.name>/<dataset.name>/
├── run.json
├── predictions.sqlite
├── predictions.tsv.gz
├── predictions/<tissue>_coefficients.tsv.gz
├── predictions/<tissue>_classification.tsv.gz
├── metrics.tsv
└── metrics.json
```

缓存绑定模型 class 与完整配置的 SHA256 fingerprint，避免不同 checkpoint 的结果混入。

AlphaGenome 与 Borzoi eQTL 的原生 tissue 粒度和 `max-variants` 裁剪语义不同。需要正式横向
比较时，使用 `scripts/compare_alphagenome_borzoi_eqtl.sh`：它固定使用同一份 Borzoi subset、
同一 evaluator，同时输出 AlphaGenome 原生精细 tissue、AlphaGenome 的 Borzoi broad-tissue
对齐视图和 Borzoi 原生视图。输入一致性的精确定义、审计文件和输出 schema 见
[`eqtl_model_comparison.md`](eqtl_model_comparison.md)。

公共比较固定输出 `common_metrics.tsv` 和 `common_metrics_summary.tsv`；模型各自的原生指标继续
保留。`shared_input_statistics.tsv`、`view_input_statistics.tsv` 和输入哈希用于证明各 view 的
variant、正负例 membership 与 causal targets 完全一致，任何模型缺少共享 variant 都直接失败。

评价完成时，`BenchmarkRunner` 会把 `metrics.tsv` 的每一行完整输出为一条 `[metric]` JSON
日志；eQTL 按 tissue 输出，sQTL/paQTL/iPaQTL 按 distance cutoff 输出。日志不会使用 pandas
的省略显示，所有指标列和计数列都会保留，`NaN` 在 JSON 中表示为 `null`。`metrics.json`
保存相同的结构化记录，适用于后续自动汇总。该机制位于统一 runner，因此 Borzoi、
AlphaGenome、NTv3 以及通用配置入口的行为一致。

## 接入新模型

### 推荐方式：逐 variant 产生 gene/tissue 表

如果模型与 AlphaGenome 类似，只需继承 `QTLModel` 并实现：

```python
class MyModel(QTLModel):
    name = "my-model"
    supported_tasks = frozenset({"eqtl", "sqtl"})

    def setup_task(self, context):
        self.model = load_checkpoint(...)

    def score_variant_table(self, record):
        # 返回 gene_id, score（或 raw_score）；eQTL 还返回 tissue/gtex_tissue
        return my_tidy_dataframe

    def predict_batch(self, task, variants, context):
        # 仅旧的标量 benchmark 要求；新 tissue-aware runner 不调用它。
        raise NotImplementedError
```

配置里的 `model.class` 指向该类。装配器会自动使用 `TidyVariantModelAdapter`，并负责：

- 模型只初始化一次；
- 相同基因/tissue 的 track 求均值；
- variant 去重与断点续跑；
- 输出 schema 验证；
- setup/teardown，即使异常也正确执行。

### 批量模型或已有产物

如果模型必须一次处理整个 VCF，直接继承 `ModelAdapter` 并实现 `predict(dataset, store,
context)`。`BorzoiSEDModelAdapter` 是参考实现：它读取 Borzoi 上游 HDF5、匹配 GTEx tracks，
然后写入相同的 `PredictionStore`。Borzoi 神经网络推理仍可由
`scripts/run_eqtl_benchmark.sh` 产生 HDF5，但数据契约和最终指标已经统一。

## 接入新 benchmark

增加新的 QTL benchmark 时实现两项即可：

1. 新 `DataModule.load()`，产生 `BenchmarkDataset`；
2. 新 `Evaluator.evaluate()`，消费标准预测表。

若模型输出契约不变，已有模型 adapter 无需修改。`MatchedQTLDataModule` 与
`MatchedQTLEvaluator` 已覆盖 sQTL/paQTL/iPaQTL，新 benchmark 只在标签或指标语义变化时才需
新增实现。测试建议使用小型假模型，验证数据划分、配对、预测 schema、缺失分数与缓存命中；
`test_unified_framework.py` 和 `test_matched_qtl_framework.py` 给出了完整例子。
