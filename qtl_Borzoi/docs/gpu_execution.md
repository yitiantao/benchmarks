# QTL benchmark 单卡与多卡完整命令

以下命令均从项目根目录运行：

```bash
cd /data/yitian_workspace/DNA/benchmarks/qtl_Borzoi
```

多卡采用“一张 GPU 一个独立进程”。Borzoi 和 AlphaGenome 都按 variant 数据分片；同一
模型的所有推理 shard 完成后，才会开始下一个模型 replicate。全部推理结束后再生成下游
使用的合并文件和评估结果。

## 1. Borzoi

Borzoi 默认 Conda 环境为 `borzoi_py310`。

### 1.1 单卡、单模型、sQTL smoke test

使用物理 GPU 0，最多选择 10 对 sQTL：

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

### 1.2 四卡、单模型、sQTL smoke test

四张卡会各自加载同一个 replicate，并分别计算该模型 positive/negative VCF 的一个数据
分片。所有分片完成后会合并为下游兼容的 `sed.h5`。

```bash
BORZOI_CONDA_ENV=borzoi_py310 \
BORZOI_GPU_IDS=0,1,2,3 \
./scripts/run_borzoi_multi_gpu.sh \
  --tasks sqtl \
  --max-variants 10 \
  --variant-batch-size 2 \
  --model models/replicate_0/model0_best.h5 \
  --model-name borzoi-replicate-0 \
  --conda-env borzoi_py310 \
  --data-dir data \
  --output-dir outputs/my_borzoi_sqtl
```

### 1.3 四卡、四个 published replicate、sQTL

`--model-dir models` 会找到四个 replicate。执行顺序严格为：四张卡共同计算 replicate 0，
等待全部 shard 完成；再共同计算 replicate 1，随后是 replicate 2 和 replicate 3。不同
replicate 不会同时占用不同 GPU。

```text
replicate 0: GPU 0/1/2/3 -> shard 0/1/2/3 -> barrier
replicate 1: GPU 0/1/2/3 -> shard 0/1/2/3 -> barrier
replicate 2: GPU 0/1/2/3 -> shard 0/1/2/3 -> barrier
replicate 3: GPU 0/1/2/3 -> shard 0/1/2/3 -> collect / ensemble
```

```bash
BORZOI_CONDA_ENV=borzoi_py310 \
BORZOI_GPU_IDS=0,1,2,3 \
./scripts/run_borzoi_multi_gpu.sh \
  --tasks sqtl \
  --max-variants 10 \
  --variant-batch-size 2 \
  --model-dir models \
  --model-name borzoi-four-replicates \
  --conda-env borzoi_py310 \
  --data-dir data \
  --output-dir outputs/my_borzoi_sqtl_four_replicates
```

### 1.4 四卡运行四类 QTL 全量 benchmark

`--max-variants 0` 表示全量。这是长时间正式任务：

```bash
BORZOI_CONDA_ENV=borzoi_py310 \
BORZOI_GPU_IDS=0,1,2,3 \
./scripts/run_borzoi_multi_gpu.sh \
  --tasks all \
  --max-variants 0 \
  --variant-batch-size 2 \
  --model-dir models \
  --model-name borzoi-four-replicates \
  --conda-env borzoi_py310 \
  --data-dir data \
  --output-dir outputs/borzoi_all_qtl
```

同一个入口也可直接指定任意 GPU。例如在 GPU 2、3 上运行 paQTL：

```bash
./scripts/run_borzoi_multi_gpu.sh \
  --gpus 2,3 \
  --tasks paqtl \
  --max-variants 10 \
  --variant-batch-size 2 \
  --model models/replicate_0/model0_best.h5 \
  --model-name borzoi-replicate-0 \
  --conda-env borzoi_py310 \
  --data-dir data \
  --output-dir outputs/my_borzoi_paqtl
```

每个 Borzoi 子进程只看到分配给它的一张 GPU。分片临时结果位于
`merge_pos/jobN/sed.h5` 和 `merge_neg/jobN/sed.h5`，合并结果仍是原路径下的
`merge_pos/sed.h5` 和 `merge_neg/sed.h5`。同一个输出目录可以续跑，已完成的 HDF5 会被
复用。更换模型时必须更换 `--output-dir`，防止混合不同权重的结果。

`--variant-batch-size N` 控制每个 GPU worker 的单次模型调用，实际 sequence batch 为
`2*N`（每个变异一条 REF 和一条 ALT）。默认是 `1`；推荐先测试 `1` 或 `2`，出现 OOM 时调小。
当前 `params_pred.json` 模型最多只能设为 `3`，`4` 会确定性触发 TensorFlow GPU kernel 的 int32
元素计数溢出，而不是显存不足。它不会把一次 TensorFlow 前向跨 GPU 拆分，多卡仍然是每卡一个
独立数据 shard。更完整的合并、完成标记、TensorFlow 日志和 fold/染色体语义见
[`implementation_details.md`](implementation_details.md)。

## 2. AlphaGenome

AlphaGenome 默认 Conda 环境为 `alphagenome`，默认 checkpoint 配置为
`configs/alphagenome_local.json`，实际 checkpoint 路径是
`/data/yitian_workspace/DNA/alphagenome_models/all_folds`。

### 2.1 单卡 sQTL smoke test

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

### 2.2 四卡 sQTL smoke test

```bash
ALPHAGENOME_CONDA_ENV=alphagenome \
./scripts/run_alphagenome_multi_gpu.sh \
  --gpus 0,1,2,3 \
  --tasks sqtl \
  --max-variants 10 \
  --variant-batch-size 2 \
  --model-config configs/alphagenome_local.json \
  --checkpoint-path /data/yitian_workspace/DNA/alphagenome_models/all_folds \
  --model-name alphagenome-all-folds \
  --data-dir data \
  --output-dir outputs/my_alphagenome_sqtl
```

等价的环境变量写法：

```bash
ALPHAGENOME_CONDA_ENV=alphagenome \
ALPHAGENOME_GPU_IDS=0,1,2,3 \
./scripts/run_alphagenome_multi_gpu.sh \
  --tasks sqtl \
  --max-variants 10 \
  --variant-batch-size 2 \
  --model-config configs/alphagenome_local.json \
  --checkpoint-path /data/yitian_workspace/DNA/alphagenome_models/all_folds \
  --model-name alphagenome-all-folds \
  --data-dir data \
  --output-dir outputs/my_alphagenome_sqtl
```

### 2.3 四卡运行四类 QTL 全量 benchmark

```bash
ALPHAGENOME_CONDA_ENV=alphagenome \
./scripts/run_alphagenome_multi_gpu.sh \
  --gpus 0,1,2,3 \
  --tasks all \
  --max-variants 0 \
  --variant-batch-size 1 \
  --model-config configs/alphagenome_local.json \
  --checkpoint-path /data/yitian_workspace/DNA/alphagenome_models/all_folds \
  --model-name alphagenome-all-folds \
  --data-dir data \
  --output-dir outputs/alphagenome_all_qtl
```

AlphaGenome launcher 按 variant 序号稳定分片，每张卡一个进程。各 shard 写入同一个 WAL
SQLite cache，但 variant 互不重叠；全部 shard 成功后只执行一次 TSV 导出和统一评估。任一
shard 失败时不会评估不完整结果，重新运行同一命令即可续跑。

`--variant-batch-size N` 控制每个 AlphaGenome GPU 进程内同时提交的本地变异数。实现调用本地
模型的 `score_variants(..., max_workers=N)`，其内部以线程并发 N 次独立的 JAX
`score_variant`，不是把 N 个变异合并为一个 tensor batch。默认 `1` 最稳妥；`2` 可能提高吞吐，
也可能接近翻倍单进程的激活显存峰值，正式扩大前应先做小样本显存测试。该参数不进入模型
fingerprint，调整后可安全续用已完成的预测缓存。

多卡 AlphaGenome 会在每张卡各加载一份 checkpoint，GPU 显存和主存必须能容纳对应数量的
模型进程。多卡入口内部管理 `--predict-only`、`--evaluate-only`、`--num-shards` 和
`--shard-index`，不要在命令中手动传递这些参数。
