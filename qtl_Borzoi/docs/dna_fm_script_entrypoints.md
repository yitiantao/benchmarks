# 统一 benchmark 脚本入口

以下命令均假定当前目录是项目根目录：

```bash
cd /data/yitian_workspace/DNA/benchmarks/qtl_Borzoi
```

## 当前入口

四个模型都以 multi-GPU 脚本作为唯一推荐的模型 benchmark 入口：

| 模型 | benchmark 入口 | 单卡 |
|---|---|---|
| Borzoi | `scripts/run_borzoi_multi_gpu.sh` | `--gpus 0` |
| AlphaGenome | `scripts/run_alphagenome_multi_gpu.sh` | `--gpus 0` |
| NTv3 | `scripts/run_ntv3_multi_gpu.sh` | `--gpus 0` |
| DNA-FM | `scripts/run_dna_fm_multi_gpu.sh` | `--gpus 0` |

原来的 `run_qtl_benchmark.sh`、`run_alphagenome_qtl_benchmark.sh`、
`run_ntv3_qtl_benchmark.sh` 和 `run_dna_fm_qtl_benchmark.sh` 已被对应 multi-GPU 脚本吸收并
删除。`run_*_single_gpu.sh` 只作为兼容别名保留，内部也是调用 multi-GPU 入口并传一张 GPU。

`all_*_qtl.sh` 不是另一套执行实现，而是正式实验的参数预设。DNA-FM 另有
`run_dna_fm_step_0034000.sh`，它是当前 checkpoint 的小样本参数预设。

## multi-GPU 的含义

AlphaGenome、NTv3 和 DNA-FM 的流程是：

```text
GPU 0 -> prediction shard 0
GPU 1 -> prediction shard 1
GPU 2 -> prediction shard 2
GPU 3 -> prediction shard 3
                    |
                    +-- 全部成功后只评估一次
```

这不是模型并行。每张卡加载一份完整模型，只处理互不重叠的 variant shard。worker 共享 WAL
SQLite prediction cache；失败时不会评价不完整结果，用同一命令重跑会复用已经完成的 variant。
`--predict-only`、`--evaluate-only`、`--num-shards`、`--shard-index` 和 `--force` 属于 launcher
内部参数，不能从 multi-GPU 公共命令手工传入。

Borzoi 的预测产物是 HDF5，分片和合并仍沿用 Borzoi scorer 的实现，但 GPU 列表、数据准备、
原生推理和统一评价现在都由 `run_borzoi_multi_gpu.sh` 直接管理。

## 新 DNA-FM checkpoint

checkpoint 路径为：

```text
/data/yitian_workspace/DNA/benchmarks/step_0034000/model.safetensors
```

它有 340 个 tensor，名称、dtype 和 shape 均与原 100M checkpoint 一致，因此复用原 100M 的
训练配置和源码快照。`configs/dna_fm_100m_post_step_0034000.json` 固定了新权重的 SHA-256。
目录名是 `step_0034000`，但 `trainer_state.json` 中 `optimizer_step` 为 340000，所以模型名按
训练状态记为 `DNA-FM-100M-post-step340000`。

当前配置默认使用 FP32 推理，与官方 NTv3 post benchmark 对齐。BF16 仍可通过
`--mixed-precision bfloat16` 显式启用，但不用于默认的跨模型正式比较。精度属于模型缓存
fingerprint；已有 BF16 输出目录不能作为 FP32 预测继续复用，应为 FP32 运行指定新的输出目录。

四卡 smoke test：

```bash
./scripts/run_dna_fm_step_0034000.sh --gpus 0,1,2,3
```

等价的完整命令：

```bash
./scripts/run_dna_fm_multi_gpu.sh \
  --gpus 0,1,2,3 \
  --tasks all \
  --max-variants 1 \
  --batch-size 1 \
  --model-config configs/dna_fm_100m_post_step_0034000.json \
  --data-dir data \
  --output-dir outputs/dna_fm_step_0034000_fp32_smoke
```

这里 eQTL 是每个 tissue/split 最多 1 个 variant；另外三个 matched-QTL 任务各选最多 1 个
negative pair，并保留配对 positive。环境没有自动选对时可设置 `DNA_FM_PYTHON=/path/to/python`。

扩大到每任务 limit 2000：

```bash
./scripts/run_dna_fm_step_0034000.sh \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-variants 2000 \
  --output-dir outputs/dna_fm_step_0034000_max_2000
```

全量运行使用 `--max-variants 0`。不同 checkpoint 应使用不同的输出目录；缓存还会校验模型
fingerprint，防止混入另一套权重。默认 `--batch-size 1` 最稳妥，因为一个 variant 已经包含
REF/ALT 两条 131,072 bp 序列。
