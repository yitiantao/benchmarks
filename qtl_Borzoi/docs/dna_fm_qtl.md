# DNA-FM 接入统一 QTL benchmark

## 模型与数据边界

DNA-FM 指 `/data/yitian_workspace/DNA/DNA_FM` 中独立实现的 PyTorch NTv3。
benchmark 不复制模型源码、训练配置或权重，而是从训练产物的 `config.yaml` 重建
`NTv3ForPostTraining`，再严格加载 Accelerate 保存的 `model.safetensors`。

数据和指标与 AlphaGenome 完全共用：

- eQTL：`GTExEQTLDataModule` 和 `GTExEQTLEvaluator`；
- sQTL、paQTL、iPaQTL：`MatchedQTLDataModule` 和 `MatchedQTLEvaluator`；
- 推理输出仍为 `variant_key, gene_id, tissue, score, n_tracks`；
- SQLite 缓存、断点续跑、variant 分片和最终指标均由统一框架负责。

配置同时固定 checkpoint 和训练 YAML 的 SHA-256。模型文件内容变化后不能静默复用已有
prediction cache。

## ALT−REF 打分

每个 SNV 构造等长的 REF 和 ALT 序列，分数方向始终是 `ALT−REF`。若 QTL VCF 的 ALT
恰好等于 hg38 FASTA base，adapter 会先构造 VCF REF background，再生成 ALT 序列；FASTA
与两者均不匹配时默认报错。

默认使用训练 phase-1 的 131,072 bp context。post-training 模型只监督中间 37.5%，所以
输出覆盖中心 49,152 bp。

当前 step340000 正式配置和 `DNAFMModel` 的代码缺省值均使用 `float32` 推理，与官方 NTv3
post benchmark 的计算精度对齐。`float32` 不启用 PyTorch autocast；只有显式传入
`--mixed-precision bfloat16` 或使用旧的 step150000 历史配置时，才会让适合的 CUDA 算子在
BF16 下运行。BF16 通常更快且更省显存，但中间结果会舍入，因此不作为跨模型正式比较的默认值。

### eQTL

训练 manifest 中有 7,362 条 human functional tracks。adapter 不计算全部输出，而是只保留
`dataset=gtex` 且 `is_rna=true` 的 86 条轨道，再用 Borzoi 的 `TISSUE_KEYWORDS` 将 49 个
benchmark tissues 映射到 broad GTEx groups。每个 benchmark tissue 通常对应 2–3 条 GTEx
tracks。当前 canonical manifest 的 GTEx 子集缺少 heart；两个 heart benchmark tissues 会
显式回退到 `encode_v3` 的 heart RNA tracks，具体来源写入 `model_run.json`，不会静默混用。

对预测区间内每个 GENCODE gene，在 gene span、对应 tissue tracks 和所有输出位置上取
保留符号的最大绝对 `ALT−REF`。Brain、artery、skin 等精细 tissue 因 checkpoint 的 GTEx
head 粒度而共享 broad-tissue 分数；这与 Borzoi GTEx targets 的粒度一致，不应描述为
AlphaGenome 的精细 tissue 原生输出。

功能输出处于训练时的 target-scaled 空间。QTL adapter 直接比较 REF/ALT，不进行 inverse
scaling，也不额外施加 log-fold 或 arcsinh。

### sQTL、paQTL、iPaQTL

annotation logits 先对最后的二分类维度做 softmax，再计算 positive-class probability 的
`ALT−REF`：

| task | annotation heads |
|---|---|
| sQTL | `splice_donor`, `splice_acceptor` |
| paQTL | `polyA_signal` |
| iPaQTL | `polyA_signal`, `3UTR_plus`, `3UTR_minus` |

位置和 heads 使用 signed max-abs 聚合。输出绑定 VCF 中 `MT` gene；统一 evaluator 优先按
gene 匹配，并保留 variant fallback。

## 固定 checkpoint

当前默认配置 `configs/dna_fm_100m_post_step_0034000.json` 固定到 100M phase-1 run 的
`/data/yitian_workspace/DNA/benchmarks/step_0034000/model.safetensors`。目录名虽然是
`step_0034000`，但 `trainer_state.json` 中 `optimizer_step` 为 340000。

训练目录里的 `best/model.safetensors` 会随着验证结果变化，不适合在没有内容哈希的情况下
作为可恢复 benchmark 的隐式 checkpoint。

它的 tensor 结构与原 100M step 150000 权重一致，并在配置中固定了独立的 SHA-256。旧配置
`configs/dna_fm_100m_post.json` 保留原来的 BF16 设置，仅用于追溯历史结果。当前 checkpoint
的默认 FP32 配置可直接运行：

```bash
./scripts/run_dna_fm_step_0034000.sh --gpus 0,1,2,3
```

各模型当前推荐入口及 multi-GPU 分片语义见
[`dna_fm_script_entrypoints.md`](dna_fm_script_entrypoints.md)。

## 运行

与 AlphaGenome 正式配置对齐的零参数一键运行（8 张 GPU、4 个 QTL 任务、每任务
limit 2000）：

```bash
./scripts/all_dna_fm_qtl.sh
```

单卡 smoke test：

```bash
./scripts/run_dna_fm_multi_gpu.sh --gpus 0 \
  --tasks all \
  --max-variants 1 \
  --batch-size 1 \
  --output-dir outputs/dna_fm_smoke
```

多卡按 variant 分片：

```bash
./scripts/run_dna_fm_multi_gpu.sh --gpus 0,1,2,3 \
  --tasks all \
  --max-variants 100 \
  --batch-size 1 \
  --output-dir outputs/dna_fm_qtl
```

启动脚本优先使用当前 Conda 根目录下的 `ntv3` 环境；找不到时才尝试训练使用的
`/data/pengcheng_workspace/miniconda3/envs/dna_t5/bin/python`。可通过 `DNA_FM_PYTHON` 或
`--python` 覆盖。运行环境必须同时包含 PyTorch、safetensors、PyYAML、pandas 和 pyfaidx。
checkpoint、训练 YAML、源码快照和模型名也可以通过统一 runner 的命令行参数覆盖。

建议先保持 `batch-size=1`：一个 variant 已包含 REF/ALT 两条 131 kb 序列。多 GPU 扩展应
优先增加 variant shards，而不是增大单进程 batch。

DNA-FM 与 Borzoi、AlphaGenome、官方 NTv3 post 在 tissue 粒度、score 空间和各 QTL
head 上的完整对照见 [`qtl_model_implementation_comparison.md`](qtl_model_implementation_comparison.md)。
