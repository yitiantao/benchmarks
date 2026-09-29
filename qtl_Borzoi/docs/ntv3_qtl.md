# NTv3 接入与 QTL 测试说明

## 结论和适用范围

NTv3 可以接入当前统一 benchmark，并运行 eQTL、sQTL、paQTL、iPaQTL 四类数据与同一套
Evaluator。不过官方没有发布这四类 QTL 的专用 scorer，因此这里是**零样本迁移测试**，不是
复现一个官方 NTv3 QTL benchmark，也不能把结果解释成与 Borzoi SED 完全等价。

官方模型分两类：

| checkpoint | 官方参数量 | 本项目的 QTL 分数 |
|---|---:|---|
| `NTv3_8M_pre` | 7,692,507 | masked-base log-odds |
| `NTv3_100M_pre` | 106,463,611 | masked-base log-odds |
| `NTv3_650M_pre` | 651,829,819 | masked-base log-odds |
| `NTv3_100M_post` | 120,059,060 | task head 的 ALT−REF |
| `NTv3_650M_post` | 679,787,636 | task head 的 ALT−REF |

pre 和 post 的数值定义不同，跨类型比较时应分别报告。不同大小的 pre 模型之间、不同大小的
post 模型之间更适合作为规模对照。

官方 eQTL benchmark VCF 的 REF/ALT 不是严格的参考基因组方向，而是把 ALT 统一为 SuSiE 表中的
`minor_allele`。全量交叉检查的 97,922 条 eQTL tissue 记录全部满足
`VCF ALT == minor_allele`。当 minor allele 恰好是 hg38 reference allele 时，VCF ALT 就会匹配
FASTA、VCF REF 则不匹配；这不是 hg38 模板或坐标版本错误。当前数据中共有 21,518 次这种情况，
占 eQTL tissue 记录的 21.97%，不能作为少量坏记录跳过。所有这些记录的 VCF REF/ALT 都是
variant ID 所编码 REF/ALT 的精确交换。

NTv3 adapter 与 Borzoi `snp_seq1` 保持一致：先把该位置构造成 VCF REF background，再生成 VCF
ALT 序列，分数仍严格表示 benchmark 所需方向的 `ALT-REF`。日志会输出
`[reference/orientation]`，每个 task 的修正数量也记录在 `model_run.json` 的
`fasta_matches_vcf_alt`；若 FASTA 与 REF、ALT 都不匹配，默认仍立即报错。完整审计结果见
`reports/qtl_reference_audit/summary.json`，逐条位置见
`reports/qtl_reference_audit/exceptions.tsv`。

## 当前实现的打分方法

### post-trained：优先路线

同一 SNV 构造 REF 与 ALT 两条等长序列，一次前向后计算 `ALT - REF`。不同任务使用：

| task | NTv3 head | 聚合方式 |
|---|---|---|
| eQTL | checkpoint 内的 GTEx RNA tracks，按 Borzoi broad-tissue keyword 分组 | 在 gene span、对应 tissue tracks 和输出位置中取最大绝对变化并保留符号 |
| sQTL | `splice_donor`、`splice_acceptor` | 所有输出位置/head 中最大绝对变化并保留符号 |
| paQTL | `polyA_signal` | 同上 |
| iPaQTL | `polyA_signal`、`3UTR+`、`3UTR-` | 同上 |

BED heads 先对二分类 logits 做 softmax，再计算 positive-class 概率差；BigWig tracks 直接比较
logits。NTv3 只输出输入序列中间 37.5% 的 functional/BED tracks，eQTL 只为这一区间
重叠到的 GENCODE gene 生成预测。

eQTL 映射不再使用 K562/HepG2 两条示例 track 复制到全部 tissue。正式 post 配置使用
`configs/targets_rna.txt`（SHA-256 固定在 JSON 中）解析官方 checkpoint 的 7,362 条 human
track，再复用 Borzoi `TISSUE_KEYWORDS`：

- 49 个 benchmark tissues 中，47 个使用 `GTEX-*` RNA tracks；
- 每个 broad GTEx group 通常有 2–3 条 sample tracks；
- 官方 checkpoint 缺少 3 条 GTEx heart tracks，`Heart_Atrial_Appendage` 和
  `Heart_Left_Ventricle` 显式回退到 32 条 ENCODE heart RNA tracks；
- 去重后共选择 107 条 RNA tracks，来源逐 tissue 写入 `model_run.json`；
- brain、artery、skin 等精细 GTEx tissues 仍共享 broad group，不应解释为 AlphaGenome
  的精细 tissue 输出。

官方 100M post `config.json` 的 7,362 个 track ID 已与 DNA-FM canonical manifest 的
`file_id` 逐项验证：数量、集合和顺序全部一致。K562/HepG2 只保留为未提供
`eqtl_track_metadata_path` 的自定义旧配置 fallback，不是本仓库 100M/650M post 的正式路线。

### pre-trained：MLM fallback

pre 模型没有 functional heads。本项目把变异位置替换为 `<mask>`，计算：

```text
score = log P(ALT | sequence with masked variant)
      - log P(REF | sequence with masked variant)
```

该分数是序列似然方向，不是表达量、剪接或 polyA 方向。四类任务都可以完成 classification
流程，但 task-specific 解释弱于 post 路线。eQTL 时分数会赋给输入窗口内的 GENCODE genes，并
复制到所有 tissue；这只是让同一 evaluator 可以工作，不能视为 gene/tissue-specific 预测。

## 独立环境

NTv3 使用独立环境 `ntv3`，不会修改 `borzoi_py310` 或 `alphagenome`：

```bash
cd /data/yitian_workspace/DNA/benchmarks/qtl_Borzoi
./scripts/setup_ntv3_benchmark.sh ntv3
```

环境定义在 `environment_ntv3.yml`，包含 Python 3.10、PyTorch 2.5/CUDA 12.1、
Transformers、Hugging Face Hub、Safetensors、pyfaidx 和统一评价依赖。

## gated 权重和下载

所有官方 NTv3 仓库当前都是 gated access。必须先用同一个 Hugging Face 账号打开计划使用的
模型页面和 `InstaDeepAI/ntv3_base_model`，接受共享联系信息/模型条款，再把个人 read token
交给 Hugging Face 客户端。不要把 token 写进 JSON、shell 脚本或日志，也不要贴到聊天中：

```bash
conda run -n ntv3 hf auth login
conda run -n ntv3 hf auth whoami
```

下载器会自动读取登录后保存在用户目录中的 token；临时任务也可以通过 `HF_TOKEN` 环境变量提供。

默认 QTL suite 下载三种大小，并同时覆盖 pre/post 路线：

```bash
NTV3_CONDA_ENV=ntv3 \
./scripts/download_ntv3_models.sh --suite qtl
```

它包括 `8M_pre`、`100M_post`、`650M_post`。其他选择：

```bash
# 三个参数规模的 MLM 模型
./scripts/download_ntv3_models.sh --suite pre

# 两个参数规模的 functional-track 模型
./scripts/download_ntv3_models.sh --suite post

# 五个模型全部下载
./scripts/download_ntv3_models.sh --suite all
```

下载器固定官方 commit，只取 PyTorch `model.safetensors`、配置、tokenizer 和 remote code，跳过
重复的 JAX 权重。缓存目录是 `.benchmark_deps/ntv3_hf/`，完成后会写
`download_manifest.json`。粗略按 float32 参数计算，五个 PyTorch 权重约需 6.3 GB，实际缓存还
会包含少量配置和代码。

常见失败：

- `401`/`GatedRepoError`：账号尚未接受对应仓库条款，或 `HF_TOKEN` 没有 read 权限；
- `torch.cuda.is_available() is false`：当前节点没有挂载 GPU/驱动，应换到 GPU 节点；
- remote code 离线找不到：先在线执行下载器，它会同时缓存 `ntv3_base_model`；
- 不应在命令行回显真实 token，也不要提交 token 文件。

## 单卡运行

先用一个 variant、32 kb、batch 1 做 post 模型冒烟测试：

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

分别测试不同参数规模：

```bash
for config in \
  configs/ntv3_8m_pre.json \
  configs/ntv3_100m_pre.json \
  configs/ntv3_650m_pre.json \
  configs/ntv3_100m_post.json \
  configs/ntv3_650m_post.json; do
  NTV3_CONDA_ENV=ntv3 \
  ./scripts/run_ntv3_multi_gpu.sh --gpus 0 \
    --tasks all --max-variants 1 --batch-size 1 \
    --model-config "$config" \
    --data-dir data \
    --output-dir "outputs/$(basename "$config" .json)_smoke"
done
```

## 多卡运行

多卡模式让同一个 checkpoint 在每张卡加载一份，各卡处理不同 variant shard；全部预测完成后
只评价一次。它不是把一个 checkpoint 切到多张卡，也不是每张卡跑一个不同参数规模的模型。

100M post-training 零参数正式入口（8 张 GPU、131,072 bp context、4 个 QTL 任务、
每任务 limit 2000）：

```bash
./scripts/all_ntv3_qtl.sh
```

```bash
NTV3_CONDA_ENV=ntv3 NTV3_GPU_IDS=0,1,2,3 \
./scripts/run_ntv3_multi_gpu.sh \
  --tasks all \
  --max-variants 10 \
  --batch-size 1 \
  --model-config configs/ntv3_100m_post.json \
  --data-dir data \
  --output-dir outputs/ntv3_100m_post_all_qtl
```

如果要比较 100M 和 650M，应顺序执行两个命令；每一个模型内部再使用全部 GPU 做 variant
分片，避免“不同卡跑不同模型”导致单个模型等待时间由最慢任务决定。

## 长度、batch 和显存

- 主模型要求 `sequence_length` 是 128 的倍数，官方支持最长 1 Mb；
- 配置默认 32,768 bp，适合先验证代码；
- post 模型一个 variant 会产生 REF/ALT 两条序列，所以 `--batch-size N` 实际前向 batch 是
  `2*N`；pre 模型是 `N` 条 masked 序列；
- post 的 human BigWig 输出有 7,362 tracks，即使 eQTL 最后只使用 107 条 RNA
  tracks，Hugging Face 模型仍会先产生完整 head，因此长度和 batch 对显存影响很大；
- 建议 post 从 `--batch-size 1` 开始。只有显存监控稳定后再增大；
- eQTL 若要提高 causal gene 落入输出区间的概率，可用 `--sequence-length 131072`、`524288`
  或 `1048576`，但应先对一个 variant 测显存；
- 更改 model config、长度或 batch 会生成不同模型 fingerprint，旧缓存不会被误混用。

## 输出与断点续跑

机器可读结果位于：

```text
<output-dir>/<all|max_N>/<model-name>/<dataset>/predictions.sqlite
<output-dir>/<all|max_N>/<model-name>/<dataset>/predictions.tsv.gz
<output-dir>/<all|max_N>/<model-name>/<dataset>/metrics.tsv
<output-dir>/<all|max_N>/<model-name>/<dataset>/metrics.json
<output-dir>/<all|max_N>/<model-name>/<dataset>/model_run.json
```

预测按 variant 写入 SQLite，失败后原命令重跑会跳过已完成记录。多卡 workers 共享 WAL 数据库；
所有 shard 成功后才导出完整 TSV 并评价。评价结果还会逐行输出为 `[metric]` JSON 日志，字段与
`metrics.tsv` 完全一致。`--force` 只允许单进程运行。

## 官方资料

- NTv3 官方教程：<https://huggingface.co/spaces/InstaDeepAI/ntv3/tree/main/notebooks_tutorials>
- NTv3 官方模型与架构说明：<https://github.com/instadeepai/nucleotide-transformer/blob/main/docs/nucleotide_transformer_v3.md>
- 8M pre 模型页：<https://huggingface.co/InstaDeepAI/NTv3_8M_pre>
- 100M post 模型页：<https://huggingface.co/InstaDeepAI/NTv3_100M_post>
- 650M post 模型页：<https://huggingface.co/InstaDeepAI/NTv3_650M_post>
