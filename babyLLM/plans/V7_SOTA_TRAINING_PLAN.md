# ChineseBabyLM V7 — SOTA 冲刺详细计划

> **创建日期**: 2026-04-24
> **目标**: BabyLM 2026 (Round 4) Multilingual Track Chinese SOTA
> **硬件**: 4× NVIDIA A6000 (48GB each)
> **配置**: d=448, L=12, H=7/KV=4, ~35M params, 8K SPM vocab

---

## 一、竞赛背景

### BabyLM 2026 (Round 4) — Multilingual Track

| 项目 | 详情 |
|------|------|
| Workshop | 4th BabyLM Workshop @ EMNLP 2026, Budapest |
| 论文截止 | ~May 25, 2026 (ARR) / Mid July (OpenReview) |
| 数据预算 | 100M tokens (Byte Premium 0.9894 for Chinese) |
| Epoch 上限 | ≤10 epochs |
| 总词数上限 | ≤1B words seen |
| 外部数据 | 禁止（任何辅助模型训练数据计入预算） |
| 蒸馏限制 | 外部模型输出计入预算 |
| 提交格式 | HF Transformers 模型 + 论文(8页) + 中间checkpoints |

### 中文评测任务

**零样本 (pseudo-log-likelihood scoring)**:
- MultiBLiMP — 多语言语法最小对
- CLiMP — 16 个中文语言学现象 (16K pairs, Xiang et al. 2021)
- SLING — 38 个中文语言学现象 (38K pairs, Song et al. 2022)

**微调**:
- XNLI, XCOPA, Belebele, ARC, TruthfulQA, INCLUDE, HellaSwag 等

### 中文基准线 (BabyBabelLM Baseline)

| Benchmark | Baseline Score | Random Baseline |
|-----------|---------------|-----------------|
| MultiBLiMP | 82.6 | 50.0 |
| SIB-200 | 82.6 | 14.3 |
| XCOMPS | 70.2 | 50.0 |
| MNLI/XNLI | 52.0 / 49.6 | 33.3 |
| XCOPA | 49.2 | 50.0 |
| XStoryCloze | 48.7 | 50.0 |
| Winogrande | 49.2 | 50.0 |
| HellaSwag | 26.8 | 25.0 |
| Belebele | 26.1 | 25.0 |
| ARC | 26.6 | 25.0 |
| TruthfulQA | 28.8 | 25.0 |
| INCLUDE | 30.7 | 25.0 |
| BMLAMA | 17.4 | 10.0 |
| global-mmlu | 28.1 | 25.0 |

---

## 二、V6 失败根因详细分析

### 2.1 数据层面

| 指标 | V3 (V5 使用) | V6 | 变化 |
|------|-------------|-----|------|
| 训练行数 | 1,203,087 | 800,371 | -33.5% |
| 训练字符 | 129.5M | 28.2M | **-78.2%** |
| 训练 tokens | 73.7M | **15.5M** | **-79.0%** |
| 平均行长 | 107.6 chars | 35.3 chars | -67.2% |
| 文件大小 | 352 MB | 72 MB | -79.5% |

**根因**: V6 的 `prepare_data_v6.py` 使用 `min_length=15, max_length=300` 截断，把大量有效数据过滤掉了。V3 中很多行的长度 > 300 chars（如维基百科段落、教科书段落），被完全丢弃。

### 2.2 模型-数据匹配

| 版本 | 参数量 | 训练 tokens | tokens/param | Chinchilla 最优比 | 差距倍数 |
|------|--------|-----------|-------------|-------------------|---------|
| V5 | 51M | 73.7M | 1.44 | ~20 | 13.9× |
| V6 | 74.5M | 15.5M | **0.21** | ~20 | **95.2×** |
| **V7 目标** | **35M** | **73.7M** | **2.11** | ~20 | **9.5×** |

即使 V7 也远低于 Chinchilla 最优（20:1），但 2.11 已经是当前数据量下能做到的最好比例。

### 2.3 Tokenizer 问题

当前 32K SPM 词表:
- 73.7M tokens / 32K vocab = ~2,300 examples/vocab entry（稀疏）
- 约 5,413 个 token 出现次数 ≤ 5（几乎未训练）
- 908 个 token 是 singleton（只出现 1 次）
- **无 `<mask>` token** → MLM 阶段用不相关的 token 代替

### 2.4 CLM+MLM 实现 Bug

V6 `CLMMLMDataset.__getitem__()` 中:
```python
mlm_labels[:-1] = input_ids[1:]  # 标签偏移 1 位
```
且使用 `vocab_size - 1`（中文 `講`）作为 mask token，而不是专用的 `<mask>` token。

---

## 三、V7 架构设计

### 3.1 模型配置

```yaml
model:
  type: LlamaForCausalLM (HuggingFace)
  vocab_size: 8004           # 8000 SPM + 4 special tokens
  hidden_size: 448
  intermediate_size: 1194    # 448 * 8/3 ≈ 1194
  num_hidden_layers: 12
  num_attention_heads: 7
  num_key_value_heads: 4     # GQA ratio 1.75:1
  max_position_embeddings: 1024
  rms_norm_eps: 1e-5
  rope_theta: 10000.0
  tie_word_embeddings: true  # 共享 embedding 和 LM head
  hidden_act: silu           # SwiGLU
  attn_implementation: sdpa  # Flash Attention via PyTorch SDPA

parameters:
  embedding: 8004 × 448 = 3.58M (shared)
  attention_per_layer: 448×448×4 + 448×119×2 = 1.01M (QKV+O with GQA)
  ffn_per_layer: 448×1194×3 = 1.60M (SwiGLU 3 projections)
  norms_per_layer: 448×2 = 0.9K
  total_per_layer: ~2.61M
  total_layers: 2.61M × 12 = 31.3M
  final_norm: 448
  total: ~35M
```

### 3.2 Tokenizer 设计

```yaml
tokenizer:
  type: SentencePiece Unigram
  vocab_size: 8000
  model_type: unigram        # 比 BPE 更适合中文
  character_coverage: 0.9995
  special_tokens:
    - <unk> (0)
    - <s> (1)
    - </s> (2)
    - <pad> (3)
    - <mask> (4)             # 新增！用于 MNTP
  training_data: processed_v3/train.txt
  expected_coverage: >99.5%
```

为什么 8K 足够:
- 常用汉字: ~3,500 个 (GB2312 一级)
- 常用标点: ~50 个
- 常用词组/子词: ~4,000 个
- 总计: ~7,550 + 特殊 token ≈ 8K

### 3.3 MNTP (Masked Next Token Prediction) 实现

**GPT-BERT 核心创新** (Charpentier & Samuel, 2024):

```
CLM 模式 (12.5% 概率):
  输入:  t1  t2  t3  t4  t5  t6  t7  t8
  标签:  t2  t3  t4  t5  t6  t7  t8  -100
  (标准 next-token prediction)

MNTP 模式 (87.5% 概率):
  输入:  t1  t2  [M] t4  t5  [M] t7  t8
  标签:  t2  t3  t4  t5  t6  t7  t8  -100
  (遮蔽某些位置，但标签始终是 next token)
  
关键: 标签与 CLM 完全一致！只是输入端被部分遮蔽。
```

**实现细节**:
```python
def __getitem__(self, idx):
    input_ids = self.base_data[idx]["input_ids"]   # chunk[:-1]
    labels = self.base_data[idx]["labels"]           # chunk[1:]
    
    if random.random() < self.clm_ratio:  # 12.5% CLM
        return {"input_ids": input_ids, "labels": labels}
    
    # 87.5% MNTP
    masked_input = input_ids.clone()
    mask = torch.rand(input_ids.shape) < self.mask_ratio  # 15-30%
    masked_input[mask] = self.mask_token_id  # <mask> token
    
    return {"input_ids": masked_input, "labels": labels}
    # labels 不变！始终是 next token
```

### 3.4 数据预处理

```python
def prepare_data_v7(input_file, output_file):
    """轻量级清洗 - 保留最大数据量"""
    for line in input_file:
        line = line.strip()
        if len(line) < 5:          # 仅移除过短行
            continue
        
        # 移除 CHILDES 说话人前缀
        line = re.sub(r'^(TARGET_CHILD|MOTHER|FATHER|TEACHER|INVESTIGATOR|CHILD):\s*', '', line)
        
        # 移除控制字符
        line = clean_control_chars(line)
        
        if line:
            output.write(line + '\n')
```

预期：V3 的 1,203,087 行 → 清洗后约 1,100,000+ 行（仅损失 <10%），保留 ~70M+ tokens。

---

## 四、训练配置

### 阶段 1: CLM+MNTP 混合预训练

```yaml
stage: clm_mntp
data: processed_v7/train.txt
val_data: processed_v7/val.txt
output: output/babylm-v7-stage1

training:
  epochs: 10
  batch_size_per_gpu: 32
  num_gpus: 4
  gradient_accumulation: 1
  effective_batch: 128
  steps_per_epoch: ~540    # 70M tokens / (128 × 1024)
  total_steps: ~5400
  
optimizer:
  type: AdamW
  lr: 6e-4
  betas: [0.9, 0.95]
  weight_decay: 0.1
  no_decay_params: [bias, layernorm, rmsnorm, norm]
  max_grad_norm: 1.0
  
scheduler:
  type: cosine_with_warmup
  warmup_steps: 270        # 5% of total
  
regularization:
  attention_dropout: 0.1
  bpe_dropout: 0.1
  label_smoothing: 0.1
  dropout_anneal: true      # 70% 后线性退火到 0

mntp:
  clm_ratio: 0.125          # 1:7 CLM:MNTP
  mask_ratio_start: 0.30
  mask_ratio_end: 0.15      # 线性退火
  mask_token: <mask>        # token id = 4

early_stopping:
  patience: 8
  metric: val_loss

logging:
  wandb_project: chinese-babylm
  wandb_run_name: babylm-v7-448d-12l-clm-mntp-4gpu
  logging_steps: 50
  save_steps: 2000
  save_total_limit: 3
```

**预计训练时间**: ~4-5 小时 (4×A6000, ~5400 steps × ~3s/step)

### 阶段 2: 纯 CLM 微调 (可选)

```yaml
stage: clm_finetune
resume_from: output/babylm-v7-stage1/best_model

training:
  epochs: 5
  lr: 1e-4                  # 降低 6x
  warmup_ratio: 0.03
  attention_dropout: 0.0
  bpe_dropout: 0.0
  label_smoothing: 0.05
  patience: 5
```

**预计训练时间**: ~2 小时

---

## 五、预期结果

### PPL 预期

| 配置变化 | V6 → V7 改善 | PPL 影响 |
|---------|-------------|---------|
| 词表 32K → 8K | 预测空间缩小 4× | PPL 降低 ~70% |
| 数据 15.5M → 73.7M tokens | 4.8× 更多数据 | PPL 降低 ~60% |
| 参数 74.5M → 35M | tokens/param 从 0.21 → 2.1 | PPL 降低 ~50% |
| 修复 MNTP Bug | 正确的混合训练 | PPL 降低 ~30% |
| 训练 4 → 10 epochs | 2.5× 更多训练 | PPL 降低 ~20% |
| **综合预期** | | **PPL < 50 (BPE level)** |

### Benchmark 预期

| Benchmark | Baseline | V7 目标 | 策略 |
|-----------|----------|---------|------|
| MultiBLiMP | 82.6 | >85 | MNTP 训练增强语法能力 |
| SIB-200 | 82.6 | >85 | 语言理解能力提升 |
| XCOMPS | 70.2 | >75 | 微调后提升 |
| XNLI | 49.6 | >55 | MNTP 增强推理 |
| XCOPA | 49.2 | >55 | 因果推理提升 |
| Belebele | 26.1 | >35 | 阅读理解改善 |
| ARC | 26.6 | >35 | 知识获取改善 |

---

## 六、实现清单

### 文件清单

```
src/v7/
├── train_tokenizer.py          # Step 1: 训练 8K SPM tokenizer
├── prepare_data_v7.py          # Step 2: 轻量级数据清洗
├── train_v7.py                 # Step 3: 主训练脚本 (CLM+MNTP)
├── evaluate_v7.py              # Step 5: 评测脚本
├── launch_v7.sh                # Step 4: 启动脚本 (阶段1+2)
└── launch_v7_eval.sh           # Step 6: 评测启动脚本

data/
├── tokenizer_v7/               # Step 1 输出
│   ├── spm.model
│   ├── spm.vocab
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── processed_v7/               # Step 2 输出
    ├── train.txt
    └── val.txt

output/
└── babylm-v7-stage1/           # Step 4 输出
    ├── best_model/
    └── checkpoint-*/
```

### 执行顺序

1. **train_tokenizer.py** — 训练 8K SPM，添加 <mask> token (~5 min)
2. **prepare_data_v7.py** — 轻量清洗 V3 数据 (~2 min)
3. **train_v7.py** — 混合 CLM+MNTP 训练 (~4-5 hours)
4. (可选) **train_v7.py --stage clm_finetune** — 纯 CLM 微调 (~2 hours)
5. **evaluate_v7.py** — 运行评测 benchmarks
6. 调参迭代

---

## 七、关键技术决策与理由

### Q: 为什么用 8K 词表而不是 32K？
A: 数据量 73.7M tokens / 32K vocab = 2300 samples/entry（稀疏）。8K = 9200 samples/entry（合理）。更小的词表 → 更低的 PPL（预测不确定性降低）。BabyLM 冠军 GPT-BERT 也使用相对小的词表。

### Q: 为什么不用 LTG-BERT/DeBERTa 架构？
A: LTG-BERT 需要实现解耦注意力 (disentangled attention)、GEGLU、gated attention 等组件，工程量大。在有限时间内，LLaMA + 正确的 MNTP 是更务实的选择。如果时间允许，可以作为 V8 尝试。

### Q: 为什么不用三阶段训练？
A: V6 的经验表明三阶段流水线增加了复杂度但没有带来收益（Stage 2 的 broken MLM 反而降低了 Stage 1 的效果）。V7 简化为 1-2 阶段，降低风险。

### Q: 为什么不清洗数据？
A: V6 清洗后损失了 79% 的数据。在 100M token 预算下，每一条数据都很宝贵。只做最轻量级的清洗（移除 CHILDES 前缀、空行），保留最大数据量。

### Q: Reverse KL KD 为什么不用了？
A: 蒸馏在 BabyLM 竞赛中有限制（外部模型训练数据计入预算）。且 V6 的 KD 效果有限（PPL 仅从 3135 降到 2747）。集中精力在核心训练上。

---

## 八、多卡并行训练架构 (延续自 V5/V6)

```
┌─────────────────────────────────────────────────────────────┐
│  Server2: 4× NVIDIA RTX A6000 (48GB each)                   │
│                                                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │ GPU 0   │  │ GPU 1   │  │ GPU 2   │  │ GPU 3   │       │
│  │ Rank 0  │  │ Rank 1  │  │ Rank 2  │  │ Rank 3  │       │
│  │ (Main)  │  │         │  │         │  │         │       │
│  │         │  │         │  │         │  │         │       │
│  │ 35M模型 │  │ 35M模型 │  │ 35M模型 │  │ 35M模型 │       │
│  │ ~140MB  │  │ ~140MB  │  │ ~140MB  │  │ ~140MB  │       │
│  │         │  │         │  │         │  │         │       │
│  │ AdamW   │  │ AdamW   │  │ AdamW   │  │ AdamW   │       │
│  │ ~560MB  │  │ ~560MB  │  │ ~560MB  │  │ ~560MB  │       │
│  │         │  │         │  │         │  │         │       │
│  │ 数据分片│  │ 数据分片│  │ 数据分片│  │ 数据分片│       │
│  │ ~18K行  │  │ ~18K行  │  │ ~18K行  │  │ ~18K行  │       │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │
│       └────────────┴───── NCCL AllReduce ──────┘            │
│              (gradient synchronization)                     │
│              PCIe P2P / SYS level                           │
└─────────────────────────────────────────────────────────────┘
```

**框架**: HuggingFace Accelerate + DDP
**有效 batch**: 32 × 4 GPUs = 128 samples/step
**Tokens/step**: 128 × 1024 = 131,072 tokens
**显存预估**: ~2GB/GPU (模型+优化器+激活) → 大量余量
**通信**: NCCL AllReduce, P2P_LEVEL=SYS, 30min timeout
