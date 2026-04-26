# State-of-the-Art Techniques for Training Small Language Models (<100M Parameters)

> Research summary for BabyLM-style challenges
> Compiled: 2026-04-25
> Focus: Data-limited pretraining on ~100M words

---

## Table of Contents

1. [BabyLM 2024 Winning Solutions](#1-babylm-2024-winning-solutions)
2. [Optimal Training Objectives](#2-optimal-training-objectives)
3. [Architecture Innovations](#3-architecture-innovations)
4. [Training Recipes & Hyperparameters](#4-training-recipes--hyperparameters)
5. [Data Strategies](#5-data-strategies)
6. [Knowledge Distillation](#6-knowledge-distillation)
7. [Advanced Regularization](#7-advanced-regularization)
8. [RoPE Improvements](#8-rope-improvements)
9. [Multi-Stage Training Pipelines](#9-multi-stage-training-pipelines)
10. [Implementation Recommendations](#10-implementation-recommendations)

---

## 1. BabyLM 2024 Winning Solutions

### 1.1 GPT-BERT (Winner, BabyLM 2024 — Both Strict & Strict-Small Tracks)

**Paper**: "GPT or BERT: why not both?" (Charpentier & Samuel, CoNLL-BabyLM 2024)
**Code**: https://github.com/ltgoslo/gpt-bert
**Models**: `ltg/gpt-bert-babylm-base` (119M), `ltg/gpt-bert-babylm-small` (30M)

#### Core Innovation: Hybrid Masked-Causal Training (MNTP)

The key insight is **Masked Next-Token Prediction (MNTP)** — a reformulation of MLM where masked token predictions are shifted one position to the right, aligning them with CLM's next-token prediction pattern. This allows a single transformer to train on both objectives simultaneously with zero architectural changes.

**How MNTP works:**
- Traditional MLM: predict masked token at position k → output at position k
- MNTP: predict masked token at position k+1 → output at position k
- Both CLM and MNTP now produce output at position k representing token at position k+1
- Both minimize cross-entropy loss, share all parameters, use the same transformer

**Dataset handling**: Duplicate the dataset — one copy for causal objective, one for masked. Control the causal-to-masked ratio per batch.

**Optimal ratio**: 1:15 causal-to-masked (i.e., ~6.25% causal, ~93.75% masked). Even adding just 6.25% MNTP training yields:
- +4.2% on BLiMP
- +0.9% on MNLI
- +33.3% on LAMBADA

**Results (Strict Track, 100M words):**

| Model | BLiMP | BLiMP-S | GLUE | EWOK |
|-------|-------|---------|------|------|
| GPT-BERT | **86.1** | **76.8** | **81.5** | **58.4** |
| ELC-BERT (2023) | 85.8 | 76.8 | 78.3 | 56.3 |
| LTG-BERT | 85.3 | 76.6 | 77.9 | 56.0 |
| Encoder baseline | 69.2 | 66.5 | 68.4 | 51.9 |
| Decoder baseline | 73.1 | 60.6 | 69.0 | 52.1 |

#### Additional GPT-BERT Modifications

1. **Attention Gate** (from AlphaFold2): Gate attention outputs via a learned linear projection + element-wise multiplication. Both attention and FFN modules use the same gated pattern:
   ```python
   def layer(x, layer_id):
       residual = x
       x = layer_norm(x)          # parameter-free
       g = gate(x)                # linear projection
       if attention_layer:
           x = attention(x)
       else:
           x = linear(x)
       x = glu(x, g)              # GEGLU activation
       x = layer_norm(x)          # parameter-free
       x = output(x)              # linear projection
       return residual + x
   ```

2. **Layer Weighting** (from ELC-BERT, made more granular): Each layer selects a linear combination of outputs from all previous layers. GPT-BERT treats attention and FFN as separate layers with independent learnable scalar weights α_ij:
   ```python
   output_0 = embedding(subword_indices)
   for i in range(1, n_layers + 1):
       output_i = sum(α_ij * layer(output_{j-1}, j) for j in 1..i)
   return output_{n_layers}
   ```

3. **Batch-Size Scheduling**: Linearly increase batch size during training (start at 1/4 of max, ramp up). Max batch size = 4M tokens. Divides total tokens needed by 2×. Intuition: high-quality gradients mainly needed at late stages.

4. **Mask Scheduling**: Linearly decrease masking probability from 30% → 15% over training. Recovers more unmasked tokens early when batches are small; better aligns with downstream MLM usage.

#### Pretraining Hyperparameters

| Hyperparameter | Strict (100M) | Strict-Small (10M) |
|---|---|---|
| Parameters | 119M | 30M |
| Layers | 12 | 12 |
| Hidden size | 768 | 384 |
| FF intermediate | 2,560 | 1,280 |
| Vocab size | 16,384 | 8,192 |
| Attention heads | 12 | 6 |
| Dropout (hidden/attn) | 0.1 | 0.1 |
| Training steps | 15,625 | 7,812 |
| Batch size | 1M → 4M tokens | 1M → 4M tokens |
| Sequence length | 128 → 512 | 128 → 512 |
| Warmup ratio | 1.6% | 1.6% |
| Initial LR | 0.01 | 0.0141 |
| Final LR | 0.001 | 0.00141 |
| LR scheduler | Cosine | Cosine |
| Weight decay | 0.1 | 0.1 |
| Optimizer | **LAMB** | **LAMB** |
| LAMB β₁ | 0.9 | 0.9 |
| LAMB β₂ | 0.98 | 0.98 |
| Gradient clipping | 2.0 | 2.0 |

#### Training Corpus

1:1:1 mix of:
- BabyLM corpus (provided)
- FineWeb-Edu subset (high-quality educational web text)
- Cosmopedia subset (synthetic textbooks/stories)

Each corpus contributes different strengths: BabyLM → BLiMP (syntax), FineWeb-Edu → MNLI (understanding), Cosmopedia → EWOK (world knowledge).

---

### 1.2 ELC-BERT (Winner, BabyLM 2023 — Both Tracks)

**Paper**: "Not all layers are equally as important: Every Layer Counts BERT" (Charpentier & Samuel, BabyLM 2023)
**Code**: https://github.com/ltgoslo/elc-bert

#### Core Innovation: Learnable Layer Weights in Residual Connections

Standard transformers use uniform residual connections (equal weight to all layers). ELC-BERT replaces this with learned convex combinations:

**Original residual**: `h_in^n = h_out^{n-1} + h_in^{n-1}`
**ELC residual**: `h_in^n = Σ(i=0..n-1) α_i,n * h_out^i`

Each layer's input is a weighted sum of ALL previous layer outputs, with learnable α weights.

**Key findings from learned weights:**
- Early in pretraining: model is biased toward embedding layer and immediately preceding layers
- Late in pretraining: model reduces embedding reliance in favor of immediately preceding layers
- The embedding layer still gets more weight than a standard transformer
- Focus on previous layer for every layer; embedding layer important for first 5 and last layers
- Improved performance on (Super)GLUE, comparable on BLiMP

**Ablation results (ELC-BERT base):**

| Variant | BLiMP | Supp. | MSGS | GLUE |
|---------|-------|-------|------|------|
| ELC-BERT | 85.3 | 76.6 | -0.26±0.5 | 78.3±3.2 |
| + zero init | 84.9 | 78.5 | -0.38±0.3 | 79.4±1.0 |
| + normalization | 85.1 | 76.0 | -0.13±0.4 | 78.2±3.3 |
| + weighted output | **86.1** | 76.0 | -0.28±0.2 | 78.2±0.6 |

#### ELC-BERT Training Details (Small, Strict-Small Track)

| Hyperparameter | Value |
|---|---|
| Parameters | 24M |
| Layers | 12 |
| Hidden size | 384 |
| FF intermediate | 1,024 |
| Vocab size | 6,144 |
| Attention heads | 6 |
| Dropout | 0.1 |
| Training steps | 31,250 |
| Batch size | 512 |
| Sequence length | 128 |
| Warmup ratio | 1.6% |
| Learning rate | 0.005 |
| LR scheduler | Cosine |
| Weight decay | **0.4** |
| Optimizer | LAMB |
| Gradient clipping | 2.0 |

**Note**: The original ELC-BERT used a very high weight decay (0.4) and was trained for 2000 epochs. For isiXhosa experiments, 200 epochs with batch size 128 and LR 5e-4 was used, showing plateaus by epoch 200.

---

### 1.3 LTG-BERT (Architecture Foundation)

**Paper**: "Trained on 100 million words and still in shape: BERT meets British National Corpus" (Samuel et al., EACL 2023 Findings)
**Code**: https://github.com/ltgoslo/ltg-bert

This is the architectural backbone for both ELC-BERT and GPT-BERT. Key modifications over vanilla BERT:

#### Architecture Modifications

1. **NormFormer layer normalization**: Pre-norm style with parameter-free layer norms both before and after each sublayer (attention and FFN), improving training stability.

2. **Disentangled attention with relative positional encoding**: Position and token embeddings are treated separately in the attention mechanism (inspired by DeBERTa). Shared relative positional embeddings across all layers.

3. **GEGLU activation function**: Replaces GELU in feed-forward layers. GLU variants provide better expressivity:
   ```
   GEGLU(x) = GELU(xW + b) ⊗ (xV + c)
   ```

4. **No bias parameters in FFN layers**: Removes bias from feed-forward networks, reducing parameters with minimal performance impact.

5. **Gradual initialization scaling**: Initialize FFN layers with incrementally lower weight norms for training stability.

6. **Cosine LR decay** (vs. linear in original BERT): Performs better for low-resource training.

#### Ablation Results (LTG-BERT)

| Variant | MNLI | Edge/BLiMP |
|---------|------|------------|
| LTG-BERT (full) | 85.1±0.2 | 95.3±0.1 |
| w/ post-norm | -0.5 | -0.6 |
| w/ pre-norm | -1.3 | -0.2 |
| w/ GELU (not GEGLU) | -0.3 | 0.0 |
| w/ absolute pos. emb. | -1.0 | -0.1 |
| w/ standard masking (not span) | -0.3 | -0.5 |
| w/ linear LR decay | -0.2 | -0.1 |

---

## 2. Optimal Training Objectives

### 2.1 Objective Comparison for Small LMs

| Objective | Strengths | Best For | Notes |
|-----------|-----------|----------|-------|
| **MLM** (15-30% masking) | Rich bidirectional representations | NLU tasks, syntax | Standard for encoder models |
| **CLM** (causal) | Text generation, in-context learning | Generation tasks | Unidirectional context |
| **MNTP** (masked next-token) | Unifies MLM + CLM | Both NLU and NLG | Best of both worlds |
| **Span masking** | More difficult objective, better learning | General pretraining | Used in LTG-BERT |
| **Denoising** (BART-style) | Robust to noise | Seq2seq tasks | Token deletion, infilling, rotation |
| **ELECTRA-style RTD** | Compute-efficient | When compute-limited | Replace detection, not generation |
| **NSP/SOP** | Sentence-level understanding | (Often dropped) | RoBERTa showed it's not needed |

### 2.2 MNTP Implementation Details

```python
# Simplified MNTP training loop
def train_step(batch, model, is_causal):
    if is_causal:
        # Standard causal LM: predict next token
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        outputs = model(input_ids, attention_mask=causal_mask)
    else:
        # MNTP: mask some tokens, predict at shifted positions
        masked_input, masked_positions = apply_masking(batch, mask_ratio=0.15)
        input_ids = masked_input[:, :-1]  # shifted
        targets = batch[:, 1:]             # original shifted
        outputs = model(input_ids)         # bidirectional attention
    
    loss = cross_entropy(outputs.logits.view(-1), targets.view(-1))
    return loss

# Batch-level mixing: e.g., 1:15 causal-to-masked ratio
# For every 16 batches: 1 causal, 15 masked
```

### 2.3 Mask Scheduling

Start with 30% masking, linearly decrease to 15% over training:
```python
mask_ratio = 0.30 - (0.15 * current_step / total_steps)
mask_ratio = max(mask_ratio, 0.15)
```

This recovers more signal early when batches are small (due to batch-size scheduling), and better aligns final model with standard MLM usage.

---

## 3. Architecture Innovations

### 3.1 LTG-BERT Architecture Block

```
Input
  │
  ├─→ LayerNorm (no params) ─→ Disentangled Attention ─→ Gate ─→ LayerNorm (no params) ─→ Linear
  │         │                                                              │              │
  │         └──────────────────── Residual Connection ─────────────────────┘              │
  │                                                                                      │
  ├─→ LayerNorm (no params) ─→ Linear ─→ GEGLU ─→ LayerNorm (no params) ─→ Linear        │
  │         │                                                      │              │        │
  │         └──────────────────── Residual Connection ─────────────┘              │        │
  │                                                                              │        │
  └──────────────────────────────────────────────────────────────────────────────┘        │
                                                                                          │
  + Learned layer weights from all previous layers (ELC-BERT / GPT-BERT modification) ───┘
```

### 3.2 SmolLM / SmolLM2 Architecture (135M-360M)

**From HuggingFace's SmolLM2** (state-of-the-art for sub-500M):

| Feature | 135M | 360M |
|---------|------|------|
| Layers | 30 | 32 |
| Hidden size | 576 | 960 |
| Attention heads (Q) | 9 | 15 |
| KV heads (GQA) | 3 | 5 |
| Vocab size | 49,152 | 49,152 |
| Context length | 2,048 | 2,048 |
| Activation | SiLU (SwiGLU) | SiLU (SwiGLU) |
| Position encoding | RoPE | RoPE |
| Embedding tying | Yes | Yes |
| Training tokens | 2T | 4T |

**Key design principles:**
- **Depth over width**: Many thin layers outperform few wide layers for small models
- **GQA**: Grouped-Query Attention reduces KV cache size
- **Embedding tying**: Share input/output embeddings to save parameters
- **MobileLLM-inspired**: Similar to MobileLLM design philosophy

### 3.3 TinyLlama Architecture (1.1B, techniques applicable scaled down)

- Llama 2 decoder architecture
- RoPE, RMSNorm pre-norm, SwiGLU
- GQA (32 Q heads → fewer KV heads)
- Flash Attention 2
- FSDP (Fully Sharded Data Parallel)
- Trained on 300B tokens (~3 epochs of 950B token dataset)
- SlimPajama + Starcoder data (7:3 ratio)

### 3.4 Phi-1/1.5/2 Training Recipe

**Key insight**: "Textbook-quality" data is more important than data quantity.

1. **Synthetic data**: Create datasets specifically designed to teach reasoning, common sense, and knowledge
2. **Careful web filtering**: Select web data based on educational value and content quality
3. **Knowledge embedding**: Scale from smaller model (Phi-1.5 1.3B → Phi-2 2.7B) by embedding knowledge from the smaller model
4. **Data quality > data quantity**: Focus on diversity and educational value

### 3.5 Shakti Series (100M-500M, Edge-Optimized)

| Model | Layers | Hidden | GQA | Special |
|-------|--------|--------|-----|---------|
| Shakti-100M | 10 | 640 | Variable GQA | IoT/mobile |
| Shakti-250M | 16 | 1,024 | Variable GQA | Domain tasks |
| Shakti-500M | 24 | 2,048 | Block Sparse Attn | Multilingual |

Integrates RoPE and advanced attention variants for edge deployment.

---

## 4. Training Recipes & Hyperparameters

### 4.1 Optimizer Choice for Data-Limited Settings

**LAMB optimizer** is preferred for BabyLM-style settings (used by both GPT-BERT and ELC-BERT):
- Enables scaling batch sizes without accuracy loss
- Better for large-batch training in data-limited regimes
- β₁ = 0.9, β₂ = 0.98, ε = 1e-8

**AdamW** is the standard for most modern LLMs:
- β₁ = 0.9, β₂ = 0.95 (note: 0.95 not 0.999 for LLM-style training)
- ε = 1e-8
- Weight decay = 0.1 (standard) or 0.4 (aggressive, ELC-BERT)

### 4.2 Learning Rate Schedules

**Cosine annealing** (best for data-limited):
```python
lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * step / total_steps))
```
- Warmup: 1.6-5% of total steps
- Peak LR: 0.01 (LAMB) or 3e-4 to 6e-4 (AdamW)
- Min LR: 10% of peak (GPT-BERT) or ~0

**WSD (Warmup-Stable-Decay)** (used by SmolLM2):
- Warmup: linear ramp to peak
- Stable: constant peak LR for majority of training
- Decay: cosine decay in final 20% of training
- Peak LR: 3e-3 (SmolLM2)
- Good for long training runs where you don't know total steps upfront

**Cosine with restarts** (used in some BabyLM multi-phase setups):
- Cycle every N epochs
- Allows multiple "passes" at different LR regimes

### 4.3 Batch Size Strategy

**Batch-size scheduling** (GPT-BERT):
- Start at 1/4 of max batch size
- Linearly increase over training
- Max = 4M tokens per step
- Rationale: early steps don't need precise gradients; saves compute

**Standard approach**:
- 128-512 samples per step for small models
- Global batch sizes of 0.5M-4M tokens

### 4.4 Sequence Length Strategy

**Progressive length increase** (GPT-BERT):
- Start at 128 tokens
- Gradually increase to 512 tokens
- Allows more training steps with same token budget
- Model learns local patterns first, then global

### 4.5 Weight Decay

| Setting | Weight Decay | Notes |
|---------|-------------|-------|
| Standard LLM | 0.1 | Most common |
| ELC-BERT | **0.4** | Very aggressive, works for data-limited |
| GPT-BERT | 0.1 | Standard |
| SmolLM2 | 0.1 | Standard |

Higher weight decay (0.3-0.4) is worth trying for severely data-limited settings.

---

## 5. Data Strategies

### 5.1 Data Mixing (GPT-BERT Approach)

Mix data sources at 1:1:1 ratio:
- **BabyLM corpus** → best for syntax/grammar (BLiMP)
- **FineWeb-Edu** → best for NLU (MNLI/GLUE)
- **Cosmopedia** → best for world knowledge (EWOK)

**FineWeb-Edu**: 1.3T tokens of educational text filtered from Common Crawl using an educational quality classifier. Dramatically improves knowledge and reasoning benchmarks.

**Cosmopedia v2**: 25B+ tokens of synthetic textbooks and stories generated by LLMs. Covers hundreds of subjects with <1% duplicate rate. Largest open synthetic dataset.

### 5.2 Data Quality Over Quantity

From Phi models and SmolLM2:
- **Educational value filtering**: Score web data for educational quality, keep only high-scoring samples
- **Deduplication**: Aggressive exact and fuzzy deduplication (critical for small datasets)
- **Synthetic augmentation**: Use LLMs to generate "textbook-quality" training data
- **Domain balance**: Ensure coverage of different domains (science, daily life, reasoning, theory of mind)

### 5.3 Data Augmentation for LM Pretraining

**Rule-based augmentation:**
- Synonym replacement
- Random word insertion, deletion, swapping (EDA, α=0.1)
- Character-level noise injection

**Back-translation:**
- Translate text → pivot language → back to original
- Creates diverse paraphrases while preserving meaning
- Effective for low-resource languages
- Best pivot languages: those with well-developed MT (e.g., Russian, French for English)

**LM-based augmentation:**
- Use a larger model to generate paraphrases
- Generate synthetic "textbook" content (Cosmopedia approach)
- LLM2LLM: iteratively identify hard samples and generate more for those

**Data augmentation for pretraining (not just finetuning):**
- BPE dropout (0.1): randomly drop merges during tokenization → different segmentations of same text
- This is essentially free data augmentation at the tokenizer level

### 5.4 Data Selection

**Principled Data Selection (PDS)** (ICLR 2025):
- Select high-quality pre-training data using optimal control
- Training on 1/4 of data selected by PDS for 4 epochs achieves best results vs. full data for 1 epoch
- Quality > quantity for data-constrained settings

---

## 6. Knowledge Distillation

### 6.1 MiniPLM (ICLR 2025)

**Key innovation**: Offline KD that refines the training data distribution using the teacher's knowledge.

**Advantages:**
- **Efficient**: Offline teacher inference — no additional training cost per student
- **Flexible**: Operates on training corpus only, enables KD across model families (e.g., Qwen → Mamba, LLaMA)
- **Effective**: Enhances training data difficulty and diversity based on differences between large and small LMs

**Implementation**: Teacher processes the corpus to create a refined data distribution that is harder and more diverse, then student trains on this refined distribution.

### 6.2 BabyLlama Approach (BabyLM 2023)

Knowledge distillation from an ensemble of teachers:
- Multiple teacher models vote on soft targets
- Student trained on small dataset with ensemble soft targets
- Outperforms teachers trained from scratch on small data

### 6.3 General KD Strategies for Small LMs

| Method | Type | Cost | Effectiveness |
|--------|------|------|--------------|
| **Soft targets** (logits matching) | White-box | Medium | High |
| **Feature matching** (hidden states) | White-box | Medium | Medium |
| **Synthetic data generation** | Black-box | Low | High |
| **MiniPLM data refinement** | Hybrid | Low (offline) | High |
| **Curriculum distillation** (L2M-KD) | White-box | Medium | High |

### 6.4 Practical KD Recipe for BabyLM

1. Train or use a larger teacher model (e.g., 1B+ params)
2. Generate soft targets on the BabyLM corpus
3. Train student on KL divergence between student and teacher logits
4. Optionally use attention transfer for intermediate layers
5. Mix KD loss with standard LM loss (e.g., α=0.5 KD + 0.5 standard)

---

## 7. Advanced Regularization

### 7.1 Dropout Strategy

For data-limited small models:
- **Hidden dropout**: 0.1 (standard, used by GPT-BERT and ELC-BERT)
- **Attention dropout**: 0.1 (standard)
- **Embedding dropout**: 0.1 (optional)
- Some work suggests reducing dropout to very small values when pretraining for limited steps (overfitting risk is low early in training)
- Increase dropout in later training epochs if overfitting is observed

### 7.2 Weight Decay

- **AdamW weight decay**: 0.1 (standard) to 0.4 (aggressive, ELC-BERT)
- Decoupled weight decay (don't apply to bias, LayerNorm, embedding parameters)
- Higher values (0.3-0.4) work well for severely data-constrained settings

### 7.3 Label Smoothing

- Standard value: 0.1
- Helps prevent overconfident predictions
- More important for small models with limited data
- Apply to both MLM and CLM objectives

### 7.4 Embedding Tying

Share input embedding matrix with output LM head:
- Reduces parameters significantly (vocab × hidden can be large)
- Used by SmolLM, GPT-2, and most modern small LMs
- Particularly important for sub-100M models where embeddings are a large fraction of parameters

### 7.5 BPE Dropout

- Randomly drop BPE merges during tokenization (p=0.1)
- Creates different subword segmentations of the same text each epoch
- Effectively free data augmentation
- Anneal to 0 near end of training

### 7.6 Gradient Clipping

- Standard: 1.0 (AdamW) or 2.0 (LAMB)
- Prevents gradient explosion in early training
- More important for small models with high learning rates

---

## 8. RoPE Improvements

### 8.1 NTK-Aware Scaling

Modifies the RoPE base frequency to preserve high-frequency information when extending context:

```python
# Original: base = 10000
# NTK-aware: base = 10000 * (scale ^ (dim / (dim - 2)))
def ntk_aware_frequencies(dim, base=10000, scale=1.0):
    ntk_base = base * (scale ** (dim / (dim - 2)))
    return compute_rope_frequencies(dim, ntk_base)
```

**Key property**: Higher frequencies (low dimensions) are preserved → better local pattern recognition. Lower frequencies are scaled → extends context.

**For small models**: Use a slightly higher base (e.g., 50000 instead of 10000) even for standard context lengths to improve extrapolation.

### 8.2 YaRN (Yet another RoPE extensioN, ICLR 2024)

Builds on NTK-aware scaling with "NTK-by-parts" interpolation:
- Applies different scaling strategies per frequency band
- High frequencies (low dims): linear interpolation (preserve local patterns)
- Low frequencies (high dims): NTK scaling (extend long-range coherence)
- Adds attention temperature scaling: scale attention logits by 1/√(scale_factor)

```python
# YaRN simplified
for dim in range(d):
    wavelength = 2 * π / θ_dim
    if wavelength < low_freq_wavelength:
        # No scaling (preserve high frequencies)
        freq[dim] = original_freq[dim]
    elif wavelength > high_freq_wavelength:
        # NTK scaling (extend low frequencies)
        freq[dim] = original_freq[dim] / scale
    else:
        # Smooth interpolation
        freq[dim] = interpolated_freq(dim, scale)

# Attention scaling
attention_scores /= sqrt(scale_factor)
```

**Practical for BabyLM**: If you need context > 512 tokens, apply YaRN with scale factor = target_length / training_length.

### 8.3 Dynamic NTK Scaling

Adjusts RoPE frequencies dynamically based on actual sequence length at inference:
- No fine-tuning required
- Good for models that see variable-length inputs
- Performance degrades beyond 2× training length

### 8.4 Practical RoPE Recommendations for Small LMs

- **Training**: Use RoPE with base=10000 (standard) or 50000 (SmolLM2)
- **Context extension**: Apply NTK-aware or YaRN if you need longer contexts
- **Critical dimensions**: Research (NeurIPS 2024) shows RoPE base bounds context length — higher base → longer achievable context
- **Start short, extend later**: Pretrain with 128-512 tokens, then extend via YaRN/NTK fine-tuning

---

## 9. Multi-Stage Training Pipelines

### 9.1 GPT-BERT / LTG-BERT Pipeline (Recommended for BabyLM)

```
Stage 1: Short-context pretraining
├── Sequence length: 128
├── Small batch size (1M tokens)
├── High masking ratio (30%)
├── High learning rate (0.01-0.014)
└── Warmup: 1.6% of steps

Stage 2: Full-context pretraining
├── Sequence length: 512
├── Full batch size (4M tokens)
├── Standard masking (15%)
├── Cosine decay from peak LR
└── Continue until convergence
```

Both stages use the hybrid CLM+MNTP objective with 1:15 causal-to-masked ratio.

### 9.2 SmolLM2 Pipeline (For larger token budgets)

```
Stage 1: Pretraining (2-4T tokens)
├── Single-stage with high-quality filtered data
├── WSD scheduler (warmup-stable-decay)
├── Peak LR: 3e-3, decay in final 20%
├── AdamW (β₂=0.95), weight decay 0.1
├── FineWeb-Edu + DCLM filtered + Stack-Edu + Math data
└── Context length: 2048

Stage 2: SFT (Supervised Fine-Tuning)
├── SmolTalk dataset (instruction-following)
├── 1 epoch, LR 3e-4
└── Filtered for model capacity

Stage 3: DPO (Direct Preference Optimization)
├── UltraFeedback / HelpSteer
├── 1 epoch
└── Optimizes for helpfulness
```

### 9.3 Multi-Round Training (Pangu Approach, from SLM Survey)

For small models specifically:
1. **Round 1**: Train on full data, record per-sample losses
2. **Round 2**: Resample data with higher probability for hard examples (50% sampling rate)
3. Optionally repeat for Round 3

**Result**: Two rounds with 50% hard example resampling is the sweet spot between performance and efficiency.

### 9.4 Curriculum Learning Approaches

**Active Curriculum Language Modeling (ACLM)** (BabyLM 2024 submission):
- Uses a surprisal oracle to rank training samples by difficulty
- Feeds easy examples first, progressively harder
- Based on ELC-BERT backbone
- Mixed results — curriculum learning has shown inconsistent benefits in BabyLM

**Domain ordering**: Train on simpler domains first (children's books) → more complex (Wikipedia, books)

**Length ordering**: Start with shorter sequences → progressively longer

---

## 10. Implementation Recommendations

### 10.1 Recommended Architecture for BabyLM-style Challenges

Based on all the research, the recommended architecture combines insights from GPT-BERT, LTG-BERT, and SmolLM2:

```yaml
Architecture:
  type: "LTG-BERT backbone with hybrid CLM+MNTP"
  hidden_size: 768          # 119M params for strict; 384 for strict-small
  num_layers: 12
  num_attention_heads: 12   # 6 for strict-small
  num_kv_heads: 4           # GQA for efficiency
  intermediate_size: 2560   # ~3.3x hidden for SwiGLU
  activation: "GEGLU"       # or SwiGLU for decoder
  normalization: "RMSNorm"  # or parameter-free LayerNorm
  position_encoding: "RoPE" # base=10000-50000
  vocab_size: 16384         # 8192 for strict-small
  dropout: 0.1
  attention_gate: true
  embedding_tying: true
  learned_layer_weights: true # ELC-BERT style

Training:
  objective: "Hybrid CLM+MNTP (1:15 ratio)"
  optimizer: "LAMB"
  peak_lr: 0.01
  min_lr_ratio: 0.1
  lr_scheduler: "cosine"
  warmup_ratio: 0.016
  weight_decay: 0.1         # try 0.4 if overfitting
  gradient_clipping: 2.0
  batch_size_schedule: "linear (1M→4M tokens)"
  mask_schedule: "linear (30%→15%)"
  sequence_length_schedule: "128→512"
  precision: "bf16"
  
Data:
  sources: "BabyLM + FineWeb-Edu + Cosmopedia (1:1:1)"
  augmentation: "BPE dropout (0.1, annealed)"
  deduplication: "aggressive exact + fuzzy"
```

### 10.2 Quick Wins (High Impact, Low Effort)

1. **Switch from pure CLM to hybrid CLM+MNTP** (1:15 ratio) — biggest single improvement
2. **Add attention gating** — small code change, consistent improvement
3. **Use GEGLU/SwiGLU activation** instead of GELU — free performance
4. **Use LAMB optimizer** instead of AdamW for data-limited settings
5. **Increase weight decay to 0.3-0.4** if overfitting
6. **Add mask scheduling** (30%→15%) — helps with early training stability
7. **Use batch-size scheduling** — 2× training efficiency for free
8. **Mix in high-quality external data** (FineWeb-Edu, Cosmopedia) if allowed

### 10.3 Medium Effort Improvements

1. **Learned layer weights** (ELC-BERT) — 1.5× slowdown but better performance
2. **Knowledge distillation** from a larger teacher model
3. **Multi-round training** with hard example resampling
4. **Data quality filtering** (educational value classifier)
5. **Synthetic data augmentation** (back-translation, LLM paraphrasing)

### 10.4 Key Hyperparameter Reference

| Setting | BabyLM Strict (100M words) | BabyLM Strict-Small (10M words) |
|---------|---------------------------|-------------------------------|
| Params | ~120M | ~30M |
| Hidden | 768 | 384 |
| Layers | 12 | 12 |
| Heads | 12 | 6 |
| Vocab | 16,384 | 8,192 |
| LR (peak) | 0.01 | 0.014 |
| Warmup | 1.6% | 1.6% |
| Steps | ~15,000 | ~8,000 |
| Batch (tokens) | 1M→4M | 1M→4M |
| Seq length | 128→512 | 128→512 |
| Optimizer | LAMB | LAMB |
| Weight decay | 0.1 | 0.1 |

---

## References

### BabyLM-Specific Papers
- Charpentier & Samuel (2024). "GPT or BERT: why not both?" CoNLL-BabyLM 2024.
- Charpentier & Samuel (2023). "Not all layers are equally as important: ELC-BERT." BabyLM 2023.
- Samuel et al. (2023). "Trained on 100 million words and still in shape: BERT meets BNC." EACL 2023 Findings.
- Behr (2024). "ELC-ParserBERT: Low-Resource Language Modeling Utilizing a Parser Network With ELC-BERT." CoNLL-BabyLM 2024.
- Warstadt et al. (2023). "Findings of the BabyLM Challenge." CoNLL-BabyLM 2023.

### Architecture & Training
- BehnamGhader et al. (2024). "LLM2Vec." (MNTP objective)
- Shazeer (2020). "GLU Variants Improve Transformer." (GEGLU)
- Jumper et al. (2021). "Highly accurate protein structure prediction." (Attention gating)
- Pagliardini et al. (2024). "DenseFormer." (Layer weight generalization)

### Small LM Models
- Ben Allal et al. (2024). "SmolLM / SmolLM2." Hugging Face.
- Zhang et al. (2024). "TinyLlama: An Open-Source Small Language Model."
- Li et al. (2023). "Textbooks Are All You Need." (Phi-1)
- MobileLLM (2024). "MobileLLM: Optimizing Sub-billion Parameter Language Models."

### RoPE & Context Extension
- Peng et al. (2024). "YaRN: Efficient Context Window Extension of Large Language Models." ICLR 2024.
- bloc97 (2023). "NTK-Aware Scaled RoPE."
- Chen et al. (2023). "Position Interpolation."

### Knowledge Distillation
- Gu et al. (2025). "MiniPLM: Knowledge Distillation for Pre-Training Language Models." ICLR 2025.
- Timiryasov & Tastet (2023). "Baby Llama: Knowledge Distillation from an Ensemble of Teachers."

### Data
- Lozhkov et al. (2024). "FineWeb / FineWeb-Edu." NeurIPS 2024.
- Ben Allal et al. (2024). "Cosmopedia."
- Gu et al. (2025). "Data Selection via Optimal Control for Language Models." ICLR 2025 Oral.
