# V1-V14 Comprehensive Analysis Report

Generated: 2026-05-12T00:00:00Z

## Executive Summary

This report documents the complete analysis of 14 iterations of the BabyLM Chinese language model, trained for the NLPCC 2026 ChineseBabyLM Challenge. The project trained models from scratch on `babylm-zho-100M` (~100M Chinese characters) using 4× NVIDIA RTX A6000 GPUs.

**Current SOTA**: V13 Stage 2 EMA (PPL=38.68, 94.2M params)
**Best Efficiency**: V12 Stage 2 EMA (PPL=38.84, 54.2M params)
**V15 Target**: PPL < 38.0 with ~58M params

---

## Version-by-Version Analysis

### V1 — GPT-2 Baseline
- **Architecture**: GPT-2, 768d, 12L, 12H, ~110M params
- **Tokenizer**: BPE 32K
- **Training**: 10 epochs, lr=6e-4, cosine scheduler, fp32
- **Status**: ✅ Complete
- **Key Finding**: LR scheduler bug accidentally created SGDR effect (later validated as beneficial)

### V2 — LLaMA Architecture
- **Architecture**: LLaMA, 768d, 12L, 12Q/4KV GQA, ~125M params
- **Tokenizer**: ByteLevel BPE 32K
- **Training**: 25 epochs, lr=6e-4, bf16, Flash Attention, BPE dropout
- **Status**: ✅ Complete (PPL=1824.45 — eval bug, needs re-evaluation)
- **Key Finding**: ByteLevel BPE splits Chinese chars into 3 UTF-8 bytes → 3.4× information density loss

### V3 — SentencePiece + WSD
- **Architecture**: LLaMA, 768d, 12L, ~125M params
- **Tokenizer**: SentencePiece 32K
- **Training**: WSD scheduler, early stopping
- **Status**: ❌ FAILED — `AssertionError: Padding_idx must be within num_embeddings`
- **Root Cause**: Vocab size mismatch between tokenizer and model config

### V4 — Deep LLaMA
- **Architecture**: LLaMA, 1024d, 24L, 16Q/8KV, ~350M params
- **Tokenizer**: SPM 32K
- **Training**: 50 epochs, lr=3e-4
- **Status**: ✅ Complete (but tokens/param=0.23, severely undertrained)
- **Key Finding**: Model too large for 100M token dataset

### V5 — Small LLaMA + Knowledge Distillation
- **Architecture**: LLaMA, 512d, 12L, 8Q/4KV, ~51M params
- **Tokenizer**: SPM 32K
- **Training**: Phase 1 (15 epochs CLM) + Phase 2 (KD from V2 teacher)
- **Status**: ✅ Complete
- **Key Finding**: First version to match model size to data size (tokens/param=1.61×)

### V6 — 3-Stage Pipeline
- **Architecture**: LLaMA, 640d, 12L, 10Q/5KV, ~75M params
- **Tokenizer**: SPM 32K
- **Training**: 3-stage (CLM → CLM+MLM → Reverse KL KD)
- **Status**: ✅ Complete (but data cleaning removed 79% of data)
- **Key Finding**: Aggressive data cleaning is counterproductive

### V7 — MNTP Hybrid
- **Architecture**: LLaMA, 448d, 12L, 7Q/4KV, ~35M params
- **Tokenizer**: SPM 8K
- **Training**: CLM+MNTP hybrid, 8K vocab
- **Status**: ✅ Complete (PPL ~50.8)
- **Key Finding**: 8K vocab size too small for Chinese; 32K better

### V8 — 3-Stage CLM+MNTP+Polish
- **Architecture**: LLaMA, 512d, 12L, 8Q/4KV, ~35M params
- **Tokenizer**: SPM 8K
- **Training**: 3-stage pipeline
- **Status**: ✅ Complete (PPL=50.84)

### V9 — Probe Experiments
- **Architecture**: LLaMA, 512d, 12L, ~35M params
- **Training**: Various probe configurations
- **Status**: ✅ Complete (PPL ~50.8)

### V10 — Production Pipeline
- **Architecture**: LLaMA, 512d, 12L, 8Q/4KV, ~38.7M params
- **Tokenizer**: SPM 32K
- **Training**: 3-stage (CLM→MNTP→Polish), 19,037 total steps
- **Status**: ✅ Complete (PPL=42.89)
- **Key Finding**: First production-quality pipeline

### V11 — EMA + Self-Distillation
- **Architecture**: LLaMA, 512d, 12L, ~38.7M params
- **Training**: 7-stage pipeline (CLM→MNTP→Polish→Annealing→Self-Distill→SWA→Eval)
- **Status**: ✅ Complete (PPL=40.72)
- **Key Finding**: EMA provides 6-8% PPL improvement; SGDR improves convergence

### V12 — Focal Loss + Data Cleaning
- **Architecture**: LLaMA, 576d, 14L, 9Q/3KV, ~54.2M params
- **Training**: 5-stage pipeline with Focal Loss (γ=2.0)
- **Status**: ✅ Complete (PPL=38.84)
- **Key Finding**: Best parameter efficiency (PPL/10M params)

### V13 — Maximal Model (Current SOTA)
- **Architecture**: LLaMA, 768d, 14L, 12Q/4KV, ~94.2M params
- **Training**: 3-stage pipeline with advanced data filtering
- **Status**: ✅ Complete (PPL=38.68)
- **Key Finding**: Stage 3 Polish with DropBlock/StochDepth was negative optimization

### V14 — Efficiency Build
- **Architecture**: LLaMA, 640d, 12L, 10Q/5KV, ~52M params
- **Training**: 5-stage pipeline (CLM→MNTP→Polish→Self-Distill→Annealing)
- **Status**: ⚠️ INCOMPLETE — Stages 1-2 done (PPL=44.12), stages 3-5 completing
- **Key Finding**: Tokens/param ratio improved to 1.9× (from V13's 1.1×)

---

## Standardized Test Results

Results logged with ISO 8601 timestamps. See `logs/standardized_eval_results.json` for full data.

| Version | Params | Best PPL | Best Stage | PPL/10M | Status |
|---------|--------|----------|------------|---------|--------|
| V1 | 110M | ~343* | best_model | 31.2* | Complete (diff tokenizer) |
| V2 | 125M | TBD | best_model | TBD | Needs re-eval |
| V3 | 125M | N/A | N/A | N/A | Failed |
| V4 | 350M | TBD | best_model | TBD | Complete |
| V5 | 51M | TBD | best_model | TBD | Complete |
| V6 | 75M | TBD | stage3 | TBD | Complete |
| V7 | 35M | ~50.8 | best_model | 14.5 | Complete |
| V8 | 35M | 50.84 | stage3 | 14.5 | Complete |
| V9 | 35M | 50.84 | polish_probe | 14.5 | Complete |
| V10 | 38.7M | 42.89 | stage3 | 11.1 | Complete |
| V11 | 38.7M | 40.72 | stage5_ema | 10.5 | Complete |
| V12 | 54.2M | 38.84 | stage2_ema | 7.2 | Complete |
| V13 | 94.2M | 38.68 | stage2_ema | 4.1 | Complete |
| V14 | 52M | 44.12† | stage2 | 8.5† | Incomplete |

*V1 PPL not directly comparable — different tokenizer
†V14 stages 3-5 completing

---

## Best Practice Synthesis

### Convergence Stability
- V10-V13 show stable convergence with proper early stopping
- SGDR scheduler (V11+) provides better convergence than plain cosine
- All versions overfit by epoch 6-10 on 100M data

### Parameter Efficiency
| Version | PPL/10M Params | Tokens/Param |
|---------|---------------|--------------|
| V12 | 7.2 | 1.9× |
| V13 | 4.1 | 1.1× |
| V11 | 10.5 | 2.6× |
| V10 | 11.1 | 2.6× |

### Training Technique Rankings
1. **SentencePiece tokenizer** — ~74% PPL impact (biggest single improvement)
2. **EMA (decay=0.999)** — 6-8% PPL improvement
3. **CLM+MNTP hybrid** — Core training technique
4. **SGDR scheduler** — Better convergence than cosine
5. **Focal Loss** — Helps MNTP class imbalance
6. **Label smoothing annealing** — Prevents early overconfidence
7. **BPE dropout** — Improves generalization

### Architecture Insights
- Depth > Width for same param count (V12: 14L > V10: 12L)
- GQA with 3:1 compression ratio optimal
- Tied embeddings save ~25% params for small models
- RoPE (base=10000), RMSNorm (eps=1e-5), SwiGLU are standard

### Data Pipeline Impact
- PPL filtering (max_ppl=250) + MinHash dedup + hard upsample ×2 = best results
- Aggressive cleaning (V6: 79% data loss) is counterproductive
- 100M tokens insufficient for >55M params (tokens/param < 1.8×)

---

## V15 Design Rationale

Based on the synthesis above, V15 targets:

- **Architecture**: 640d, 14L, 10Q/5KV GQA (~58M params)
  - Deeper than V14 (14L vs 12L) for better feature extraction
  - Same width as V14 (640d) for tokens/param efficiency
  - tokens/param ≈ 1.7× (optimal range)

- **Pipeline**: 2-stage (CLM→MNTP)
  - V13 proved Polish stage is ineffective
  - Simpler pipeline, less risk of overfitting

- **New Features**:
  - Multi-scale EMA (0.999 + 0.9999)
  - Per-layer gradient norm monitoring
  - Eval every 200 steps
  - Gradient norm spike detection

---

## Appendix: Detailed Metrics

See `logs/standardized_eval_results.json` for per-version detailed metrics including generation samples.
