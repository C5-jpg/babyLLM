# V1-V14 Standardized Evaluation Results

Generated: 2026-05-12T01:35:00+08:00

## Summary

| Rank | Version | Params | Best PPL | Loss | Best Stage | PPL/10M | Status |
|------|---------|--------|----------|------|------------|---------|--------|
| 1 | V13 | 94.2M | **38.68** | 3.6554 | Stage 2 EMA | 4.11 | ✅ Complete |
| 2 | V12 | 54.2M | **38.84** | 3.6595 | Stage 2 EMA | 7.17 | ✅ Complete |
| 3 | V11 | 38.7M | **40.72** | 3.7068 | Stage 5 EMA | 10.52 | ✅ Complete |
| 4 | V10 | 38.7M | **42.89** | 3.7585 | Stage 3 | 11.08 | ✅ Complete |
| 5 | V14 | 52M | 44.12† | 3.7870 | Stage 2 | 8.48 | ⚠️ Incomplete |
| 6 | V7 | 35M | 50.84 | 3.9286 | best_model | 14.53 | ✅ Complete |
| 7 | V8 | 35M | 50.84 | 3.9286 | Stage 3 | 14.53 | ✅ Complete |
| 8 | V9 | 35M | 50.84 | 3.9288 | Polish Probe | 14.53 | ✅ Complete |
| - | V1 | 110M | TBD | - | best_model | - | Pending re-eval |
| - | V2 | 125M | TBD | - | best_model | - | Pending re-eval |
| - | V4 | 350M | TBD | - | best_model | - | Pending re-eval |
| - | V5 | 51M | TBD | - | best_model | - | Pending re-eval |
| - | V6 | 75M | TBD | - | Stage 3 | - | Pending re-eval |
| - | V3 | 125M | N/A | N/A | N/A | N/A | ❌ Failed |

†V14 stages 3-5 training in progress

## Per-Version Best Results

### V13 — Current SOTA
- **Model**: `/mnt/sda/kehe/babyllm_output/babylm-v13/stage2_mntp/best_model_ema`
- **Architecture**: LLaMA 768d/14L/12Q/4KV GQA, 94.2M params
- **PPL**: 38.68 (loss=3.6554)
- **Tokens evaluated**: 4,886,528

### V12 — Best Efficiency
- **Model**: `/mnt/sda/kehe/babyllm_output/babylm-v12/stage2_mntp/best_model_ema`
- **Architecture**: LLaMA 576d/14L/9Q/3KV GQA, 54.2M params
- **PPL**: 38.84 (loss=3.6595)
- **PPL/10M params**: 7.17 (best efficiency)

### V11 — EMA + Self-Distillation
- **Model**: `/mnt/sda/kehe/babyllm_output/babylm-v11/stage5_self_distill/best_model_ema`
- **PPL**: 40.72 (loss=3.7068)
- **7-stage pipeline**: CLM→MNTP→Polish→Annealing→Self-Distill→SWA→Eval

### V10 — Production Pipeline
- **Model**: `/mnt/sda/kehe/babyllm_output/babylm-v10/stage3_polish/best_model`
- **PPL**: 42.89 (loss=3.7585)
- **3-stage pipeline**: CLM→MNTP→Polish

### V14 — Efficiency Build (In Progress)
- **Current best**: PPL=44.12 (Stage 2 only)
- **Training**: Stage 3 Polish in progress as of 2026-05-12
- **Target**: PPL < 42 with 5-stage pipeline

### V7/V8/V9 — Early MNTP Models
- All achieved PPL ~50.8 with 35M params and 8K vocab
- V7 introduced MNTP hybrid training
- V8 added 3-stage pipeline
- V9 added probe experiments

## Stage-by-Stage Analysis (V10-V13)

### V10 Stages
| Stage | Loss | PPL | Improvement |
|-------|------|-----|-------------|
| Stage 1 CLM | 4.0183 | 55.60 | baseline |
| Stage 2 MNTP | 3.9255 | 50.68 | -4.92 PPL |
| Stage 3 Polish | 3.7585 | 42.89 | -7.79 PPL |

### V11 Stages
| Stage | Loss | PPL | EMA PPL |
|-------|------|-----|---------|
| Stage 1 CLM+SGDR | 3.8339 | 46.24 | 43.07 |
| Stage 2 MNTP | 3.7099 | 40.85 | 40.84 |
| Stage 3 Polish | 3.7080 | 40.77 | 40.77 |
| Stage 4 Annealing | 3.7075 | 40.75 | 40.75 |
| Stage 5 Self-Distill | 3.7068 | 40.73 | **40.72** |
| SWA | 3.7071 | 40.74 | - |

### V12 Stages
| Stage | Loss | PPL | EMA PPL |
|-------|------|-----|---------|
| Stage 1 CLM+SGDR | 3.7335 | 41.83 | 39.36 |
| Stage 2 MNTP | 3.6627 | 38.96 | **38.84** |
| Stage 3 Polish | 3.6640 | 39.02 | 38.99 |
| Stage 4 Self-Distill | 3.6646 | 39.04 | 39.03 |
| Stage 5 Annealing | 3.6646 | 39.04 | 39.04 |

### V13 Stages
| Stage | Loss | PPL | EMA PPL |
|-------|------|-----|---------|
| Stage 1 CLM+SGDR | 3.7647 | 43.15 | 39.51 |
| Stage 2 MNTP | 3.7245 | 41.45 | **38.68** |
| Stage 3 Polish | 3.6989 | 40.40 | 40.08 |

## Key Observations

1. **EMA is most valuable in Stage 1** (high LR, high noise): V12 Stage 1 EMA improves 2.47 PPL over non-EMA
2. **Stage 2 MNTP is the core improvement stage** across all versions
3. **Stage 3+ shows diminishing returns**: V12 stages 3-5 actually worsened PPL slightly
4. **V13 Stage 3 was negative**: DropBlock/StochDepth over-regularized, EMA PPL worsened from 38.68 to 40.08
5. **V14 is on track**: Stage 2 PPL=44.12 with smaller model (52M), stages 3-5 may improve

## Detailed Data

Full evaluation JSONs with generation samples available at:
- `/mnt/sda/kehe/babyllm_output/babylm-v{10,11,12,13}/eval_stage*.json`
- `/home/kehe/babyllm/babyLLM/logs/standardized_eval_results.json`
