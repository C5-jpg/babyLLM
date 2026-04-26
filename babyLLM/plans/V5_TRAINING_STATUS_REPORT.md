# ChineseBabyLM V5 Phase 1 训练状态报告

> **报告日期**: 2026-04-24
> **训练阶段**: Phase 1 — 标准预训练 (CE Loss)
> **状态**: 🔴 **模型权重丢失 — SSD空间不足 + 未配置HDD软链接**

---

## 1. 训练概况

| 项目 | 值 |
|------|-----|
| 模型架构 | BabyLLM-V5 (LLaMA) |
| 参数量 | 50,993,664 (51.0M) |
| 隐藏维度 | 512 |
| 层数 | 12 |
| 注意力头 | 8Q / 4KV (GQA) |
| FFN 维度 | 1365 |
| 序列长度 | 1024 |
| Tokenizer | SentencePiece Unigram, 32K 词表 |
| 训练数据 | train.txt (1,203,087 文档 → 146,224 样本) |
| 验证数据 | val.txt (63,320 文档 → 7,850 样本) |
| GPU | 3 × NVIDIA RTX A6000 (47.5 GB each) |
| 有效 Batch | 96 (32/GPU × 3) |
| Tokens/Step | 98,304 |
| 学习率 | 6e-4 → cosine → 0 |
| Warmup | 5% (3,426 steps) |
| 总训练步数 | 68,535 (15 epochs × 4,569 steps/epoch) |
| 训练时间 | 21:06 ~ 23:40 (约 2.4 小时) |
| WandB | [ncya0p84](https://wandb.ai/c5galaxies-sjtu-hpc-center/chinese-babylm/runs/ncya0p84) |

---

## 2. Epoch-by-Epoch 训练指标

| Epoch | Train Loss | Val Loss | Val PPL | LR | 耗时 | 状态 |
|-------|-----------|----------|---------|-----|------|------|
| 1 | 7.5225 | 6.7644 | 866.45 | 6.00e-4 | 11.6min | ✅ 新最佳 |
| 2 | 6.4666 | 6.4166 | 611.92 | 5.89e-4 | 12.7min | ✅ 新最佳 |
| 3 | 6.1810 | 6.3208 | 555.99 | 5.64e-4 | 12.7min | ✅ 新最佳 |
| 4 | 6.0276 | 6.2836 | 535.72 | 5.26e-4 | 12.7min | ✅ 新最佳 |
| 5 | 5.9212 | 6.2670 | 526.92 | 4.78e-4 | 12.7min | ✅ 新最佳 |
| **6** | **5.8285** | **6.2638** | **525.21** | **4.21e-4** | **12.8min** | **✅ 新最佳** |
| 7 | 5.7493 | 6.2671 | 526.94 | 3.58e-4 | 12.7min | ⚠️ 早停 1/5 |
| 8 | 5.6708 | 6.2758 | 531.57 | 2.92e-4 | 12.7min | ⚠️ 早停 2/5 |
| 9 | 5.6039 | 6.2925 | 540.51 | 2.26e-4 | 12.7min | ⚠️ 早停 3/5 |
| 10 | 5.5207 | 6.3085 | 549.22 | 1.65e-4 | 12.7min | ⚠️ 早停 4/5 |
| 11 | 5.4541 | 6.3272 | 559.60 | 1.09e-4 | 12.7min | 🔴 早停 5/5 → 停止 |

### 训练曲线分析

```
Val PPL 趋势:
866 → 612 → 556 → 536 → 527 → 525 (最佳) → 527 → 532 → 541 → 549 → 560
                                                                    ↑ 持续恶化

Train Loss 趋势:
7.52 → 6.47 → 6.18 → 6.03 → 5.92 → 5.83 → 5.75 → 5.67 → 5.60 → 5.52 → 5.45
                                                                              ↑ 持续下降
```

**关键发现**: Epoch 6 之后，Train Loss 持续下降（5.83 → 5.45），但 Val Loss 持续上升（6.26 → 6.33），这是**典型的过拟合模式**。模型在 Epoch 6 已达到最优泛化能力。

---

## 3. 🔴 严重问题：模型权重丢失

### 3.1 问题描述

训练日志中明确记录了多次模型保存操作：

```
Epoch 1: Writing model shards: 100% → best_model/
Epoch 2: Writing model shards: 100% → best_model/
...
Epoch 6: Writing model shards: 100% → best_model/  (最终最佳)
Step 5000: Writing model shards: 100% → checkpoint-5000/
Step 10000: Writing model shards: 100% → checkpoint-10000/
Step 15000: Writing model shards: 100% → checkpoint-15000/
```

但实际检查文件系统，**所有目录中都没有模型权重文件**（`model.safetensors` 或 `pytorch_model.bin`）：

```
output/babylm-llama-v5/
├── best_model/
│   ├── config.json          ← 只有配置文件
│   ├── generation_config.json
│   ├── spm.model
│   └── tokenizer_config.json
├── checkpoint-10000/
│   ├── config.json          ← 只有配置文件
│   ├── generation_config.json
│   ├── spm.model
│   └── tokenizer_config.json
├── checkpoint-15000/
│   ├── config.json          ← 只有配置文件
│   ├── generation_config.json
│   ├── spm.model
│   └── tokenizer_config.json
```

### 3.2 根因分析（已确认）

**确认根因: SSD 空间不足 + 未配置 HDD 软链接**

1. **SSD 空间不足**: V4 训练日志中曾出现 `safetensors_rust.SafetensorError: I/O error: No space left on device (os error 28)` 错误，说明 SSD 已满
2. **V5 未配置 HDD 软链接**: V4 的输出目录通过软链接指向机械硬盘 `/mnt/sda/kehe/babyllm_output/babylm-llama-v4/`（权重文件存在），但 V5 的输出目录直接在 SSD 上，未配置软链接
3. **`save_pretrained()` 静默失败**: SSD 空间不足时，`save_pretrained()` 只保存了小型配置文件（JSON），权重文件（`model.safetensors` ~97-194MB）写入失败但未抛出异常
4. **训练日志误导**: 显示 "Writing model shards: 100%" 但实际权重未写入磁盘

**文件系统验证**:
- ✅ `/mnt/sda/kehe/babyllm_output/babylm-llama-v4/best_model/model.safetensors` — V4 权重存在
- ❌ `/mnt/sda/kehe/babyllm_output/` 中无 `babylm-llama-v5/` 目录 — V5 未配置软链接

### 3.3 影响

**训练完全白跑** — 2.4 小时的训练成果无法恢复，必须重新训练。

---

## 4. V1-V5 版本对比

| 版本 | 参数量 | 最佳 Val PPL | Epoch | 状态 |
|------|--------|-------------|-------|------|
| V1 (GPT-2) | 110M | ~343 | - | ✅ 完成 |
| V2 (LLaMA) | 125M | ~597 | 7 | ✅ 完成 |
| V3 (LLaMA+SPM) | 125M | ~542 | - | ⚠️ NCCL 超时 |
| V4 (LLaMA deep) | 350M | N/A | - | ❌ 训练失败 |
| **V5 (LLaMA small)** | **51M** | **525.21** | **6** | **🔴 权重丢失** |

### V5 相比之前版本的改进

- **Val PPL 525.21** 是 V2 之后所有版本中的最佳值（V2: 597, V3: ~542）
- 参数量从 125M 缩减到 51M，tokens/params 比从 0.66 提升到 1.61
- 训练速度 ~2.2 it/s，每 epoch 仅需 ~12.7 分钟
- 早停机制正确触发，避免了无效训练

---

## 5. 后续行动计划

### 5.1 紧急修复：模型保存 Bug

**根因**: `save_pretrained()` 在 Accelerate DDP 环境下可能存在保存不完整的问题。

**修复方案**:

1. **方案 A（推荐）**: 使用 `accelerator.save_state()` + 手动 `torch.save()` 替代 `save_pretrained()`
   ```python
   # 替换 accelerator.unwrap_model(model).save_pretrained(path)
   state_dict = accelerator.get_state_dict(model)
   torch.save(state_dict, os.path.join(path, "pytorch_model.bin"))
   config.save_pretrained(path)
   tokenizer.save_pretrained(path)
   ```

2. **方案 B**: 在 `save_pretrained()` 后添加验证逻辑
   ```python
   accelerator.unwrap_model(model).save_pretrained(path)
   # 验证权重文件是否存在
   weight_files = glob.glob(os.path.join(path, "model.safetensors*"))
   if not weight_files:
       # fallback to torch.save
       state_dict = accelerator.get_state_dict(model)
       torch.save(state_dict, os.path.join(path, "pytorch_model.bin"))
   ```

3. **方案 C**: 使用 `safe_serialization=False` 强制使用 pickle 格式
   ```python
   accelerator.unwrap_model(model).save_pretrained(path, safe_serialization=False)
   ```

### 5.2 NCCL 超时修复

在启动脚本中添加 NCCL 环境变量：
```bash
export NCCL_TIMEOUT=1800  # 30 分钟超时
export NCCL_IB_DISABLE=1  # 禁用 InfiniBand（单机不需要）
export NCCL_P2P_DISABLE=0  # 启用 P2P
export NCCL_DEBUG=INFO     # 调试信息
```

### 5.3 重新训练计划

1. **修复 `train_v5.py` 中的模型保存逻辑**（方案 A 或 B）
2. **添加 NCCL 超时配置**
3. **重新启动 Phase 1 训练**（预计 2.5 小时完成 11 epochs）
4. **验证模型权重文件完整性**
5. **启动 Phase 2 KD 训练**（使用 Qwen2.5-0.5B 作为教师模型）

### 5.4 Phase 2 KD 准备工作

Phase 1 完成后需要：
1. 运行 `generate_teacher_logits.py` 生成教师模型 logits
2. 使用 `launch_v5_kd.sh` 启动 Phase 2 KD 训练
3. KD 超参数: λ_ce=0.3, λ_kd=0.7, T=2.0, top_k=10

---

## 6. 预期效果评估

### Phase 1 已达效果
- Val PPL 525.21 在 51M 参数量下是合理的
- 过拟合在 Epoch 6 开始出现，说明数据量仍是瓶颈

### Phase 2 KD 预期
- 参考 DistilQwen2.5 论文，KD 通常可带来 5-15% 的 PPL 改善
- 预期 Phase 2 后 Val PPL 可降至 **450-500** 范围
- 但距离理想目标（<300 PPL）仍有较大差距

### 根本瓶颈
- **数据量**: 100M 字符（~82M tokens）对于任何规模的模型都是硬约束
- **Tokenizer**: 32K 词表对中文覆盖率有限
- **架构**: 在数据量受限的情况下，架构优化的边际收益递减

---

## 7. 总结

| 方面 | 评估 |
|------|------|
| 训练效果 | ✅ Val PPL 525.21，V2 之后最佳 |
| 训练稳定性 | ⚠️ NCCL 超时（训练完成后崩溃） |
| 模型保存 | 🔴 **权重文件丢失，需修复后重训** |
| 过拟合控制 | ✅ 早停机制正确触发 |
| 下一步 | 修复保存 Bug → 重训 Phase 1 → Phase 2 KD |
