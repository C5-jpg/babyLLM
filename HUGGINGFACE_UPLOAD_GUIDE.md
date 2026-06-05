# BabyLLM 大文件上传指南：HuggingFace & ModelScope

> 本项目模型权重、训练数据、教师 logits 等大文件总计约 **100GB+**，远超 GitHub 限制。
> 以下提供两种主流平台的完整上传方案。

---

## 📦 需要上传的文件清单

| 类别 | 路径 | 大小 | 优先级 |
|:-----|:-----|:-----|:-------|
| **V13 SOTA 模型** | `babylm-v13/stage2_mntp/best_model_ema/` | **~181MB** | 🔴 必须 |
| V13 Stage 1 EMA | `babylm-v13/stage1_clm_sgdr/best_model_ema/` | ~360MB | 🟡 推荐 |
| V12 效率最佳 | `babylm-v12/stage2_mntp/best_model_ema/` | ~207MB | 🟡 推荐 |
| V14 模型 | `babylm-v14/stage4_self_distill/best_model/` | ~240MB | 🟢 可选 |
| V15 模型 | `babylm-v15/stage2_mntp/best_model_ema/` | ~260MB | 🟢 可选 |
| V15.1 模型 | `babylm-v15-1/stage2_mntp/best_model_ema/` | ~260MB | 🟢 可选 |
| Tokenizer | `data/tokenizer_v7/` (spm.model 等) | ~1MB | 🔴 必须 |
| 训练数据 (processed_v7) | `data/processed_v7/` | ~350MB | 🟡 推荐 |
| 训练数据 (processed_v3) | `data/processed_v3/` | ~354MB | 🟢 可选 |
| 教师 logits | `output/teacher_logits_v2/` | **14GB** | 🟢 可选 |
| 教师 logits | `output/teacher_logits_v5kd/` | **3.1GB** | 🟢 可选 |
| 全部模型总大小 | `/mnt/sda/kehe/babyllm_output/` | **~100GB** | — |

---

## 方案一：HuggingFace Hub（推荐，国际标准）

### 1.1 注册与获取 Token

1. 注册 HuggingFace 账号: https://huggingface.co/join
2. 进入 Settings → Access Tokens → New token
3. 命名为 `babyllm-upload`，权限选择 **Write**
4. 复制 token（格式：`hf_xxxxxxxxxxxxxxxxxxxxxxxx`）

### 1.2 安装依赖

```bash
pip install huggingface_hub
```

### 1.3 登录 HuggingFace

```bash
huggingface-cli login
# 粘贴你的 token: hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

或者用环境变量：
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxx"
```

### 1.4 创建仓库

你可以选择创建**一个 Model 仓库**（推荐）或**多个仓库**（按版本分）：

```bash
# 方式一：单一仓库（推荐）
huggingface-cli repo create babyllm-chinese --type model --organization C5-jpg

# 方式二：按版本分仓库
huggingface-cli repo create babyllm-v13-sota --type model
huggingface-cli repo create babyllm-v12-efficient --type model
```

### 1.5 上传方法 A：Python API（推荐，支持大文件 + 断点续传）

创建上传脚本 `upload_to_hf.py`：

```python
#!/usr/bin/env python3
"""上传 BabyLLM 模型到 HuggingFace Hub"""

from huggingface_hub import HfApi, create_repo
import os

# ============ 配置区 ============
HF_REPO_ID = "C5-jpg/babyllm-chinese"  # 你的 HF 仓库 ID
MODEL_BASE = "/mnt/sda/kehe/babyllm_output"  # 模型文件根目录
TOKENIZER_DIR = "/home/kehe/babyllm/babyLLM/data/tokenizer_v7"  # Tokenizer 目录

api = HfApi()

# 创建仓库（如不存在）
try:
    create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
    print(f"✅ 仓库 {HF_REPO_ID} 已就绪")
except Exception as e:
    print(f"仓库创建: {e}")


def upload_model(version, stage, model_subdir="best_model_ema", prefix=None):
    """上传一个模型目录到 HF"""
    local_dir = os.path.join(MODEL_BASE, version, stage, model_subdir)
    if not os.path.exists(local_dir):
        print(f"⚠️  跳过 {local_dir} (不存在)")
        return

    path_in_repo = prefix or f"{version}/{stage}/{model_subdir}"
    print(f"📤 上传 {local_dir} → {path_in_repo}/")

    api.upload_folder(
        folder_path=local_dir,
        repo_id=HF_REPO_ID,
        path_in_repo=path_in_repo,
        repo_type="model",
        # 支持断点续传，大文件自动用 LFS
        multi_commits=True,
        multi_commits_message=f"Upload {version}/{stage}/{model_subdir}",
    )
    print(f"✅ 完成 {version}/{stage}/{model_subdir}")


def upload_tokenizer():
    """上传 SentencePiece Tokenizer"""
    print(f"📤 上传 Tokenizer: {TOKENIZER_DIR}")
    api.upload_folder(
        folder_path=TOKENIZER_DIR,
        repo_id=HF_REPO_ID,
        path_in_repo="tokenizer",
        repo_type="model",
    )
    print("✅ Tokenizer 上传完成")


def upload_readme():
    """上传项目 README 作为 Model Card"""
    readme_path = "/home/kehe/babyllm/README.md"
    if os.path.exists(readme_path):
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=HF_REPO_ID,
            repo_type="model",
        )
        print("✅ README 上传完成")


# ============ 执行上传 ============
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python upload_to_hf.py [all|v13|v12|tokenizer|readme]")
        sys.exit(1)

    target = sys.argv[1]

    if target == "all":
        # 上传所有关键模型
        upload_tokenizer()
        upload_readme()
        upload_model("babylm-v13", "stage2_mntp", "best_model_ema", prefix="v13-sota")
        upload_model("babylm-v13", "stage1_clm_sgdr", "best_model_ema", prefix="v13-stage1")
        upload_model("babylm-v12", "stage2_mntp", "best_model_ema", prefix="v12-efficient")
        upload_model("babylm-v14", "stage4_self_distill", "best_model", prefix="v14")
        upload_model("babylm-v15", "stage2_mntp", "best_model_ema", prefix="v15")
        upload_model("babylm-v15-1", "stage2_mntp", "best_model_ema", prefix="v15-1")
    elif target == "v13":
        upload_model("babylm-v13", "stage2_mntp", "best_model_ema", prefix="v13-sota")
    elif target == "v12":
        upload_model("babylm-v12", "stage2_mntp", "best_model_ema", prefix="v12-efficient")
    elif target == "tokenizer":
        upload_tokenizer()
    elif target == "readme":
        upload_readme()
    else:
        print(f"未知目标: {target}")
```

使用方法：

```bash
# 上传 V13 SOTA 模型（必须）
python upload_to_hf.py v13

# 上传 V12 效率最佳模型
python upload_to_hf.py v12

# 上传 Tokenizer
python upload_to_hf.py tokenizer

# 一键上传所有关键模型
python upload_to_hf.py all
```

### 1.6 上传方法 B：Git 命令（适合小文件）

```bash
# 克隆 HF 仓库（自动配置 Git LFS）
git clone https://huggingface.co/C5-jpg/babyllm-chinese
cd babyllm-chinese

# 复制模型文件
cp /mnt/sda/kehe/babyllm_output/babylm-v13/stage2_mntp/best_model_ema/model.safetensors ./v13-sota/
cp /mnt/sda/kehe/babyllm_output/babylm-v13/stage2_mntp/best_model_ema/config.json ./v13-sota/
cp /mnt/sda/kehe/babyllm_output/babylm-v13/stage2_mntp/best_model_ema/spm.model ./v13-sota/
cp /mnt/sda/kehe/babyllm_output/babylm-v13/stage2_mntp/best_model_ema/tokenizer.json ./v13-sota/

# 创建 Model Card
cat > README.md << 'EOF'
---
language: zh
license: mit
library_name: transformers
tags:
  - text-generation
  - chinese
  - babylm
  - llama
metrics:
  - perplexity
  - accuracy
---

# BabyLLM Chinese (V13 SOTA)

在 NLPCC 2026 ChineseBabyLM 挑战赛上从零预训练的中文语言模型。

- PPL: 38.68 | 参数量: 94.2M | 架构: LLaMA 768d/14L
- SentencePiece Unigram 32K tokenizer
- 2-stage pipeline: CLM → MNTP + EMA

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("C5-jpg/babyllm-chinese", subfolder="v13-sota")
tokenizer = AutoTokenizer.from_pretrained("C5-jpg/babyllm-chinese", subfolder="v13-sota")
```
EOF

# 提交推送
git lfs track "*.safetensors"
git lfs track "*.bin"
git lfs track "*.model"
git add .gitattributes
git add .
git commit -m "Upload BabyLLM V13 SOTA model"
git push
```

### 1.7 上传训练数据到 Dataset 仓库

```python
from huggingface_hub import HfApi

api = HfApi()

# 创建 Dataset 仓库
api.create_repo(repo_id="C5-jpg/babylm-zho-processed", repo_type="dataset", exist_ok=True)

# 上传处理后的数据
api.upload_folder(
    folder_path="/home/kehe/babyllm/babyLLM/data/processed_v7",
    repo_id="C5-jpg/babylm-zho-processed",
    path_in_repo="processed_v7",
    repo_type="dataset",
)

# 上传 tokenizer
api.upload_folder(
    folder_path="/home/kehe/babyllm/babyLLM/data/tokenizer_v7",
    repo_id="C5-jpg/babylm-zho-processed",
    path_in_repo="tokenizer_v7",
    repo_type="dataset",
)
```

---

## 方案二：ModelScope 魔搭（国内访问更快）

### 2.1 注册与获取 Token

1. 注册 ModelScope 账号: https://www.modelscope.cn/register
2. 进入 个人中心 → API Token 管理 → 创建 Token
3. 复制 SDK Access Token

### 2.2 安装依赖

```bash
pip install modelscope
```

### 2.3 登录 ModelScope

```bash
modelscope login
# 粘贴你的 Access Token
```

或环境变量：
```bash
export MODELSCOPE_API_TOKEN="your-token-here"
```

### 2.4 上传方法 A：Python SDK（推荐）

```python
#!/usr/bin/env python3
"""上传 BabyLLM 模型到 ModelScope"""

from modelscope.hub.api import HubApi
from modelscope.hub.file_download import model_file_download
import os

# ============ 配置区 ============
MODELSCOPE_REPO = "C5-jpg/babyllm-chinese"  # ModelScope 仓库 ID
MODEL_BASE = "/mnt/sda/kehe/babyllm_output"
TOKENIZER_DIR = "/home/kehe/babyllm/babyLLM/data/tokenizer_v7"

api = HubApi()
# 如果没通过 modelscope login，可以手动设置:
# api.login("your-token-here")


def upload_model_ms(version, stage, model_subdir="best_model_ema", prefix=None):
    """上传模型到 ModelScope"""
    local_dir = os.path.join(MODEL_BASE, version, stage, model_subdir)
    if not os.path.exists(local_dir):
        print(f"⚠️  跳过 {local_dir} (不存在)")
        return

    target_dir = prefix or f"{version}/{stage}/{model_subdir}"

    for filename in os.listdir(local_dir):
        filepath = os.path.join(local_dir, filename)
        if os.path.isfile(filepath):
            print(f"  📤 {filename} ({os.path.getsize(filepath)/1e6:.1f}MB)")
            api.push_model(
                model_id=MODELSCOPE_REPO,
                model_dir=local_dir,
                # 或者用 push_file 单文件上传:
                # commit_message=f"Upload {target_dir}/{filename}"
            )
            break  # push_model 上传整个目录，只需调用一次

    print(f"✅ 完成 {target_dir}")


# 上传 V13 SOTA
upload_model_ms("babylm-v13", "stage2_mntp", "best_model_ema", prefix="v13-sota")
```

### 2.5 上传方法 B：Git 命令

```bash
# 克隆 ModelScope 仓库
git clone https://www.modelscope.cn/C5-jpg/babyllm-chinese.git
cd babyllm-chinese

# 复制模型文件（和 HuggingFace 的 Git 方式一样）
mkdir -p v13-sota
cp /mnt/sda/kehe/babyllm_output/babylm-v13/stage2_mntp/best_model_ema/* v13-sota/

# Git LFS 自动跟踪大文件
git lfs track "*.safetensors"
git lfs track "*.model"
git add .
git commit -m "Upload BabyLLM V13 SOTA"
git push
```

> ⚠️ ModelScope Git 方式单文件限制 **200MB**，超过建议用 SDK 方式（单文件最大 50GB）

---

## 方案三：同时上传两个平台

```python
#!/usr/bin/env python3
"""同时上传到 HuggingFace 和 ModelScope"""

from huggingface_hub import HfApi as HFApi
from modelscope.hub.api import HubApi as MSApi
import os

# 配置
HF_REPO = "C5-jpg/babyllm-chinese"
MS_REPO = "C5-jpg/babyllm-chinese"
MODEL_BASE = "/mnt/sda/kehe/babyllm_output"

hf_api = HFApi()
ms_api = MSApi()


def upload_to_both(local_dir, hf_path, ms_path):
    """同时上传到 HF 和 MS"""
    print(f"\n{'='*60}")
    print(f"📤 上传: {local_dir}")
    print(f"   HF: {HF_REPO}/{hf_path}")
    print(f"   MS: {MS_REPO}/{ms_path}")
    print(f"{'='*60}")

    # 上传到 HuggingFace
    try:
        hf_api.upload_folder(
            folder_path=local_dir,
            repo_id=HF_REPO,
            path_in_repo=hf_path,
            repo_type="model",
        )
        print("✅ HuggingFace 上传完成")
    except Exception as e:
        print(f"❌ HuggingFace 失败: {e}")

    # 上传到 ModelScope
    try:
        ms_api.push_model(
            model_id=MS_REPO,
            model_dir=local_dir,
        )
        print("✅ ModelScope 上传完成")
    except Exception as e:
        print(f"❌ ModelScope 失败: {e}")


# 上传 V13 SOTA
v13_dir = os.path.join(MODEL_BASE, "babylm-v13/stage2_mntp/best_model_ema")
upload_to_both(v13_dir, "v13-sota", "v13-sota")
```

---

## 📋 推荐上传计划

### 最小必要集（~200MB）

```bash
# 1. V13 SOTA 模型权重 + config + tokenizer（必须）
python upload_to_hf.py v13        # 181MB

# 2. Tokenizer（必须，供他人使用）
python upload_to_hf.py tokenizer  # ~1MB
```

### 推荐集（~1GB）

```bash
# 加上 V12 效率最佳模型
python upload_to_hf.py v12        # 207MB

# 加上 V13 Stage 1（消融研究用）
# 手动上传 babylm-v13/stage1_clm_sgdr/best_model_ema/  ~360MB
```

### 完整集（~100GB）

全部版本的模型权重 + 教师 logits + 全部训练数据。

---

## 🔧 常见问题

### Q: HuggingFace 上传速度慢怎么办？
- 使用 `multi_commits=True` 参数支持并行上传
- 上传大文件时考虑使用 `hf_transfer`：`pip install hf_transfer && export HF_HUB_ENABLE_HF_TRANSFER=1`
- 可以设置 `HF_ENDPOINT=https://hf-mirror.com` 使用镜像站

### Q: ModelScope 在国外服务器上访问慢？
- ModelScope 是国内平台，国外服务器建议用 HuggingFace
- 如果在国内，ModelScope 速度更快

### Q: GitHub 仓库中如何引用 HF/MS 上的模型？
在 GitHub README 中添加链接：
```markdown
## 模型权重
- 🤗 [HuggingFace](https://huggingface.co/C5-jpg/babyllm-chinese)
- 🤖 [ModelScope](https://www.modelscope.cn/models/C5-jpg/babyllm-chinese)
```

### Q: 上传中断了怎么办？
HuggingFace 的 `upload_folder` 支持**断点续传**，重新运行同一命令即可。已上传的文件会被跳过。
