# 🛠️ VaLiK Custom Patches & Scripts

> 本文档记录我们在复现 `VaLiK` 过程中，对原仓库所做的关键修复与扩展脚本。  
> 这些修改主要用于解决数据路径、批处理、CoE 级联生成、文本清洗与 ScienceQA 评测等问题。

---

## 📑 目录

- [1. Fix：修复 ScienceQA 数据集下载与目录问题](#1-fix修复-scienceqa-数据集下载与目录问题)
- [2. Add：新增 CoE 级联图像描述脚本](#2-add新增-coe-级联图像描述脚本)
- [3. Fix：修改相似度过滤脚本的文本读取逻辑](#3-fix修改相似度过滤脚本的文本读取逻辑)
- [4. Add：新增 ScienceQA 评测脚本](#4-add新增-scienceqa-评测脚本)
- [5. Add：批处理脚本](#5-add批处理脚本)

---

## 1. Fix：修复 ScienceQA 数据集下载与目录问题

**目标文件：**
- `datasets/Preprocess_ScienceQA.sh`

### 问题描述
原仓库在下载 ScienceQA 图片时，默认直接进入：

```bash
cd data/scienceqa/images
```

但该目录在部分环境下并不存在。  
这会导致：
- `cd` 失败
- 后续 `wget` 和 `unzip` 在错误目录执行
- 最终 `train / val / test` 解压到仓库根目录，而不是 `ScienceQA/data/scienceqa/images/`

### 修改内容

在下载前显式创建图片目录：

```bash
cd ScienceQA

mkdir -p data/scienceqa/images   # 新增：确保图片目录存在

bash tools/download.sh
```

同时修正后续返回目录层级，确保可以正确进入图像描述脚本所在位置：

```bash
cd ../..   # 修正原始返回路径
cd src
cd Image_to_Text
python CLIP_Interrogator_ScienceQA.py
```

### 修改效果
- 保证 ScienceQA 图片下载到正确位置
- 避免 `train / val / test` 目录解压错位
- 为后续 CoE / Prune / 评测脚本提供统一数据路径

---

## 2. Add：新增 CoE 级联图像描述脚本

**新增文件：**
- `src/CoE_Image_to_Text.py`
- `src/coe_batch.sh`

### 背景
原仓库 `src/Image_to_Text.py` 只支持单模型图像描述生成，不支持论文中的 CoE（Cascade-of-Experts）级联方式。

### 新增功能
我们新增了 `src/CoE_Image_to_Text.py`，支持：

1. 多阶段级联图像描述生成  
2. 自动读取前一阶段输出作为下一阶段输入  
3. 输出文件名按前缀链命名，避免覆盖与跳过
4. 支持不同模型组合，如：
   - `BLIP-2`
   - `Qwen2-VL-2B`
   - `LLaVA-7B`

### 输出命名示例

```text
image.blip2-flan-t5.txt
image.blip2-flan-t5.qwen2vl2b.txt
image.blip2-flan-t5.qwen2vl2b.llava7b.txt
```

### 解决的问题
- 原始单模型脚本无法体现 CoE 级联流程
- 已存在 `image.txt` 时会被跳过，难以做多阶段输出
- 无法显式记录当前描述来自哪个阶段/模型

### 批处理脚本
新增 `src/coe_batch.sh` 用于按 problem id 批量执行 CoE 三阶段流程，减少手动重复命令。

---

## 3. Fix：修改相似度过滤脚本的文本读取逻辑

**目标文件：**
- `src/Prune/similarity_verification.py`

### 问题描述
原始脚本直接读取文本文件：

```python
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
```

但在我们的 CoE 输出中，文本文件通常包含：
- 文件头 `[Description]`
- 多余空行
- 多段落换行

这些内容虽然不会导致脚本报错，但会引入无意义噪声，影响后续相似度过滤效果。

### 修改内容

我们将其修改为：

```python
import re

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    text = re.sub(r'^\s*\[Description\]\s*', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()

    return text
```

### 修改效果
- 去掉 `[Description]` 文件头
- 压缩多余空行
- 保留正常段落结构
- 让 prune 输入更干净、更稳定

---

## 4. Add：新增 ScienceQA 评测脚本

**新增文件：**
- `src/evaluate_scienceqa_rag_vs_kg.py`

### 背景
原仓库缺少面向 ScienceQA 的完整题目级评测脚本。  
我们新增统一评测脚本，支持三种模式：

- `baseline`
- `rag`
- `kg`

### 功能说明

#### baseline
- 不使用任何图像知识
- 只根据题目和选项作答

#### rag
- 使用图像描述文本做普通文本检索
- 将检索到的文本片段作为上下文送入 LLM

#### kg
- 使用 LightRAG 将过滤后的图像描述构建为轻量图结构
- 通过 `local / global / hybrid` 图检索模式查询
- 将图式检索结果送入 LLM 作答

### 脚本特点
- 支持按 `problem_id` 评测
- 若缺少 CoE 或 prune 结果，可现场自动构建
- 输出预测答案、是否正确、检索知识等信息
- 支持 batch 评测与结果保存

---

## 5. Add：批处理脚本

**新增文件：**
- `src/evaluate_batch.sh`

### 背景
为了在多个 ScienceQA 样本上自动运行实验，我们新增了批处理脚本，统一调度：

- `baseline`
- `rag`
- `kg`


---