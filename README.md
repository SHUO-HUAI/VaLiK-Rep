# VaLiK on ScienceQA：复现、分析与代码说明

## 1. 项目目标

本项目围绕论文 **“Aligning Vision to Language: Annotation-Free Multimodal Knowledge Graph Construction for Enhanced LLMs Reasoning”** 展开，任务要求包括：

1. **基础复现**  
   跑通 VaLiK 完整流程，并在 **ScienceQA** 上做评测。
2. **分析理解**  
   回答以下问题：
   - CoE 级联 VLM 的设计动机是什么？与单一 VLM 相比优势在哪？
   - 跨模态相似度验证如何过滤噪声？
   - 如果去掉知识图谱，直接用 RAG 检索图像描述会有什么问题？
3. **迁移创新（bonus）**  
   提出评估文本描述对图像信息覆盖率的方法与指标。

本仓库当前提交内容重点完成了：
- CoE 三阶段图像描述生成
- CLIP 相似度过滤（Prune）
- baseline / RAG / KG 三种评测模式
- 小规模与批量实验脚本
- 对论文核心问题的分析与回答
- `patches.md` 描述了我们对原始仓库的改动

---

## 2. 环境与依赖

### 2.1 安装环境

使用本项目提交版本中的环境配置文件 `environment_new.yml`。

```bash
grep -v '^prefix:' environment_new.yml > /tmp/valik.environment.yml
conda env create -f /tmp/valik.environment.yml
conda activate valik
```

### 2.2 Ollama 服务与模型

本项目本地推理依赖 Ollama。

安装并启动 Ollama 后，需要准备以下模型：

```bash
ollama pull deepseek-r1:7b
ollama pull llava:7b
ollama pull nomic-embed-text
```

---

## 3. ScienceQA 数据准备

### 3.1 数据集下载

运行脚本：

```bash
cd datasets
bash Preprocess_ScienceQA.sh
cd ..
```

其功能主要包括：
- 下载 ScienceQA
- 调用 `CLIP_Interrogator_ScienceQA.py`


### 3.2 正确的数据目录

最终使用的数据路径为：

```text
datasets/ScienceQA/data/scienceqa/
```

图片目录为：

```text
datasets/ScienceQA/data/scienceqa/images/train
datasets/ScienceQA/data/scienceqa/images/val
datasets/ScienceQA/data/scienceqa/images/test
```

---

## 4. CoE 三阶段图像描述生成

我们将 `src/Image_to_Text.py` 改写为支持 **CoE（Cascade-of-Experts）** 的版本：`src/CoE_Image_to_Text.py`。

### 4.1 CoE 设计

三阶段模型顺序如下：

1. **BLIP-2**：生成第一阶段基础描述  
2. **Qwen2-VL-2B**：在 BLIP-2 描述基础上补充细节  
3. **LLaVA-7B**：在前两阶段基础上进一步补充视觉信息

输出命名采用前缀链方式，例如：

```text
image.blip2-flan-t5.txt
image.blip2-flan-t5.qwen2vl2b.txt
image.blip2-flan-t5.qwen2vl2b.llava7b.txt
```

这种方式有一个优点：
- 能显式表示 CoE 级联顺序

### 4.2 运行命令

如果直接用 Python 命令，则逻辑对应为：
- 第一阶段：BLIP-2 生成基础描述
- 第二阶段：Qwen2-VL-2B 读取前序描述并补充细节
- 第三阶段：LLaVA-7B 继续补充视觉信息

以 `test/5` 为例：

```bash
# Stage 1
python src/CoE_Image_to_Text.py \
  --input datasets/ScienceQA/data/scienceqa/images/test/5 \
  blip2 --blip2_version flan-t5

# Stage 2
python src/CoE_Image_to_Text.py \
  --input datasets/ScienceQA/data/scienceqa/images/test/5 \
  --previous_prefixes blip2-flan-t5 \
  qwen2-vl --qwen2vl_version 2b

# Stage 3
python src/CoE_Image_to_Text.py \
  --input datasets/ScienceQA/data/scienceqa/images/test/5 \
  --previous_prefixes blip2-flan-t5,qwen2vl2b \
  llava --llava_version 7b
```

### 4.3 CoE 实现说明

我们的 CoE 不是简单把三份描述拼接，而是：

- 第一阶段：生成完整基础描述
- 第二阶段：读取前序描述，只补充遗漏细节
- 第三阶段：再次读取前序描述，进一步补充与完善

因此它更接近论文中的 **级联 VLM** 思路，而不是多模型独立生成后简单拼接。

### 4.4 批量处理说明

由于显存限制，我们采用按 problem id 逐个目录处理的方式运行 CoE。为方便批量实验，我们另外提供了对应的批处理脚本。

```bash
bash src/coe_batch.sh <ID1> <ID2> ...
```

---

## 5. Prune：跨模态相似度过滤

在生成 CoE 最终描述后，我们使用 `src/Prune/similarity_verification.py` 做噪声过滤。

### 5.1 运行命令

```bash
python src/Prune/similarity_verification.py \
  --image_path datasets/ScienceQA/data/scienceqa/images/test/5/image.png \
  --text_path  datasets/ScienceQA/data/scienceqa/images/test/5/image.blip2-flan-t5.qwen2vl2b.llava7b.txt \
  --threshold 0.20 \
  --mode sentence
```

输出文件为：

```text
image.blip2-flan-t5.qwen2vl2b.llava7b_filtered.txt
```

### 5.2 原理

该脚本：
1. 将文本按 `word / sentence / window` 切分
2. 使用 `openai/clip-vit-large-patch14` 计算图像与每个文本 chunk 的相似度
3. 仅保留相似度大于阈值的 chunk

对 ScienceQA，我们使用：

- `threshold = 0.20`
- `mode = sentence`

这是论文与官方仓库中推荐的典型设置。

此外，我们提供了一个批处理脚本。支持多 ID 连续输入：

```bash
bash src/prune_batch.sh <ID1> <ID2> ...
```

---

## 6. 评测脚本设计

### 6.1 我们实现的三种模式

我们实现了统一评测脚本：

```text
src/evaluate_scienceqa_rag_vs_kg.py
```

支持三种模式：

- `baseline`
- `rag`
- `kg`

### 6.2 baseline

不使用任何图像知识，仅根据题目和选项让 LLM 作答。

### 6.3 rag

将图像描述文本作为普通文本语料：
- 切分为句子或段落
- 根据题目和选项做文本检索
- 取 top-k 片段作为补充上下文
- 交给 LLM 作答

这代表：
> **去掉知识图谱，仅用文本 RAG 检索图像描述**

### 6.4 kg

将过滤后的图像描述送入 LightRAG：
- 自动抽取实体与关系
- 构建轻量图结构
- 通过 `naive / local / global / hybrid` 方式查询
- 将图检索到的知识送给 LLM 作答

这代表：
> **保留知识图谱结构进行图式检索**

### 6.5 运行方式

#### baseline

```bash
mkdir results
```

```bash
python src/evaluate_scienceqa_rag_vs_kg.py \
  --repo_root . \
  --scienceqa_root datasets/ScienceQA/data/scienceqa \
  --split test \
  --problem_ids 5 \
  --mode baseline \
  --ollama_model deepseek-r1:7b \
  --output results/baseline.json
```

#### rag

```bash
python src/evaluate_scienceqa_rag_vs_kg.py \
  --repo_root . \
  --scienceqa_root datasets/ScienceQA/data/scienceqa \
  --split test \
  --problem_ids 5 \
  --mode rag \
  --auto_build \
  --rag_source all \
  --rag_chunk_by sentence \
  --rag_topk 5 \
  --ollama_model deepseek-r1:7b \
  --output results/rag.json
```

#### kg

```bash
python src/evaluate_scienceqa_rag_vs_kg.py \
  --repo_root . \
  --scienceqa_root datasets/ScienceQA/data/scienceqa \
  --split test \
  --problem_ids 5 \
  --mode kg \
  --auto_build \
  --kg_working_dir tmp_lightkg_5 \
  --kg_llm_model deepseek-r1:7b \
  --kg_query_mode hybrid \
  --ollama_model deepseek-r1:7b \
  --output results/kg.json
```


此外，我们也提供了批量评测脚本，用于在多个 ScienceQA 样本上自动运行 baseline / RAG / KG 三种模式。

```bash
bash src/evaluate_batch.sh <ID1> <ID2> ...
```

---

## 7. 实验结果

由于计算资源和时间关系，在 165 个 ScienceQA 测试样本上的结果如下：

```text
==================================================
Mode      Files     Total     Correct   Accuracy
==================================================
baseline  165       165       110       0.6667
rag       165       165       130       0.7879
kg        165       165       115       0.6970
==================================================
```

### 7.1 结果解读

1. **RAG 明显优于 baseline**
   - 说明经过 CoE 与 prune 后的图像描述文本，对 ScienceQA 问答是有效的知识来源。

2. **KG 高于 baseline，但低于 RAG**
   - 说明图结构知识也提供了增益
   - 但在当前降配实现中，KG 抽取与组织造成了一定信息损失

3. **当前实现下 RAG 优于 KG**
   - 对 ScienceQA 这类任务，直接保留文本细节的 RAG 更能保留答题线索
   - KG 在结构化过程中会压缩信息，若抽取质量不足，反而弱于文本 RAG

### 7.2 为什么会出现 `RAG > KG`

这不是说知识图谱理念无效，而是说明在当前降配复现环境中：

- KG 使用的抽取模型较小（本地 `DeepSeek-R1-7B`）
- LightRAG 图构建较轻量
- 实体关系抽取较稀疏
- 许多对答题有帮助的细粒度描述，在图结构化过程中被压缩或丢失

因此当前实现呈现出：

> **文本 RAG 更强，KG 次之，baseline 最弱**

---

## 8. 对三道分析题的回答

### 问题 1：CoE 级联 VLM 的设计动机是什么？与单一 VLM 相比优势在哪？

CoE（Cascade-of-Experts）设计的动机在于：

- 单一 VLM 对图像的描述存在明显偏好与盲区，这一点可以从不同阶段生成的 CoE 文本差异中观察到
- 不同模型擅长提取的信息类型不同：
  - 有的擅长关键对象
  - 有的擅长局部属性
  - 有的擅长背景与场景关系

因此，论文提出让多个 VLM **按级联方式逐步补充描述**，而不是依赖单一模型一次性完成描述。

#### 相比单一 VLM 的优势
1. **覆盖率更高**  
   后续模型可以补充前序模型遗漏的对象、属性和关系。
2. **描述更细粒度**  
   多阶段生成能获得更丰富的视觉细节。
3. **模型偏差可互补**  
   不同模型的感知偏差可以相互弥补。

在我们的实践中也观察到：
- 第一阶段给出主体描述
- 第二、三阶段会补充背景、细节、局部属性和空间信息

因此 CoE 相比单一 VLM 更适合作为后续知识构建的输入。

---

### 问题 2：跨模态相似度验证具体如何过滤噪声？

该过程的核心思想是：

> **只保留真正被图像支持的文本片段**

具体步骤如下：

1. 将图像描述文本切分为若干 chunk（word / sentence / window）
2. 使用 CLIP 计算图像与每个文本 chunk 的相似度
3. 设置阈值 `τ`
4. 仅保留相似度大于阈值的 chunk

这样做可以过滤掉：

- VLM 幻觉出的错误细节
- 与图像无关的扩展性描述
- 过度解释性语言
- 低相关的冗余文本

在 ScienceQA 上，我们采用：

- `mode = sentence`
- `threshold = 0.20`

这能在保留主要图像内容的同时，降低噪声输入对后续问答的干扰。

---

### 问题 3：如果去掉知识图谱，直接用 RAG 检索图像描述会有什么问题？

我们的实验显示：

- `baseline = 66.67%`
- `rag = 78.79%`
- `kg = 69.70%`

说明在当前实现中，RAG 是很强的基线。  
但“RAG 更强”并不意味着它没有问题。

#### 直接用 RAG 的问题

1. **检索单位是文本块，不是显式关系**
   - RAG 召回的是句子或段落
   - 实体之间的关系仍然隐含在自然语言中
   - 需要 LLM 自己再去拼接逻辑链

2. **检索结果容易碎片化**
   - 对应问题的信息可能分散在多个文本块中
   - LLM 需要自己整合多个片段

3. **噪声和冗余容易一起进入上下文**
   - 即使做了 prune，文本片段中仍可能存在无关描述
   - 这些片段会和关键信息一起被送入模型

4. **对多跳推理与结构化对齐支持弱**
   - RAG 适合“召回相关描述”
   - 但不擅长显式表达“实体—关系—属性”的结构

#### 知识图谱的理论优势

知识图谱将图像描述进一步组织为：

- 实体节点
- 关系边
- 属性信息

这样检索时不再仅返回文本块，而是返回：

- 实体邻域
- 关系路径
- 结构化上下文

它更适合：
- 多跳推理
- 关系组合
- 可解释检索

#### 为什么我们这里 KG 没超过 RAG

这是因为当前复现环境中：

- 图谱抽取与查询使用的是本地 `LightRAG + DeepSeek-R1-7B` 轻量配置，结构化抽取能力相对有限
- 实体和关系抽取得较稀疏
- 图结构化带来了额外的信息压缩

因此在本实验中：

> **RAG 更好地保留了原始视觉细节，而 KG 由于抽取质量有限，尚未充分发挥其结构化优势。**

所以正确结论不是“KG 没用”，而是：

> **在当前降配复现条件下，RAG 是更强的工程基线；KG 理论上更适合结构化推理，但需要更高质量的抽取与构图才能体现优势。**

---

## 9. original_text 的处理说明

论文中的 `original_text` 指的是 ScienceQA 题目原始文本知识，例如：

- question
- hint
- lecture
- solution

原始仓库提供了 `Get_Text_ScienceQA.py`，用于把这些内容拼成一个大文本库 `ScienceQA_Text.txt`。

但是在我们的复现中发现：

- 若将整个 `ScienceQA_Text.txt` 全量灌入 LightRAG
- 构图成本极高
- chunk 数量暴涨
- 在当前环境下非常慢

因此当前提交版本采用的策略是：

- **先完成 image-only 路线**
- 不将整个 `original_text` 全量放入 KG
- 后续如果扩展到 text-image KG，应优先采用 **按 problem id 定向构造文本知识**，而不是把整个 ScienceQA 文本库一次性塞入
- 同时这种做法也更符合按题目构造图像知识、避免引入过强全局文本先验的实验设定

---

## 10. 当前复现的局限性

1. **非论文主配置**
   - 论文使用更大模型和更强硬件
   - 我们在当前有限资源环境下进行了降配实现

2. **KG 路线为轻量复现**
   - 当前图构建使用本地 `LightRAG + DeepSeek-R1-7B`
   - 实体关系抽取质量有限

3. **RAG/KG 主要完成了方法流程的闭环验证**
   - 当前实现已经验证 baseline / RAG / KG 三条路线均可运行，并可进一步扩展到更大模型和更强配置
   - 但当前结果不应简单视为与论文绝对数值的严格逐点对齐

---

## 11. 文本描述覆盖率评估方法

为评估生成文本对图像信息的覆盖率，我们可以设计一套**无需人工逐项标注、可零样本运行**的评价框架。核心思想是同时衡量：

- 文本中提到的信息，是否真的出现在图像中
- 图像中重要的信息，是否被文本覆盖到

因此，该方案不仅惩罚**幻觉**，也惩罚**遗漏**。

---

### 11.1 细粒度覆盖率（对象、属性、关系）

传统基于目标检测或场景图标注的方法，通常依赖繁琐的 Ground Truth 标注，难以快速扩展到大规模实验。为此，我们采用 **VLM 作为零样本裁判**，直接对“生成文本—原始图像”进行结构化事实核查。

我们关注三类信息：

- **对象（Object）**
- **属性（Attribute）**
- **关系（Relation）**

#### 评估流程
1. 读取生成的图像描述（Generated Text）与原始图像（Image）。
2. 将图像与文本一同输入 VLM。
3. 要求 VLM 输出结构化 JSON，包含：
   - `verified_matches`：文本中正确且被图像支持的事实
   - `hallucinations`：文本中提及但图像中不存在的事实
   - `critical_misses`：图像中重要但文本遗漏的事实
4. 根据 JSON 结果计算 Precision、Recall 和 F1。

#### VLM Prompt Template

```text
// System Prompt
You are a rigorous visual-linguistic evaluator. Your task is to evaluate a generated description against its corresponding image.
You must identify three categories of information:
1. Verified Matches: Objects/attributes/relations mentioned in the text that ACCURATELY exist in the image.
2. Hallucinations: Objects/attributes/relations mentioned in the text that DO NOT exist in the image.
3. Critical Misses: Salient, major objects or crucial relationships clearly visible in the image but COMPLETELY MISSING from the description.

Output STRICTLY in the following JSON format without any markdown wrappers or additional text.

// User Prompt
Description: [Insert generated text here]
Image: [Attach image]

Expected JSON Schema:
{
  "verified_matches": ["list of correct claims"],
  "hallucinations": ["list of fabricated claims in text"],
  "critical_misses": ["list of major elements in image ignored by text"]
}
```

#### 指标计算方式

设：

- `V = len(verified_matches)`
- `H = len(hallucinations)`
- `M = len(critical_misses)`

则：

```text
Precision = V / (V + H)
Recall = V / (V + M)
F1 = 2 * Precision * Recall / (Precision + Recall)
```

其中：

- **Precision** 惩罚“胡说八道”
- **Recall** 惩罚“漏掉关键信息”
- **F1** 同时兼顾两者，可作为对象/属性/关系覆盖率的综合指标

---

### 11.2 任务解耦：先拆文本，再做视觉验证

为了进一步提高评估稳定性，我们将任务拆成两个阶段，避免单次多模态评估中出现“裁判漏看”或“裁判漏判”的问题。

#### Step 1：纯文本拆解（Text-only Extraction）

首先，不让 VLM 直接处理整段文本，而是先调用一个**纯文本 LLM**（例如 `DeepSeek-V3`、`GPT-4o-mini`、`Qwen2.5-7B` 等），将生成描述拆成一条条原子化断言（Assertions）。

例如输入：

```text
A red car is parked next to a large tree.
```

纯文本 LLM 输出：

```text
1. There is a car.
2. The car is red.
3. There is a tree.
4. The tree is large.
5. The car is parked next to the tree.
```

这样做的优点是：

- 文本拆解任务交给纯文本模型，速度快、成本低
- 断言列表更完整，不容易漏掉文本中的信息点
- 后续视觉验证可以逐条进行，更稳定

#### Step 2：视觉逐条验证（Visual Verification）

拿到断言列表后，再调用 VLM 对每一条断言逐条做 True / False 判定。

示例 Prompt：

```text
Look at the image and answer True or False for each of the following claims:
1. There is a car.
2. The car is red.
3. There is a tree.
4. The tree is large.
5. The car is parked next to the tree.
```

这样做的优势在于：

- 视觉模型不需要自己“想出”该评估什么
- 它只需要逐条核查清单中的事实是否成立
- 从工程上降低了裁判遗漏评估项的风险

---

### 11.3 双向覆盖评估

仅检查“文本是否正确”是不够的，因为文本可能虽然没有幻觉，但仍然**漏掉大量重要图像内容**。因此，本方案采用双向评估：

#### 方向 1：文本 → 图像
检查文本中的对象、属性、关系是否真的存在于图像中。  
对应指标：**Precision**

#### 方向 2：图像 → 文本
检查图像中的显著对象、属性、关系是否被文本覆盖。  
对应指标：**Recall**

最终使用 **F1** 作为细粒度覆盖率总分。

这种设计能够同时惩罚：

- **Hallucination**：文本说了图里没有的内容
- **Omission**：图里有关键内容，但文本没写

---

### 11.4 语义覆盖率（Semantic Coverage）

除细粒度对象/属性/关系覆盖外，我们还引入**整体语义层面**的覆盖率指标，用于衡量文本是否覆盖了图像主旨。

实现方式可选：

- 使用 **CLIP** 计算图文整体语义相似度
- 使用 **BLIP / VLM** 对图像和文本进行语义一致性评分
- 或者使用裁判型 VLM 给出图文整体匹配评分

该指标主要反映：

- 文本是否抓住了图像的核心语义
- 文本是否整体上“说对了图像在讲什么”

---

### 11.5 综合覆盖率指标

在最终评价中，可将细粒度覆盖与整体语义覆盖统一起来，定义综合得分：

```text
CoverageScore = α * Object + β * Attribute + γ * Relation + δ * Semantic
```

其中：

- `Object`：对象覆盖率
- `Attribute`：属性覆盖率
- `Relation`：关系覆盖率
- `Semantic`：整体语义覆盖率

这样既考虑结构化细节覆盖，也考虑整体语义一致性。

---

### 11.6 评价体系的鲁棒性保证（Robustness Guarantees）

为了防止评估过程中因模型遗漏信息而导致指标失真，本方案引入以下鲁棒性机制：

#### 1. 双向覆盖惩罚（Bidirectional Penalty）
评估不仅检查文本到图像的 Precision，也显式检查图像到文本的 Recall。  
这意味着：
- 文本出现幻觉会被惩罚
- 文本遗漏重要视觉内容同样会被惩罚

最终用 F1 统一两者。

#### 2. 任务解耦（Task Decoupling）
采用两阶段 Pipeline：

1. 纯文本 LLM 做原子化事实拆解
2. VLM 逐条做布尔验证

这样可以避免多模态模型在复杂任务中既负责“提取事实”又负责“视觉核查”时出现漏项问题，从工程上提高覆盖率评估的稳定性与可重复性。

---

---

## 12. 推荐复现实验顺序

若在新环境中从头运行，建议按如下顺序执行：

### Step 0. 创建环境
```bash
grep -v '^prefix:' environment_new.yml > /tmp/valik.environment.yml
conda env create -f /tmp/valik.environment.yml
conda activate valik
```

### Step 1. 准备 ScienceQA 数据
```bash
bash datasets/Preprocess_ScienceQA.sh
```


### Step 2. 跑 CoE
```bash
bash src/coe_batch.sh <IDs>
```

### Step 4. 跑 Prune
```bash
bash src/prune_batch.sh <IDs>
```

### Step 5. 跑评测（直接跑评测会自动生成CoE和Prune的结果）
```bash
bash src/evaluate_batch.sh <IDs>
```

---

## 13. 总结

本项目完成了 VaLiK 在 ScienceQA 上的降配复现，并实现了：

- CoE 三阶段图像描述生成
- CLIP 相似度过滤
- baseline / RAG / KG 三种评测模式

实验结果表明：

- 图像知识注入整体有效
- 当前实现下，**RAG > KG > baseline**
- RAG 在当前环境中保留了更多原始细节，因此表现最好
- KG 的结构化思路具有理论优势，但在当前轻量实现中受到抽取质量限制

因此，本项目既验证了 VaLiK 的核心流程可运行，也揭示了：

> **在资源受限条件下，文本 RAG 是更强、更稳的工程基线；而知识图谱路线要发挥优势，需要更高质量的结构化抽取与更强模型支持。**


因资源与时间限制未完成部分

- **尚未完成与论文主配置完全一致的大模型复现。**
   
   我们已经完成主流程复现与 baseline / RAG / KG 对比，但由于当前实验环境与论文所用高显存配置存在差距，暂未系统跑完 Qwen2-VL-7B / 72B、更强图构建与推理模型等更高配置实验。因此，本次结果更适合作为降配复现与趋势验证，而非对论文绝对指标的逐点对齐。这部分是代码已完成，由于资源时间而没有点到点复现。

- **尚未完成更大规模的扩展实验与 bonus 验证。**

   当前已完成 image-only 路线及核心评测流程，但对于更完整的 text-image KG、系统性的多模型消融，以及“文本描述覆盖率评估”方案的实验验证，仍需要额外的时间与算力支持，因此暂未纳入本次提交的主结果。

