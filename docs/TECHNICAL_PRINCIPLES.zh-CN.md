# 技术原理文档

> [English Technical Principles](https://github.com/xiaoyesoso/XIntent/blob/main/docs/TECHNICAL_PRINCIPLES.md)
>
> 相关文档：[HTTP API 接口文档](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.zh-CN.md) | [项目说明](https://github.com/xiaoyesoso/XIntent/blob/main/README.zh-CN.md)

本文档是一篇**技术博客**，不是干巴巴的参考手册。每个设计决策都配有真实场景类比、具体示例，以及"为什么这样设计"的思考过程。如果你正在构建 Agent，却发现意图识别总是不够稳定，这篇文档就是为你写的。

XIntent 框架包含两个模块，各自回答一个根本问题：

- **用户输入规范化**（第 1-13 节）：*为什么 Agent 听不懂用户在说什么？* -- 因为自然语言本身就是混乱的。这个模块在意图识别之前就把输入收拾干净。
- **三层意图识别**（第 14-22 节）：*为什么 Agent 总是选错意图？* -- 因为单次大模型调用既不够快也不够稳。这个模块用瀑布式架构平衡成本、速度和准确率。
- **准确率提升**（D17-D25）：*如何把准确率从 85% 推到 99%？* -- 八个可选扩展，每个针对一类特定失败模式。
- **面试洞察**（D26-D31）：*面试时怎么讲这个系统？* -- 六个来自真实面试反馈的设计决策，覆盖证据分级、边界原则、结构化输出契约。

每个原理都标注了对应的**服务接口**和**代码实现**。

---

## 目录

### 模块一：用户输入规范化

- [1. 问题背景：为什么 Agent 听不懂用户输入](#1-问题背景为什么-agent-听不懂用户输入)
- [2. 六大类输入问题的语言学基础](#2-六大类输入问题的语言学基础)
- [3. 两阶段流水线设计原理](#3-两阶段流水线设计原理)
- [4. 职责边界：为什么大模型会"越权"](#4-职责边界为什么大模型会越权)
- [5. 指代消解与跨轮记忆](#5-指代消解与跨轮记忆)
- [6. 按名索引：认知负荷与引用准确性](#6-按名索引认知负荷与引用准确性)
- [7. 澄清机制：不确定性处理](#7-澄清机制不确定性处理)
- [8. 属性检索指代消解：RAG 长期记忆](#8-属性检索指代消解rag-长期记忆)
- [9. 词汇表自我演进：涌现式知识](#9-词汇表自我演进涌现式知识)
- [10. 形容词可量化：从主观到客观](#10-形容词可量化从主观到客观)
- [11. 三层上下文整合](#11-三层上下文整合)
- [12. 完整性校验：守门员机制](#12-完整性校验守门员机制)
- [13. few-shot 检索注入：经济性与闭环](#13-few-shot-检索注入经济性与闭环)

### 模块二：三层意图识别

- [14. 为什么"直接调大模型"还不够](#14-为什么直接调大模型还不够)
- [15. 三层瀑布式架构](#15-三层瀑布式架构)
- [16. 置信度路由与多信号融合](#16-置信度路由与多信号融合)
- [17. 槽位填充：硬/软约束与跨轮累积](#17-槽位填充硬软约束与跨轮累积)
- [18. 拒识与澄清：双出口](#18-拒识与澄清双出口)
- [19. 下一轮隐式评估](#19-下一轮隐式评估)
- [20. 意图边界定义与 few-shot 注入](#20-意图边界定义与-few-shot-注入)
- [21. 分层意图与决策树](#21-分层意图与决策树)
- [22. 评估指标：4 层指标体系与 9 场景测试集](#22-评估指标4-层指标体系与-9-场景测试集)

### 扩展章节

- [准确率提升 (D17-D25)](#准确率提升-d17-d25)
- [面试洞察（D26-D31）](#面试洞察d26-d31)

---

## 1. 问题背景：为什么 Agent 听不懂用户输入

### 现实困境

在真实场景中，用户输入高度随意--正如日常说话一样随意。用户会使用病句、倒装句、代词、缩写，甚至"黑话"。这不是用户的问题，而是自然语言的本质特性：**自然语言是冗余的、模糊的、上下文依赖的**。

传统软件通过 UI 约束用户输入（表单、下拉框、校验规则），而 Agent 直接接收自然语言文本，失去了 UI 层的"输入规范化"保护。这导致：

- **意图识别失准**：主语缺失导致意图漂移（"市场占有率多少？" -- 谁的市场占有率？）
- **上下文断裂**：跨轮指代无法消解（第 3 轮"第二个"指什么？）
- **参数漂移**：形容词无量化导致工具调用参数不可控（"性价比"被理解为"便宜"）
- **事实伪造**：大模型"幻觉"编造不存在的信息

### 类比：需求分析

用户输入规范化相当于 Agent 醉域的"需求分析"。在传统软件工程中，需求分析是所有后续设计、开发、测试的基础；在 Agent 中，输入规范化是意图识别、工具调用、最终输出的基础。**输入不规范，后续全白搭**。

### 对应实现

整个框架通过 `POST /normalize` 接口对外提供服务。用户传入原始文本，服务返回结构化规范化结果：

```bash
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "第二个适合生产吗？", "session_id": "s1", "user_id": "u1", "turn": 2}'
```

- **服务入口**：`POST /normalize`（详见 [API 文档](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.zh-CN.md#3-post-normalize--用户输入规范化)）
- **编排器**：`NormalizationPipeline`（`src/user_input_normalization/pipeline.py`）
- **服务启动**：`python -m user_input_normalization.server`（`src/user_input_normalization/server.py`）

---

## 2. 六大类输入问题的语言学基础

本框架将用户输入问题归纳为六大类，每类都有对应的语言学/认知科学基础：

### 2.1 指代问题（Anaphora）

**语言学基础**：语用学中的"指代消解"（Anaphora Resolution）。代词本身没有固定含义，其指代对象依赖上下文。

**典型表现**：序号指代（"第二个适合生产吗？"）、时间指代（"刚才那个项目怎么包装？"）、属性指代（"看樱花的那个地方"）

**难点**：指代消解需要召回历史对话细节，跨多轮、跨时间跨度时短期记忆已失效。

### 2.2 缺失问题（Ellipsis）

**语言学基础**：省略句（Ellipsis）是自然语言中常见的经济性表达，说话者省略共享上下文中可推断的部分。

**典型表现**：缺主语（"市场占有率多少？"）、缺宾语（"帮我优化一下。"）、缺约束（"推荐一条裤子。"）

### 2.3 表达问题（Expression）

**语言学基础**：口语化表达与书面语的本质差异。口语是线性、跳跃、可改口的；书面语是结构化、完整的。

**典型表现**：语序混乱（"这个不太行，换个更像面试能讲的。"）、临时改口（"不是，我说的是另一个。"）

### 2.4 词义问题（Semantic）

**语言学基础**：语义学中的"一词多义"（Polysemy）和"行业方言"（Jargon）。同一词汇在不同领域有不同含义。

**典型表现**：缩写（RAG、CRM）、黑话（抓手、赋能、闭环、不够 P8）、同义词（知识库 / 检索增强 / RAG）

### 2.5 主观判断问题（Subjective）

**语言学基础**：主观性（Subjectivity）与评价理论（Appraisal Theory）。判断词的含义依赖个人偏好、场景和标准。

**典型表现**："哪个最有性价比？"、"再高级一点。"

**难点**：主观判断词必须转化为可量化参数，否则工具调用不可控。很多 Agent 处理粗暴："性价比高"直接等于"价格低"，忽略了质量、性能、品牌、售后。

### 2.6 外部事实问题（External Fact）

**语言学基础**：指代外部世界状态的表达，其真值依赖实时数据。

**典型表现**："最近哪个框架更火？"、"现在最便宜的是哪个？"

**难点**：无法在预处理阶段完成，必须依赖工具调用返回实时数据。不能伪造。

### 对应实现

分类结果通过 API 响应的 `classification_tags` 字段返回（数组，支持多标签）：

```json
{
  "classification_tags": ["指代问题", "主观判断问题"]
}
```

- **API 响应字段**：`classification_tags`（详见 [API 文档 - classification_tags 可选值](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.zh-CN.md#classification_tags-可选值)）
- **分类器**：`InputClassifier`（`src/user_input_normalization/classification/classifier.py`）
- **分类规则**：`CLASSIFICATION_RULES`（`src/user_input_normalization/classification/rules.py`，含 82 条词义规则 + 80 个已知术语）
- **路由逻辑**：`SUBJECTIVE`/`EXTERNAL_FACT` -> `deep` 阶段；其余 -> `pre` 阶段

---

## 3. 两阶段流水线设计原理

### 核心矛盾

用户输入规范化存在一个根本矛盾：
- **部分问题**（如"第二个"指代）可以在意图识别之前立即解决
- **部分问题**（如"现在最便宜的"外部事实）必须等到工具调用后才能解决

如果全部在预处理阶段做：跨时间指代和外部事实无法处理。
如果全部推迟到 ReAct 循环：简单指代也要等工具调用，增加不必要时延。

### 解决方案：两阶段分工

```
用户输入 -> [pre-normalization] -> 意图识别 -> [deep-normalization] -> 输出
              ↑                              ↑
         能立即解决的                    需要上下文/工具的
```

| 阶段 | 时机 | 处理内容 | 不处理内容 |
|------|------|----------|------------|
| pre | 意图识别之前 | 指代消解、省略补全、病句修正、术语标准化 | 依赖工具返回的量化、外部事实 |
| deep | ReAct 循环中 | 判断词量化、外部事实消解、Observation 回溯 | 已在 pre 完成的 |

### 对应实现

通过 API 请求的 `observation` 参数控制是否进入 deep 阶段，响应的 `stage_reached` 字段标识实际达到的阶段：

```bash
# 仅 pre 阶段（无 observation）
curl -X POST http://localhost:8000/normalize \
  -d '{"raw_input": "第二个适合生产吗？", "session_id": "s1", "turn": 2}'
# -> "stage_reached": "pre"

# 触发 deep 阶段（带 observation）
curl -X POST http://localhost:8000/normalize \
  -d '{"raw_input": "帮我推荐更有性价比的牛仔裤", "session_id": "s2", "turn": 1,
       "observation": {"current_price": 200}}'
# -> "stage_reached": "deep"
```

- **API 请求字段**：`observation`（可选，传入工具返回数据触发 deep 阶段）
- **API 响应字段**：`stage_reached`（`"pre"` 或 `"deep"`）
- **编排器**：`NormalizationPipeline.process()`（`src/user_input_normalization/pipeline.py`）
- **pre 阶段**：`PreNormalizer`（`src/user_input_normalization/pre_normalization/normalizer.py`）
- **deep 阶段**：`DeepNormalizer`（`src/user_input_normalization/deep_normalization/normalizer.py`）

---

## 4. 职责边界：为什么大模型会"越权"

### 大模型的"过度自主"问题

大模型被训练为"helpful assistant"，倾向于尽可能完成用户的请求。如果不约束，它会一股脑把输入规范、意图识别、回答问题全部干完：

- 用户问"市场占有率多少？"，大模型直接编造一个数字回答
- 用户问"帮我推荐裤子"，大模型直接推荐产品
- 用户问"现在最便宜的"，大模型伪造实时价格

这在规范化阶段是灾难性的：规范化的职责是**理解输入**，而非**回答问题**。

### 解决方案：显式"能做/不能做"清单

| 能做（8 项） | 不能做（8 项禁止） |
|------|--------|
| 指代消解 | 不直接回答用户问题 |
| 省略句补全 | 不执行工具 |
| 病句修正 | 不做最终推荐 |
| 术语标准化 | 不判断业务意图 |
| 黑话解释 | 不生成完整方案 |
| 多义词消歧 | 不擅自补充事实 |
| 判断是否需要澄清 | 不强行猜测低置信内容 |
| 判断是否需要搜索 | 不伪造实时信息 |

**原理**：显式清单是最简单、最有效的约束方式。模糊的角色描述（如"你是一个助手"）不足以约束大模型行为。清单将隐式期望转化为显式规则，便于审计与回归测试。

### 对应实现

职责边界通过 System Prompt 硬约束注入，规范化结果中**不会出现**对用户问题的直接回答：

- **Prompt 常量**：`SYSTEM_PROMPT`、`CAN_DO`、`CANNOT_DO`（`src/user_input_normalization/pre_normalization/prompts.py`）
- **边界校验**：`PreNormalizer.validate_boundary()` 检查是否越权（`normalizer.py`）
- **回归测试**：`tests/test_pre_normalization.py::TestResponsibilityBoundary` 验证不回答/不执行工具/不伪造事实

---

## 5. 指代消解与跨轮记忆

### 代词消解表格作为"关键事实"

指代消解不是一次性操作--消解结果需要**跨轮复用**：

```
第 3 轮: "第一个方案适合生产吗？" -> 消解: 第一个方案 -> TCC方案
第 5 轮: "那个 TCC 方案成本多少？" -> 按名索引直接命中
第 7 轮: "第一个呢？" -> 复用第 3 轮消解结果
```

如果每轮都重新消解：成本高、可能不一致、用户体验差。

### 对应实现

消解结果通过 API 响应的 `pronoun_resolutions` 字段返回，同一 `session_id` 下自动跨轮复用：

```json
{
  "pronoun_resolutions": [
    {
      "pronoun": "第二个",
      "resolved_to": "TCC方案",
      "confidence": 0.95,
      "evidence_source": "对话历史第3轮",
      "named_entity": "TCC方案"
    }
  ]
}
```

- **API 响应字段**：`pronoun_resolutions`（详见 [API 文档](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.zh-CN.md#pronoun_resolutions-元素结构)）
- **数据模型**：`PronounResolution`（`src/user_input_normalization/models.py`）
- **跨轮存储**：`KeyFactStore.find_pronoun_resolution()`（`src/user_input_normalization/storage/base.py`）
- **写入逻辑**：`PreNormalizer._write_pronoun_facts()`（`normalizer.py`）
- **跨轮复用测试**：`tests/test_pre_normalization.py::TestCrossTurnReuse`

---

## 6. 按名索引：认知负荷与引用准确性

### 序号引用的问题

如果 Agent 输出"方案一、方案二、方案三"，用户后续用"第二个"引用，大模型在推理阶段**容易引用错误**：序号无语义，大模型 attention 可能对错。

### 语义化命名的优势

如果 Agent 输出"TCC 方案、轻量方案、企业方案"，用户后续用"TCC 方案"引用：按名索引匹配准确率高、认知负荷低、规避推理错误。

**原理**：这是"正名"思维--名不正则言不顺，言不顺则事不成。序号是弱标识，语义化命名是强标识。

### 对应实现

- **API 响应字段**：`pronoun_resolutions[].named_entity`（语义化名称）
- **命名逻辑**：`PreNormalizer._ensure_named_entities()`（`normalizer.py`）
- **按名索引测试**：`tests/test_pre_normalization.py::TestPronounResolution::test_named_entity_set`

---

## 7. 澄清机制：不确定性处理

### 为什么不能"瞎猜"

误差累积是真实存在的风险：第 1 轮猜错 -> 第 2 轮基于错误继续推断 -> 最终答案完全偏离。

### 置信度阈值机制

```
置信度 >= 0.6 -> 直接消解
置信度 < 0.6  -> 触发澄清，询问用户
```

**原理**：这是"宁可不答，不可错答"的原则。阈值 θ_clarify 可配置（默认 0.6），平衡效率与准确率。

### 对应实现

当 `paused_for_clarification` 为 `true` 时，响应中包含 `clarification` 对象，客户端应暂停处理并向用户展示澄清问题：

```json
{
  "paused_for_clarification": true,
  "clarification": {
    "reason": "代词消解失败",
    "item": "那个帅气的同事",
    "candidates": [],
    "question": "您说的'那个帅气的同事'具体是指什么？请提供更多信息。",
    "confidence": 0.2
  }
}
```

- **API 响应字段**：`paused_for_clarification`、`clarification`（详见 [API 文档](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.zh-CN.md#clarification-结构)）
- **处理器**：`ClarificationHandler`（`src/user_input_normalization/clarification/handler.py`）
- **配置项**：`theta_clarify=0.6`、`max_consecutive_clarifications=3`（`src/user_input_normalization/config.py`）
- **恢复接口**：`NormalizationPipeline.resume_after_clarification()`（`pipeline.py`）

---

## 8. 属性检索指代消解：RAG 长期记忆

### 问题场景

用户假期前让 Agent 推荐出游地，Agent 推荐了鸡鸣寺看樱花。假期结束后用户输入："你上次推荐的看樱花的那个地方……"

难点：时间跨度长（短期记忆失效）+ 属性指代（"看樱花的"是属性修饰）。

### 两步推理 + 补偿机制

1. **属性提取 + 向量检索**：提取 ["樱花", "旅游"]，检索历史对话库召回细节
2. **大模型推断**：基于窗口内容推断 => "鸡鸣寺"
3. **补偿机制**（检索失败时）：提取属性发起工具调用，再次推理

### 对应实现

属性检索消解在流水线中自动触发（检测到属性指代模式时），消解结果回填到 `pronoun_resolutions`：

- **解析器**：`AttributeResolver`（`src/user_input_normalization/attribute_resolution/resolver.py`）
- **两步推理**：`AttributeResolver.resolve()` -> `extract_attributes()` -> `recall_details()` -> `infer()`
- **补偿机制**：`AttributeResolver.compensation()`（模拟工具调用 + 重新推理）
- **触发条件**：`NormalizationPipeline._is_attribute_anaphora()` 检测属性指代模式
- **测试**：`tests/test_attribute_resolution.py::TestJimingTempleCase`（鸡鸣寺案例端到端验证）

---

## 9. 词汇表自我演进：涌现式知识

### 问题

同样的词汇/缩写在不同行业、领域、上下文有不同含义。人工维护词汇表不现实--新词汇不断涌现。

### 自我迭代机制

```
线上对话 -> 离线分析 -> 标记候选词汇 -> 阈值判定 -> 升级入词汇表
```

**阈值规则**：总次数 > 100 / 讨论人数 > 3（公共词汇）/ 连续讨论 > 10 次

**原理**：这是"涌现式知识"--词汇表自下而上从实际对话中涌现，阈值过滤确保稳定性，人工审核兜底。

### 对应实现

术语标准化结果通过 API 响应的 `term_mappings` 字段返回：

```json
{
  "term_mappings": [
    {"original": "RAG", "standard": "检索增强生成", "source": "vocabulary-table"}
  ]
}
```

- **API 响应字段**：`term_mappings`（详见 [API 文档](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.zh-CN.md#term_mappings-元素结构)）
- **词汇表服务**：`VocabularyTable`（`src/user_input_normalization/vocabulary/table.py`）
- **离线分析器**：`OfflineAnalyzer`（`src/user_input_normalization/vocabulary/offline_analyzer.py`）
- **存储接口**：`VocabStore`（`src/user_input_normalization/storage/base.py`）
- **配置项**：`min_total_count=100`、`min_discussant_count=3`、`min_consecutive_count=10`（`config.py`）

---

## 10. 形容词可量化：从主观到客观

### 问题

任何未量化的形容词都会导致 Agent 输出漂移。很多 Agent 处理粗暴："性价比高"直接等于"价格低"，忽略质量、性能、品牌、售后。

### 量化策略

"性价比"有两种合理解释：

| 策略 | 含义 | 工具参数 |
|------|------|----------|
| 同等价格更好质量 | 同价位下，质量/配置/服务更好 | `price_range: [150, 200], quality_rank: top 30%` |
| 同等质量更低价格 | 同质量下，价格更便宜 | `price_range: [100, 150], quality_rank: top 50%` |

### 对应实现

量化结果通过 API 响应的 `quantifiable_adjectives` 字段返回，需在请求中传入 `observation`（如当前价格）才能完成量化：

```bash
curl -X POST http://localhost:8000/normalize \
  -d '{"raw_input": "帮我推荐更有性价比的牛仔裤", "session_id": "s2", "turn": 1,
       "observation": {"current_price": 200}}'
```

```json
{
  "quantifiable_adjectives": [
    {
      "adjective": "性价比",
      "quantified": true,
      "quantified_value": {"price_range": [150, 200], "quality_rank": "top 30%"},
      "route_to": null
    }
  ],
  "stage_reached": "deep"
}
```

- **API 请求字段**：`observation`（传入当前价格等上下文，触发 deep 阶段量化）
- **API 响应字段**：`quantifiable_adjectives`（详见 [API 文档](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.zh-CN.md#quantifiable_adjectives-元素结构)）
- **量化引擎**：`QuantificationEngine`（`src/user_input_normalization/quantification/engine.py`）
- **内置规则**：`DEFAULT_RULES`（6 条规则：性价比、划算、再高级一点等）（`quantification/rules.py`）
- **深度规范化**：`DeepNormalizer.quantify_adjectives()`（`deep_normalization/normalizer.py`）

---

## 11. 三层上下文整合

### 核心思想

> 所有的规范化措施都可以归结为一句话：深度整合上下文。

| 层级 | 内容 | 来源 | 用途 |
|------|------|------|------|
| 用户画像 | 行业、偏好、历史行为 | 用户画像存储 | 消歧倾向 |
| 关键事实 | 喜好、观点、态度、已认可点 | 关键事实存储 | 跨轮复用、补全依据 |
| 对话历史 | 摘要 + 细节 | 短期记忆 + RAG 长期记忆 | 指代消解证据、省略补全来源 |

### 对应实现

三层上下文在 `PreNormalizer.normalize()` 内部自动组装，通过 `user_id` 和 `session_id` 关联：

- **整合器**：`ContextIntegrator.assemble()`（`src/user_input_normalization/context/integrator.py`）
- **上下文模型**：`ContextBundle`（user_profile + key_facts + dialogue_summary + recalled_details）
- **优先级冲突**：`ContextIntegrator.resolve_conflict()`（关键事实 > 用户画像 > 对话历史）
- **存储接口**：`UserProfileStore`、`KeyFactStore`、`DialogueHistoryStore`（`storage/base.py`）
- **可观测性**：`ContextIntegrator.explain_influence()` 返回各层影响报告

---

## 12. 完整性校验：守门员机制

### 为什么需要校验

规范化不是"调一次大模型就完了"--大模型可能遗漏：主谓宾不完整、代词未完全消解、形容词未量化。

### 三项校验 + 路由

```
校验通过 -> 流入意图识别
缺主谓宾 / 代词未消解 -> 触发澄清机制（paused_for_clarification: true）
形容词未量化 -> 路由至 deep-normalization（stage_reached: "deep"）
```

### 对应实现

校验结果影响 API 响应的 `stage_reached` 和 `paused_for_clarification` 字段：

- **校验器**：`CompletenessChecker`（`src/user_input_normalization/pre_normalization/completeness_checker.py`）
- **校验方法**：`check_and_route()` 返回 `(校验结果, 路由目标, 路由原因)`
- **校验项**：`spo_complete`（主谓宾）、`pronouns_resolved`（代词）、`adjectives_quantified`（形容词）
- **API 影响**：未量化形容词 -> `stage_reached: "deep"`；代词未消解 -> `paused_for_clarification: true`

---

## 13. few-shot 检索注入：经济性与闭环

### 问题

few-shot 例子能显著提升规范化质量，但全量注入会撑爆上下文窗口。

### 检索注入 + 线上沉淀

```
用户输入 -> 向量化 -> 检索 top-k 相似例子 -> 注入 Prompt
线上遇到古怪输入 -> 成功消解 -> 整理为 few-shot 例子入库 -> 下次可参考
```

**原理**：这是"数据飞轮"--线上运行越久，例子库越丰富，规范化质量越高。

### 对应实现

few-shot 注入在 `PreNormalizer.normalize()` 内部自动执行，对 API 调用方透明：

- **例子存储**：`FewShotStore`（`src/user_input_normalization/storage/base.py`）
- **检索注入**：`PreNormalizer._build_user_prompt()` 调用 `FewShotStore.search(top_k=3)`
- **线上沉淀**：`PreNormalizer._sink_to_fewshot()` 将古怪输入自动入库
- **格式化**：`format_fewshot()`（`pre_normalization/prompts.py`）
- **配置项**：`top_k=3`、`enabled=True`（`config.py`）

---

## 总结

用户输入规范化（模块一）的核心思想可以概括为：

1. **深度整合上下文** -- 所有的措施都围绕这一句话
2. **两阶段分工** -- 能立即解决的在 pre，需要工具的在 deep
3. **显式约束** -- 职责边界、完整性校验、澄清阈值，防止大模型越权和瞎猜
4. **结构化输出** -- 消解表格、主谓宾、量化字段，让下游可程序化消费
5. **自我演进** -- 词汇表迭代、few-shot 沉淀，系统越用越聪明

三层意图识别（模块二）的核心思想可以概括为：

1. **瀑布式升级** -- 从便宜快的开始，按需升级（Code → Flash → Pro）
2. **多信号置信度** -- 绝不只信大模型自报的 confidence；融合规则命中、向量相似度、历史准确率
3. **槽位填充是识别的一部分** -- 意图 + 参数 + 约束是一个结构化输出，而非独立阶段
4. **不确定性的双出口** -- 不支持的意图给出原因拒识；不清晰的意图触发有界澄清循环
5. **闭环评估** -- 测试集 + 下一轮隐式信号回填指标，持续改进

### 原理与接口映射总表

#### 模块一：用户输入规范化

| 技术原理 | API 请求字段 | API 响应字段 | 核心类 |
|----------|-------------|-------------|--------|
| 六大类分类 | - | `classification_tags` | `InputClassifier` |
| 两阶段流水线 | `observation` | `stage_reached` | `NormalizationPipeline` |
| 职责边界 | - | （不出现回答内容） | `SYSTEM_PROMPT` / `CAN_DO` / `CANNOT_DO` |
| 指代消解 | `session_id` | `pronoun_resolutions` | `PreNormalizer` / `KeyFactStore` |
| 按名索引 | - | `pronoun_resolutions[].named_entity` | `PreNormalizer` |
| 澄清机制 | - | `paused_for_clarification` / `clarification` | `ClarificationHandler` |
| 属性检索消解 | `session_id` | `pronoun_resolutions`（回填） | `AttributeResolver` |
| 词汇表 | - | `term_mappings` | `VocabularyTable` / `OfflineAnalyzer` |
| 形容词量化 | `observation` | `quantifiable_adjectives` | `QuantificationEngine` / `DeepNormalizer` |
| 省略补全 | - | `completions` | `PreNormalizer` |
| 完整性校验 | - | `stage_reached` / `paused_for_clarification` | `CompletenessChecker` |
| few-shot 注入 | - | （内部自动） | `FewShotStore` / `PreNormalizer` |

#### 模块二：三层意图识别

| 技术原理 | API 请求字段 | API 响应字段 | 核心类 |
|----------|-------------|-------------|--------|
| 三层瀑布 | `text` | `source` / `layer_reached` | `IntentRecognitionPipeline` |
| 置信度路由 | - | `confidence` / `need_clarification` | `ConfidenceRouter` |
| 多信号融合 | - | `confidence` | `MultiSignalFuser` |
| 槽位填充 | - | `slots` / `missing_slots` | `SlotExtractor` |
| 硬/软约束 | - | `hard_constraints` / `soft_constraints` | `ConstraintExtractor` |
| 跨轮槽位累积 | `session_id` / `turn` | `slots`（累积后） | `CrossTurnSlotMerger` / `SlotStateStore` |
| 拒识（不支持） | - | `rejection_reason` | `RejectionClarificationHandler` |
| 澄清（不清晰） | - | `need_clarification` / `clarification_question` | `RejectionClarificationHandler` |
| 意图切换 | `session_id` | `intent_switched` / `previous_intent` | `IntentHistoryStore` |
| 页面引导（代码层） | `event` | `intent` / `source: "code-layer"` | `PageGuidanceMatcher` |
| 分层意图 | - | （意图带 `parent_intent`） | `IntentRegistry` |
| 评估指标 | - | （通过 `TestRunner`，不走 HTTP） | `MetricsCalculator` / `TestSet` |

---

## 14. 为什么"直接调大模型"还不够

### 面试场景

当被问到"你的 Agent 怎么做意图识别？"很多候选人回答"调个大模型分类"。在工程上这是错的，原因有三：

1. **成本与时延**：单次深度推理调用可能耗时 5-15 秒。大多数用户输入（"继续"、"下一个"、"退款"）都是无歧义的，根本不需要它。
2. **置信度不是概率**：大模型自报的 `confidence` 是"自评得分"，不是校准过的概率。把它当真值会导致错误答案被过度接受。
3. **没有双出口**：单次 LLM 调用只产生一个答案。工程上需要显式处理"不支持"（意图不在候选列表中）和"不清晰"（低置信度或缺必填槽位）—— 两种截然不同的失败模式。

### 对应实现

本框架引入三层瀑布结构，同时解决上述三个问题：

- **服务入口**：`POST /recognize`（详见 [API 文档](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.zh-CN.md#4-post-recognize--意图识别)）
- **编排器**：`IntentRecognitionPipeline`（`src/intent_recognition/pipeline.py`）
- **设计决策**：D1（瀑布）、D2（独立阶段）、D10（双出口）

---

## 15. 三层瀑布式架构

### 级联流程

```
规范化后的输入 -> [L1 代码层] --未命中--> [L2 轻量 LLM] --低置信度--> [L3 深度 LLM]
                       |                       |                              |
                       +----- 命中(conf=1.0) --+---------- 高置信度 ----------+
                                              |
                                     [槽位填充] -> [拒识 / 澄清]
                                              |
                                     IntentRecognitionResult
```

**L1 代码层**（D3）：三种识别方式 —— 页面引导（UI 事件 → 意图）、关键字/正则匹配、规则引擎（上下文状态 + 规范化输入）。返回第一个命中或"未命中"信号。**零 LLM 调用，< 10ms。**

**L2 轻量 LLM**（D4, D5）：Flash/Mini 模型 + 结构化 Prompt（候选意图 + 槽位 + few-shot）。置信度路由：`>=0.85` 接受，`0.6-0.85` 澄清，`<0.6` 升级到 L3。

**L3 深度 LLM**（D6）：深度推理模型处理 5 类复杂场景 —— 复杂表达（病句/倒装）、跨轮上下文依赖、意图切换、多意图分解、隐式信息补全。

### 为什么是瀑布，不是并行

三层并行会让 API 成本变 3 倍，且 Flash 模型的置信度与 Pro 模型的置信度不可直接比较。瀑布只在未命中或低置信度时升级，所以大多数请求止步于 L1 或 L2。

### 优雅降级

如果 L3 失败（超时、API 报错），流水线会回退到 L2 的结果并打 warning 日志，而不是整个请求失败。

### 对应实现

- **L1 代码层**：`src/intent_recognition/code_layer/`（`classifier.py`、`page_guidance.py`、`keyword_matcher.py`、`rule_engine.py`）
- **L2 轻量 LLM**：`src/intent_recognition/lightweight_llm/`（`classifier.py`、`confidence_router.py`、`multi_signal.py`、`prompts.py`）
- **L3 深度 LLM**：`src/intent_recognition/deep_llm/`（`classifier.py`、`prompts.py`）
- **编排器**：`IntentRecognitionPipeline.recognize()`（`src/intent_recognition/pipeline.py`）
- **配置开关**：`enable_code_layer`、`enable_lightweight_llm`、`enable_deep_llm`（`src/intent_recognition/config.py`）

---

## 16. 置信度路由与多信号融合

### LLM 置信度的问题

大模型自报的置信度是"自评得分" —— 反映的是模型内部的确定性，不是校准过的概率。两种失败模式：

- **错答时过度自信**：模型对一个错分类的意图报 0.9 置信度
- **对答时不够自信**：模型对一个正确意图报 0.5 置信度，触发不必要的升级

### 多信号融合（D5）

本框架融合四个信号：

| 信号 | 权重 | 来源 |
|------|------|------|
| LLM 置信度 | 0.5 | 模型输出 |
| 规则命中 | 0.2 | 代码层关键字/正则在同一意图上命中 |
| 向量相似度 | 0.2 | 与正样本例子的 embedding 相似度 |
| 历史准确率 | 0.1 | 评估存储中按意图统计的准确率 |

融合后的置信度按以下规则路由：

- `>= 0.85` → 接受
- `>= 0.6 且 < 0.85` → 澄清（问用户）
- `< 0.6` → 升级到 L3

### 对应实现

- **置信度路由器**：`ConfidenceRouter`（`src/intent_recognition/lightweight_llm/confidence_router.py`）
- **多信号融合**：`MultiSignalFuser`（`src/intent_recognition/lightweight_llm/multi_signal.py`）
- **配置**：`ConfidenceConfig`（`src/intent_recognition/config.py`）—— `accept_threshold=0.85`、`clarify_threshold=0.6`、各信号权重字段

---

## 17. 槽位填充：硬/软约束与跨轮累积

### 槽位填充是识别的一部分（D7）

意图识别不只是"什么意图" —— 而是"什么意图 + 什么参数 + 什么约束"。槽位填充被集成进识别流水线，不是独立阶段。

### 硬约束 vs 软约束（D8）

| 约束类型 | 示例 | 语义 |
|---------|------|------|
| 硬约束 | "不超过3000"、"必须华为"、"不要二手" | 必须满足；用于过滤结果 |
| 软约束 | "最好轻薄"、"优先蓝色"、"续航越长越好" | 尽量满足；用于排序或加权 |

本框架用正则模式处理常见的中文约束表达，复杂情况回退到 LLM 抽取。约束以规范化表达式（如 `price<=3000`）+ 原始文本返回，便于追溯。

### 跨轮累积（D9）

在 ReAct/TAO 循环中，用户会跨轮迭代细化自己的请求：

```
第 1 轮: "帮我推荐一款3000元以内的手机"  -> {category: 手机, budget_max: 3000}
第 2 轮: "预算可以到4000"                -> {category: 手机, budget_max: 4000}  （最新优先）
第 3 轮: "最好是华为"                    -> {category: 手机, budget_max: 4000, brand: 华为}  （累积）
```

**最新优先策略**：当同一槽位在新轮次中出现，新值覆盖旧值。冲突会被记录用于调试。

**约束累积**：约束不覆盖 —— 唯一的表达式追加进列表。`has_constraint_conflict()` 检测相互矛盾的硬约束（如 `price<100` 与 `price>200`）。

### 对应实现

- **槽位抽取器**：`SlotExtractor`（`src/intent_recognition/slot_filling/extractor.py`）
- **约束抽取器**：`ConstraintExtractor`（`src/intent_recognition/slot_filling/constraints.py`）
- **跨轮合并器**：`CrossTurnSlotMerger`（`src/intent_recognition/slot_filling/cross_turn.py`）
- **槽位状态存储**：`SlotStateStore`（`src/intent_recognition/storage/base.py`，内存实现在 `memory.py`）
- **API 字段**：`slots`、`missing_slots`、`hard_constraints`、`soft_constraints`

---

## 18. 拒识与澄清：双出口

### 两种截然不同的失败模式

| 模式 | 触发条件 | 响应 |
|------|---------|------|
| 不支持（拒识） | 意图不在候选列表 | `rejection_reason` 列出支持的意图 |
| 不清晰（澄清） | 低置信度或缺必填槽位 | `clarification_question` 询问用户 |

一句"我不知道"会把两种情况混在一起 —— 用户分不清是该重新表述（不支持）还是补充信息（不清晰）。

### 拒识：告诉用户什么是支持的

当意图不在候选列表时，拒识原因包含：

1. 不支持的内容（原样回显）
2. 支持的意图列表，含描述和正样本
3. 来自负样本的重定向提示（如 `"iPhone 16的电池容量是多少 -> product_query"`）

### 澄清：带收敛追踪的有界循环

澄清不是无限的。框架按 session 追踪连续澄清次数：

- 达到 `max_consecutive_clarifications`（默认 3）后，系统降级为"最佳猜测"模式 —— 接受置信度最高的候选，而不是再次询问。
- 如果用户的下一轮解除了澄清（如补上了缺失的槽位），连续计数器重置。

### 对应实现

- **处理器**：`RejectionClarificationHandler`（`src/intent_recognition/rejection_clarification/handler.py`）
- **配置**：`ClarificationConfig.max_consecutive_clarifications=3`（`src/intent_recognition/config.py`）
- **API 字段**：`rejection_reason`、`need_clarification`、`clarification_question`

---

## 19. 下一轮隐式评估

### 显式反馈的问题

每次意图识别后问用户"我理解对了吗？"是糟糕的 UX。大多数用户不会答，答的人又有偏差 —— 倾向于报错而不报对。

### 隐式信号检测（D11）

本框架在**下一轮的用户输入**中检测失败信号：

```
第 N 轮: "帮我推荐手机"  -> intent: product_recommendation
第 N+1 轮: "你理解错了，我要查订单"  -> 检测到第 N 轮的失败信号
```

默认失败信号：`"你理解错了"`、`"不是这个意思"`、`"不是，我不是"`、`"不对，我要的是"`。

检测到失败信号时：

1. 上一轮的意图在评估存储中被标记为错误
2. 当前轮的意图重新识别（失败信号中常包含正确的意图）

### 收敛追踪

对澄清循环，框架追踪澄清后用户的任务是否进展：

- **已解除**：下一轮输入是正常请求（非失败信号，非又一次澄清）→ 重置连续计数器
- **未解除**：下一轮输入是又一次澄清响应 → 递增连续计数器

### 对应实现

- **失败信号检测**：`RejectionClarificationHandler.detect_implicit_failure()`（`src/intent_recognition/rejection_clarification/handler.py`）
- **配置**：`IntentRecognitionConfig.failure_signals`（`src/intent_recognition/config.py`）
- **评估存储**：`EvaluationStore`（`src/intent_recognition/storage/base.py`）

---

## 20. 意图边界定义与 few-shot 注入

### 边界问题

意图之间有重叠。"iPhone 16的电池容量是多少"既可能是 `product_query`（查参数）也可能是 `product_recommendation`（按电池推荐）。没有显式边界，大模型会猜得不一致。

### 正样本 + 负样本（D12）

每个 `IntentDefinition` 包含：

- **正样本**：明确属于该意图的输入
- **负样本**：看似相似但属于别的意图的输入，带重定向目标（如 `"3000以内有什么手机推荐 -> product_recommendation"`）

这些被注入 L2 Prompt 用于界定边界。

### few-shot 检索注入

对易混淆的意图对，框架检索 top-k 个最相似的历史例子（来自 `FewShotStore`）注入 Prompt。这与模块一第 13 节的 few-shot 注入是同一机制，跨模块复用。

### 对应实现

- **意图注册表**：`IntentRegistry`（`src/intent_recognition/intent_registry.py`）—— 存储带正/负样本的 `IntentDefinition`
- **Prompt 构造器**：`build_prompt()`（`lightweight_llm/prompts.py`）—— 注入候选意图、槽位、边界、few-shot
- **few-shot 存储**：`FewShotStore`（与模块一共享，`src/user_input_normalization/storage/base.py`）
- **配置**：`fewshot_top_k=3`、`fewshot_enabled=True`（`src/intent_recognition/config.py`）

---

## 21. 分层意图与决策树

### 上下文窗口问题

当意图空间增长到 20+ 个时，把它们全列在 LLM Prompt 中会撑爆上下文窗口并降低准确率。

### 分层调度（D13）

本框架支持父/子意图关系：

```python
IntentDefinition(name="refund", parent_intent="order_query", ...)
```

启用分层调度时：

1. L1/L2 先识别父意图（如 `order_query`）
2. 候选列表收窄到 `order_query` 的子意图（如 `refund`、`exchange`、`cancel`）
3. L1/L2 在收窄后的列表上再跑一次

这把扁平的 100 意图问题变成两个 10 意图问题。

### 何时启用

分层调度**默认关闭**（`enable_hierarchical=False`），因为它会增加时延（两次识别）。只有当意图空间大到影响 L2 准确率时才启用。

### 对应实现

- **意图注册表**：`IntentRegistry`（`src/intent_recognition/intent_registry.py`）—— `parent_intent` 字段、`get_children()` 方法
- **配置开关**：`enable_hierarchical`（`src/intent_recognition/config.py`）
- **默认意图**：服务端 `_build_default_registry()`（`server.py`）将 `refund` 注册为 `order_query` 的子意图

---

## 22. 评估指标：4 层指标体系与 9 场景测试集

### 4 层指标（D15）

| 层级 | 指标 | 衡量内容 |
|------|------|---------|
| 1. Top-K 准确率 | Top-1、Top-3 | 正确意图是否在 top-K 中 |
| 2. 按意图 | 每意图准确率 | 哪些意图容易被混为哪些 |
| 3. 拒识/澄清 | 拒识准确率、误拒率、漏拒率、澄清触发准确率、澄清后收敛率 | 双出口是否工作正常 |
| 4. 槽位填充 | 槽位准确率、槽位召回率、必填槽位完整率、槽位更新准确率、约束识别准确率 | 参数抽取是否正确 |

### 基准目标

- **领域专用 Agent**：0.99（本框架的目标市场）
- **通用 Agent**：0.85（基线）

### 9 场景测试集（D16）

本框架内置测试集，覆盖 9 类场景：

1. 清晰/简单意图（L1/L2 命中）
2. 复杂表达（需 L3）
3. 多意图输入（分解）
4. 跨轮上下文依赖
5. 意图切换
6. 槽位填充（缺必填槽位）
7. 拒识（不支持的意图）
8. 澄清（不清晰的意图）
9. 分层调度

### 对应实现

- **指标计算器**：`MetricsCalculator`（`src/intent_recognition/evaluation/metrics.py`）
- **测试集**：`TestSet`（`src/intent_recognition/evaluation/test_set.py`）—— 9 种场景类型、样例用例
- **测试运行器**：`TestRunner`（`src/intent_recognition/evaluation/runner.py`）—— 对流水线跑测试集，生成报告
- **配置**：`EvaluationConfig`（`src/intent_recognition/config.py`）—— `top_k=3`、基准目标

---

## 准确率提升 (D17-D25)

以下可选能力（全部默认关闭）针对生产环境中识别出的五类意图识别准确率问题。每项均可通过配置或 `POST /agent/intent` 的按请求覆盖标志启用。

### 问题-方案映射 (D25)

| 问题 | 方案 | 配置 |
|---------|----------|--------|
| 意图空间过大 → L2 混淆 | D17: 检索式候选收窄 | `retrieval.enable` |
| 静态 few-shot 不足 | D18: 动态 few-shot 注入 | `dynamic_fewshot.dynamic_enabled` |
| 意图边界重叠 | D19: 意图正交性检查 | `orthogonality.enable_check` |
| 伪多意图（流程描述） | D20: 真多意图分解 | `multi_intent.enable` |
| 代码层未命中（无关键字匹配） | D21: 向量匹配兜底 | `vector_fallback.enable` |
| 同一任务不必要的重复识别 | D22: 意图复用与回滚 | `reuse_strategy.enable` |
| 单识别器不确定 | D23: 多识别器仲裁 | `arbiter.enable` |
| 需要领域专用模型 | D24: 微调集成点 | `fine_tuning.enable` |

### D17: 检索式候选收窄

在 Layer 2 LLM 分类之前，使用向量相似度、LLM 粗分类或混合检索召回 Top-N 候选意图。N 值动态调整（意图空间大 → 候选多；小 → 候选少）。

- **配置**：`RetrievalConfig(enable=False, method="vector"|"llm_coarse"|"hybrid", top_n=10, dynamic_n=True)`
- **按请求覆盖**：在 `POST /agent/intent` 和 `POST /recognize` 上传 `enable_retrieval: true|false`
- **实现**：`src/intent_recognition/lightweight_llm/candidate_retriever.py`

### D18: 动态 few-shot 注入

在静态 few-shot 之外，注入动态检索到的历史例子（与当前输入相似）。使用 Jaccard token 重叠度从 `FewShotStore` 中查找相似的历史输入。

- **配置**：`DynamicFewShotConfig(dynamic_enabled=False, dynamic_top_k=3, static_kind_tag="static")`
- **实现**：`src/intent_recognition/lightweight_llm/dynamic_fewshot.py`

### D19: 意图正交性检查

检测意图定义之间的重叠（相似度 > 阈值），并提供 `split_intent()` / `merge_with_param()` 操作以维护干净、正交的意图空间。

- **配置**：`OrthogonalityConfig(enable_check=False, overlap_threshold=0.7)`
- **实现**：`IntentRegistry.detect_overlap()`、`split_intent()`、`merge_with_param()`（`src/intent_recognition/intent_registry.py`）

### D20: 真多意图分解

区分真多意图请求与流程描述（如"推荐手机然后结账"）。返回子意图之间的依赖关系以及拓扑排序后的待执行列表。

- **配置**：`MultiIntentConfig(enable=True, sequential_execution=True, filter_process_description=True)`
- **响应字段**：`relations: [{src, dst, constraints}]`、`pending_intents: ["intent_a", "intent_b"]`
- **实现**：`src/intent_recognition/deep_llm/classifier.py`（解析 + 拓扑排序）

### D21: 向量匹配兜底

当 Layer 1 代码层未命中（无关键字/正则/规则匹配）时，回退到对 `VectorMatchStore` 的向量相似度匹配。使用 128 维哈希向量化器（纯 Python 实现，无 numpy 依赖）和余弦相似度。

- **配置**：`VectorFallbackConfig(enable=False, similarity_threshold=0.92, top_k=1)`
- **按请求覆盖**：`enable_vector_fallback: true|false`
- **实现**：`src/intent_recognition/code_layer/vector_matcher.py`
- **source 取值**：`"vector-fallback"`

### D22: 意图复用与回滚

当用户跨轮继续同一任务时，直接复用上一轮的意图（confidence=1.0, source="reused"），跳过整个瀑布。在出现隐式失败信号（D11）或意图切换标记时回滚。

- **配置**：`ReuseStrategyConfig(enable=False, rollback_on_failure_signal=True, rollback_on_tool_failure_count=3)`
- **按请求覆盖**：`reuse_previous_intent: true|false`
- **意图切换标记**：`"换个话题"`、`"不是这个"`、`"我要问别的"`、`"切换"`、`"换一个"`、`"不是说这个"`、`"我想问的是"`
- **实现**：`src/intent_recognition/intent_reuse_strategy.py`
- **source 取值**：`"reused"`

### D23: 多识别器仲裁

并行运行多个识别器（向量、规则、轻量 LLM），通过投票（多数）或加权评分（可配置权重）进行仲裁。适用于单一识别器不确定的场景。

- **配置**：`ArbiterConfig(enable=False, mode="vote"|"weighted_score", recognizers=["vector","rule","lightweight_llm"], weights={"vector":0.8,"rule":0.6,"lightweight_llm":1.0})`
- **实现**：`src/intent_recognition/multi_recognizer_arbiter.py`
- **source 取值**：`"arbiter-vote"` 或 `"arbiter-weighted"`

### D24: 微调集成点

提供训练数据导出流水线（JSONL 格式）和 `model_tier="fine_tuned"` LLM 客户端层级，读取 `FINE_TUNED_MODEL` 环境变量。仅设计层面 —— 不执行实际训练；导出的数据可用于在外部微调领域专用模型。

- **配置**：`FineTuningConfig(enable=False, model=None)`
- **实现**：`src/intent_recognition/training_data_exporter.py`、评估运行器中的 `evaluate_fine_tuned()`
- **环境变量**：`FINE_TUNED_MODEL` 用于微调模型名

### 按请求覆盖模式

所有 D17/D21/D22 能力都支持通过 `POST /agent/intent` 和 `POST /recognize` 请求体的按请求覆盖：

| 字段 | 类型 | 默认值 | 效果 |
|-------|------|---------|--------|
| `enable_retrieval` | `bool \| null` | `null` | 覆盖 D17（`null`=使用配置，`true`/`false`=强制）|
| `enable_vector_fallback` | `bool \| null` | `null` | 覆盖 D21 |
| `reuse_previous_intent` | `bool \| null` | `null` | 覆盖 D22 |

覆盖会作用于流水线配置的快照，并在请求完成后恢复，因此不会跨共享单例流水线的请求泄漏。

### 准确率预设

`accuracy_preset` 配置字段（默认 `"balanced"`）提供了一次性启用多项能力的快捷方式。可用预设：
- `"balanced"`（默认）：所有 D17-D24 关闭，依赖三层瀑布
- `"high_accuracy"`：启用 D17 + D21 + D22 以获得最高准确率（成本更高）
- `"low_cost"`：全部关闭（同 balanced，显式声明）

## 面试洞察（D26-D31）

六项设计决策，提炼自意图识别面试材料（简历写法、课后习题、复用模板）。它们以证据分级、严格的子任务边界、五因子路由仲裁、扩展评估指标、方案演进方法论和统一结构化输出协议补充 D1-D25。六项默认全部开启，因为它们是增量式的（丰富输出而不改变既有行为）。

### D26：证据分级

每个槽位值和事实都标注证据等级：

- **`verified`（已验证）**——来自当前输入、活跃上下文或已确认的 `KeyFact`。硬操作（退款、支付）可直接执行。
- **`provisional`（暂定）**——来自历史、用户画像、推测或默认值。硬操作必须升级证据（用户确认 / Observation 校验）或披露假设。

流水线通过 `_collect_evidence()` 收集证据，通过 `_upgrade_provisional_to_verified()` 升级暂定证据，并在 `evidence.require_verified_for_hard_ops=True` 且意图属于 `evidence.high_risk_intents` 时通过 `_check_hard_op_evidence()` 执行硬操作前校验。

- **配置**：`EvidenceConfig(enable_grading=True, require_verified_for_hard_ops=True, high_risk_intents=[])`
- **实现**：`IntentRecognitionPipeline._collect_evidence()` / `_upgrade_provisional_to_verified()` / `_check_hard_op_evidence()`，位于 `src/intent_recognition/pipeline.py`
- **模型**：`Evidence`、`EvidenceGrade`、`SlotValue`，位于 `src/intent_recognition/models.py`

### D27：sub_tasks vs independent_intents 边界

**意图的职责是选择业务流程。** 只有能独立触发*不同*业务流程的目标才算 `independent_intents`；主流程内的步骤（比价、查配送、查参数）应归入 `sub_tasks`、槽位或约束。

- `independent_intents` 进入 D20 多意图治理（过程性过滤、`relations`、`pending_intents`、顺序执行）。
- `sub_tasks` 作为主流程内的执行步骤记录，**不进入** `relations`、`pending_intents` 或过程性过滤。

深度 LLM 提示词（规则 6-10）和轻量 LLM（`detect_boundary_simple()` 配合 `_SUB_TASK_PATTERNS` / `_INDEPENDENT_MARKERS`）共同强制此边界。

- **配置**：`BoundaryConfig(enable_sub_tasks=True, strict_mode=False)`
- **实现**：`src/intent_recognition/deep_llm/prompts.py`（规则 6-10）、`src/intent_recognition/deep_llm/classifier.py`、`src/intent_recognition/lightweight_llm/classifier.py`

### D28：五因子路由仲裁

置信度是**信号**，不是概率。当 L2 置信度落在模糊区间 `[clarify_threshold, accept_threshold)` 时，按优先级顺序执行五因子仲裁：

| # | 因子 | 失败动作 |
|---|------|---------|
| 1 | 规则校验 | 调整置信度（±0.05） |
| 2 | 槽位完整性 | 澄清 |
| 3 | 硬约束风险（高风险意图 + 暂定必填槽位） | 澄清 |
| 4 | 候选间距（Top1-Top2 < 阈值） | 升级到 L3 |
| 5 | 置信度 + 历史准确率 | 历史差且调整后置信度 < 接受阈值则澄清 |

第一个强制澄清/升级的失败因子胜出。因子 5 复用 D5 多信号融合分数，不替代它。

对于 L2/L3 不一致，`arbitrate_l2_l3()` 规则：L3 高置信度 + 槽位完整 → 接受 L3；两者均低于澄清阈值 → 强制澄清；否则对 L3 结果执行五因子仲裁。

- **配置**：`ArbitrationConfig(enable_five_factor=True, candidate_gap_threshold=0.1, risk_aware_clarify=True, high_risk_intents=[])`
- **实现**：`src/intent_recognition/lightweight_llm/confidence_router.py`（`ArbitrationInput`、`ArbitrationDecision`、`arbitrate()`、`arbitrate_l2_l3()`）
- **输出**：`IntentRecognitionResult` 的 `arbitration_breakdown` 字段（与 `signals` 分离，因为 `signals` 类型为 `dict[str, float]`）

### D29：扩展评估指标

在 D15 四层指标基础上，D29 新增 9 项细粒度指标和线上反馈回流：

| 指标 | 方法 | 衡量内容 |
|------|------|---------|
| 混淆矩阵 | `compute_confusion_matrix()` | N×N 期望 vs 预测 |
| 困难样本准确率 | `compute_hard_sample_accuracy()` | `is_hard=True` 样本的准确率 |
| 拒识准确率 | `compute_rejection_accuracy()` | `false_reject_rate`、`missed_reject_rate`、`rejection_precision`（英文键） |
| 澄清收敛率 | `compute_clarification_convergence_rate()` | 首轮澄清即收敛的比例 |
| 槽位召回率 | `compute_slot_recall()` | `correct_slots / expected_slots` |
| 槽位完整率 | `compute_slot_completeness()` | 必填槽位全部填充的会话比例 |
| 约束识别率 | `compute_constraint_identification_rate()` | `correct_constraints / total_constraints` |
| 状态更新准确率 | `compute_state_update_accuracy()` | 多轮槽位更新正确率 |

**线上反馈回流**：`TestSet.import_online_samples()` 将生产失败样本（通过 `TestRunner.collect_online_failures()` 利用 D11 失败信号收集）回灌到离线测试集。

- **配置**：`ExtendedEvaluationConfig` 逐项开关
- **实现**：`src/intent_recognition/evaluation/metrics.py`、`src/intent_recognition/evaluation/test_set.py`、`src/intent_recognition/evaluation/runner.py`

### D30：方案演进方法论

框架分三个阶段演进，每个阶段解决前一阶段的准确率瓶颈：

| 阶段 | 架构 | 关键机制 |
|------|------|---------|
| **v0** | 单深度 LLM，无规范化 | 暴力意图识别 |
| **v1-stage1** | + L2 轻量 LLM | 降成本、置信度路由 |
| **v1-stage2** | + L1 代码层 + D5 多信号融合 | 三层瀑布、<10ms 快路径 |
| **v1-stage3** | + D17-D24 高级机制 | 检索、动态 fewshot、向量回退、复用、仲裁、微调 |
| **v2** | + D26-D31 面试洞察 | 证据分级、边界、仲裁、扩展评估、协议 |

`TestRunner.assess_evolution_stage(config)` 检查配置并返回当前阶段字符串。可用于面试中表达系统成熟度，或用于功能预期门控。

**60 秒面试表达骨架**（填空模板）：

> "我们将意图识别建模为 ___ 层瀑布：代码层负责 <10ms 的关键字/正则命中，轻量 LLM 处理模糊中间地带，深度 LLM 应对 ___ 复杂场景。置信度被视为 ___（信号），因此在接受前执行五因子 ___（仲裁）。每个槽位值携带证据 ___（等级）：已验证或暂定；硬操作要求 ___（已验证）证据否则必须披露假设。我们用 ___（9）项指标评估，包括混淆矩阵和线上失败反馈回流，系统按 v0 → v1 → v2 阶段演进。"

- **实现**：`src/intent_recognition/evaluation/runner.py::assess_evolution_stage()`

### D31：统一输出协议

10 字段协议作为规范化与意图识别之间、以及意图识别与下游 ReAct/TAO 循环之间的交接契约：

| # | 字段 | 来源 | 默认值 |
|---|------|------|--------|
| 1 | `normalized_query` | 规范化输出（跳过时为原始输入） | `/recognize` 上为 `""` |
| 2 | `intent` | 识别结果 | `null` |
| 3 | `sub_tasks` | D27 流程内步骤 | `[]` |
| 4 | `independent_intents` | D27 多流程目标 | `[]` |
| 5 | `slots` | 槽位抽取器 | `{}` |
| 6 | `missing_slots` | 槽位抽取器 | `[]` |
| 7 | `hard_constraints` | 约束抽取器 | `[]` |
| 8 | `soft_constraints` | 约束抽取器 | `[]` |
| 9 | `verified_evidence` | D26 证据收集器 | `[]` |
| 10 | `provisional_evidence` | D26 证据收集器 | `[]` |

`sub_intents` 作为 `independent_intents` 的**已弃用别名**保留（通过 `model_validator(mode="after")` 同步），以兼容旧客户端。

- **配置**：`ProtocolConfig(enable_structured_output=True, deprecate_sub_intents=False)`
- **实现**：`src/user_input_normalization/server.py` 中的 `AgentIntentResponse` / `AgentIntentNormalizationDetail` / `RecognizeResponse`
