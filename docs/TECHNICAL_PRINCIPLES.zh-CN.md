# Agent 听不懂人话？我把这个问题拆成了两个模块来解决

> [English Technical Principles](https://github.com/xiaoyesoso/XIntent/blob/main/docs/TECHNICAL_PRINCIPLES.md)
>
> 相关文档，[HTTP API 接口文档](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.zh-CN.md) | [项目说明](https://github.com/xiaoyesoso/XIntent/blob/main/README.zh-CN.md)

你跟 Agent 说「那个第二个再便宜点的帮我看看」，Agent 直接懵了。

哪个？第二个什么？再便宜点是多便宜？

这种对话每天都在发生。用户觉得自己的表达很清楚，Agent 却像在听天书。不是 Agent 蠢，是自然语言天生就充满了指代、省略、口语和主观判断，而大模型直接处理这些原始输入，就像让一个刚到中国的外国人去理解「那个」「这玩意儿」「性价比高的」一样，每个字都认识，连起来完全不知道在说什么。

我花了很长时间研究这个问题，最后发现，**这其实不是一个问题，是两个**。

第一个问题，Agent 听不懂用户在说什么。用户说「第二个」，Agent 不知道第二个是啥。用户说「再便宜点」，Agent 不知道便宜多少。这是输入规范化的问题，得在意图识别之前先把用户的输入收拾干净。

第二个问题，Agent 不知道用户想干什么。哪怕输入已经清晰了，「帮我推荐一款手机」到底是推荐意图还是搜索意图？要不要调用工具？需要哪些参数？这是意图识别的问题，而且单次大模型调用既不够快也不够稳。

XIntent 框架就是围绕这两个问题设计的。两个模块，各管一段。输入规范化在意图识别之前运行，把口语化的、模糊的、依赖上下文的用户输入转化为结构化的、机器能消费的文本。三层瀑布式意图识别再拿着这个干净的输入，从最便宜的代码层开始，逐层升级到轻量 LLM，最后才是深度 LLM，在成本、速度和准确率之间找到那个平衡点。

项目开源在 GitHub，地址是 https://github.com/xiaoyesoso/XIntent ，所有代码、测试、文档都可以直接去看。

后来在生产环境中又踩了很多坑，加了一批可选的准确率提升机制，每个针对一类特定失败模式。再后来拿这个系统去面试，面试官的问题反过来又帮我们发现了很多设计盲点，于是又有了六个来自真实面试反馈的增强。

这篇文章就是把这些设计过程从头到尾聊一遍。不是参考手册，是一个把系统从零到一搭起来的人，坐在你对面，跟你聊他为什么这么设计，中间踩了什么坑，哪些地方让他自己都觉得有点得意。

每个原理都标了对应的服务接口和代码实现，你可以边读边去代码里翻。

---

## 为什么 Agent 总是听不懂你在说什么

### 一个让人崩溃的现实

你想想看，我们平时怎么说话的。

病句，倒装，代词，缩写，甚至各种行业黑话，张口就来。这不是谁的表达能力有问题，自然语言天生就是冗余的、模糊的、高度依赖上下文的。

传统软件有 UI 兜着，表单、下拉框、校验规则，把用户的输入框得死死的。但 Agent 直接接收自然语言文本，这层保护没了。后果就是一连串的灾难。

意图识别失准，主语一缺意图就飘了，比如「市场占有率多少？」谁的市场占有率？

上下文断裂，跨轮指代消解不了，第 3 轮说的「第二个」到底指什么？

参数漂移，形容词不量化，工具调用参数完全不可控，「性价比」直接被理解成「便宜」。

还有最离谱的，大模型幻觉，编造不存在的信息，一本正经地胡说八道。

### 这事儿其实跟需求分析是一回事

我有时候觉得，**用户输入规范化就是 Agent 时代的「需求分析」**。

做过传统软件工程的朋友都知道，需求分析是后面所有设计、开发、测试的地基。**地基歪了，楼怎么盖都是歪的**。Agent 也一样，输入不规范，后面的意图识别、工具调用、最终输出，全白搭。

### 对应实现

来，看看这玩意儿怎么跑起来的。整个框架通过 `POST /normalize` 接口对外提供服务，用户传入原始文本，服务返回结构化的规范化结果。

```bash
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "第二个适合生产吗？", "session_id": "s1", "user_id": "u1", "turn": 2}'
```

- **服务入口**，`POST /normalize`（详见 [API 文档](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.zh-CN.md#3-post-normalize--用户输入规范化)）
- **编排器**，`NormalizationPipeline`（`src/user_input_normalization/pipeline.py`）
- **服务启动**，`python -m user_input_normalization.server`（`src/user_input_normalization/server.py`）

---

## 用户输入的六种「病」，每种都有语言学根子

聊着聊着自然想到了一个问题，用户的输入到底乱在哪？

我们花了很多时间分析真实对话数据，最后发现所有的「乱」都可以归到六大类。有意思的是，每一类背后都有对应的语言学或认知科学理论撑着。这不是我们拍脑袋分的，是有根据的。

### 指代问题（Anaphora）

语言学上这叫「指代消解」，语用学的经典课题。代词本身没有固定含义，它的指代对象全靠上下文。

序号指代，「第二个适合生产吗？」时间指代，「刚才那个项目怎么包装？」属性指代，「看樱花的那个地方」。

难点在哪？指代消解需要召回历史对话细节。跨多轮、跨很长时间的话，短期记忆早就失效了。

### 缺失问题（Ellipsis）

省略句是自然语言里非常常见的经济性表达，说话者会省掉双方都能推断的部分。

缺主语，「市场占有率多少？」缺宾语，「帮我优化一下。」缺约束，「推荐一条裤子。」

人是听得懂的，机器不行。

### 表达问题（Expression）

口语和书面语是两码事。口语是线性的、跳跃的、可以随时改口的。书面语是结构化的、完整的。

语序混乱，「这个不太行，换个更像面试能讲的。」临时改口，「不是，我说的是另一个。」

你想想看，自己说话的时候是不是也这样？想到哪说到哪，中间还会自我纠正。但大模型处理这种输入就很痛苦。

### 词义问题（Semantic）

语义学里的「一词多义」和「行业方言」。同一个词，换个领域意思完全不一样。

缩写，RAG、CRM。黑话，抓手、赋能、闭环、不够 P8。同义词，知识库 / 检索增强 / RAG，三个词说的是一回事。

### 主观判断问题（Subjective）

主观性和评价理论，判断词的含义完全依赖个人偏好、场景和标准。

「哪个最有性价比？」「再高级一点。」

这里有意思了，主观判断词必须转化为可量化参数，否则工具调用就不可控。很多 Agent 处理得太粗暴，「性价比高」直接等于「价格低」，质量、性能、品牌、售后全忽略了。这能不出事吗？

### 外部事实问题（External Fact）

指代外部世界状态的表达，真值依赖实时数据。

「最近哪个框架更火？」「现在最便宜的是哪个？」

这种没法在预处理阶段搞定，必须依赖工具调用返回实时数据。不能伪造，伪造就是幻觉。

### 对应实现

分类结果通过 API 响应的 `classification_tags` 字段返回，是个数组，支持多标签。

```json
{
  "classification_tags": ["指代问题", "主观判断问题"]
}
```

- **API 响应字段**，`classification_tags`（详见 [API 文档 - classification_tags 可选值](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.zh-CN.md#classification_tags-可选值)）
- **分类器**，`InputClassifier`（`src/user_input_normalization/classification/classifier.py`）
- **分类规则**，`CLASSIFICATION_RULES`（`src/user_input_normalization/classification/rules.py`，含 6 大类共 121 条分类规则，其中词义类 82 条由 80 个已知术语自动生成 + 2 条补充模式）
- **路由逻辑**，`SUBJECTIVE`/`EXTERNAL_FACT` 进 deep 阶段，其余进 pre 阶段

---

## 两阶段流水线，一个看似简单但很关键的设计决策

### 一个绕不开的矛盾

做用户输入规范化的时候，我们撞上了一个根本矛盾。

有些问题，比如「第二个」这种指代，在意图识别之前就能解决。但有些问题，比如「现在最便宜的」这种外部事实，必须等到工具调用返回数据之后才能处理。

那怎么办？全部在预处理阶段做？跨时间指代和外部事实没法处理。全部推迟到 ReAct 循环？简单指代也要等工具调用，白白增加时延。

这两头都不对。

### 我们的解法，两阶段分工

```
用户输入 -> [pre-normalization] -> 意图识别 -> [deep-normalization] -> 输出
              ↑                              ↑
         能立即解决的                    需要上下文/工具的
```

| 阶段 | 时机 | 处理内容 | 不处理内容 |
|------|------|----------|------------|
| pre | 意图识别之前 | 指代消解、省略补全、病句修正、术语标准化 | 依赖工具返回的量化、外部事实 |
| deep | ReAct 循环中 | 判断词量化、外部事实消解、Observation 回溯 | 已在 pre 完成的 |

说真的，这个设计看起来不起眼，但它解决了时延和完整性之间的张力。**能快的先快，不能快的等着，各干各的活**。

### 对应实现

通过 API 请求的 `observation` 参数控制是否进入 deep 阶段，响应的 `stage_reached` 字段标识实际跑到了哪个阶段。

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

- **API 请求字段**，`observation`（可选，传入工具返回数据触发 deep 阶段）
- **API 响应字段**，`stage_reached`（`"pre"` 或 `"deep"`）
- **编排器**，`NormalizationPipeline.process()`（`src/user_input_normalization/pipeline.py`）
- **pre 阶段**，`PreNormalizer`（`src/user_input_normalization/pre_normalization/normalizer.py`）
- **deep 阶段**，`DeepNormalizer`（`src/user_input_normalization/deep_normalization/normalizer.py`）

---

## 大模型为什么会「越权」，以及怎么管住它

### 一个很头疼的问题

大模型被训练成「helpful assistant」，它天生就想着尽可能完成用户的请求。你要是不管它，它会一股脑把输入规范、意图识别、回答问题全干完。

用户问「市场占有率多少？」，它直接编一个数字回答你。

用户问「帮我推荐裤子」，它直接给你推荐产品。

用户问「现在最便宜的」，它伪造一个实时价格。

这在规范化阶段是灾难性的。**规范化的职责是理解输入，不是回答问题**。你想想看，如果需求分析师还没把需求整理清楚，开发就开始写代码了，那能对吗？

### 我们的解法，显式的「能做 / 不能做」清单

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

我自己的感受是，显式清单是最简单、最有效的约束方式。模糊的角色描述，比如「你是一个助手」，根本管不住大模型。清单把隐式期望变成了显式规则，好审计，好回归测试。

### 对应实现

职责边界通过 System Prompt 硬约束注入，规范化结果中**不会出现**对用户问题的直接回答。

- **Prompt 常量**，`SYSTEM_PROMPT`、`CAN_DO`、`CANNOT_DO`（`src/user_input_normalization/pre_normalization/prompts.py`）
- **边界校验**，`PreNormalizer.validate_boundary()` 检查是否越权（`normalizer.py`）
- **回归测试**，`tests/test_pre_normalization.py::TestResponsibilityBoundary` 验证不回答/不执行工具/不伪造事实

---

## 指代消解与跨轮记忆，让 Agent 真正记住你说过什么

### 消解结果不是一次性的，得跨轮复用

指代消解不是做完就完了，消解结果需要**跨轮复用**。

```
第 3 轮: "第一个方案适合生产吗？" -> 消解: 第一个方案 -> TCC方案
第 5 轮: "那个 TCC 方案成本多少？" -> 按名索引直接命中
第 7 轮: "第一个呢？" -> 复用第 3 轮消解结果
```

你想想看，如果每轮都重新消解一遍，成本高不说，还可能不一致。第 3 轮说「第一个」是 TCC 方案，第 7 轮再消解一次说不定就飘了。用户体验也差。

### 对应实现

消解结果通过 API 响应的 `pronoun_resolutions` 字段返回，同一个 `session_id` 下自动跨轮复用。

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

- **API 响应字段**，`pronoun_resolutions`（详见 [API 文档](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.zh-CN.md#pronoun_resolutions-元素结构)）
- **数据模型**，`PronounResolution`（`src/user_input_normalization/models.py`）
- **跨轮存储**，`KeyFactStore.find_pronoun_resolution()`（`src/user_input_normalization/storage/base.py`）
- **写入逻辑**，`PreNormalizer._write_pronoun_facts()`（`normalizer.py`）
- **跨轮复用测试**，`tests/test_pre_normalization.py::TestCrossTurnReuse`

---

## 按名索引，为什么「TCC 方案」比「方案二」好

### 序号引用是个坑

如果 Agent 输出「方案一、方案二、方案三」，用户后面用「第二个」来引用，大模型在推理阶段**容易引用错误**。

为什么？序号没有语义，大模型的 attention 容易对错。第几个第几个，它自己也搞不清。

### 语义化命名的优势

如果 Agent 输出「TCC 方案、轻量方案、企业方案」，用户后面用「TCC 方案」引用，按名索引匹配准确率高，认知负荷低，规避推理错误。

我是真的觉得，这事儿孔老夫子两千多年前就说过了，「名不正则言不顺，言不顺则事不成」。**序号是弱标识，语义化命名是强标识**。古人诚不欺我。

### 对应实现

- **API 响应字段**，`pronoun_resolutions[].named_entity`（语义化名称）
- **命名逻辑**，`PreNormalizer._ensure_named_entities()`（`normalizer.py`）
- **按名索引测试**，`tests/test_pre_normalization.py::TestPronounResolution::test_named_entity_set`

---

## 澄清机制，宁可问一句也不要瞎猜

### 为什么不能「瞎猜」

误差累积是真实存在的风险。

第 1 轮猜错了，第 2 轮基于错误继续推断，到最后答案完全偏离。这就像你导航起点设错了，越开越远，还觉得自己挺对。

### 置信度阈值机制

```
置信度 >= 0.6 -> 直接消解
置信度 < 0.6  -> 触发澄清，询问用户
```

我始终坚信「**宁可不答，不可错答**」这个原则。阈值 θ_clarify 可配置，默认 0.6，在效率和准确率之间找平衡。

### 对应实现

当 `paused_for_clarification` 为 `true` 时，响应中包含 `clarification` 对象，客户端应该暂停处理，向用户展示澄清问题。

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

- **API 响应字段**，`paused_for_clarification`、`clarification`（详见 [API 文档](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.zh-CN.md#clarification-结构)）
- **处理器**，`ClarificationHandler`（`src/user_input_normalization/clarification/handler.py`）
- **配置项**，`theta_clarify=0.6`、`max_consecutive_clarifications=3`（`src/user_input_normalization/config.py`）
- **恢复接口**，`NormalizationPipeline.resume_after_clarification()`（`pipeline.py`）

---

## 属性检索指代消解，给 Agent 装上 RAG 长期记忆

### 一个特别真实的场景

用户假期前让 Agent 推荐出游地，Agent 推荐了鸡鸣寺看樱花。假期结束了，用户输入，「你上次推荐的看樱花的那个地方……」

难点在哪？时间跨度太长，短期记忆早失效了。而且这是属性指代，「看樱花的」是属性修饰，不是简单的序号指代。

### 两步推理 + 补偿机制

1. **属性提取 + 向量检索**，提取 ["樱花", "旅游"]，检索历史对话库召回细节
2. **大模型推断**，基于窗口内容推断出「鸡鸣寺」
3. **补偿机制**（检索失败时），提取属性发起工具调用，再次推理

我当时真的被这个场景震到了，因为它完美展示了短期记忆不够用的时候该怎么办。

### 对应实现

属性检索消解在流水线中自动触发，检测到属性指代模式时，消解结果回填到 `pronoun_resolutions`。

- **解析器**，`AttributeResolver`（`src/user_input_normalization/attribute_resolution/resolver.py`）
- **两步推理**，`AttributeResolver.resolve()` -> `extract_attributes()` -> `recall_details()` -> `infer()`
- **补偿机制**，`AttributeResolver.compensation()`（模拟工具调用 + 重新推理）
- **触发条件**，`NormalizationPipeline._is_attribute_anaphora()` 检测属性指代模式
- **测试**，`tests/test_attribute_resolution.py::TestJimingTempleCase`（鸡鸣寺案例端到端验证）

---

## 词汇表自我演进，让知识从对话中「长」出来

### 问题

同样的词汇或缩写，在不同行业、领域、上下文里意思完全不一样。人工维护词汇表？不现实，新词每天都在冒出来。

### 自我迭代机制

```
线上对话 -> 离线分析 -> 标记候选词汇 -> 阈值判定 -> 升级入词汇表
```

**阈值规则**，总次数 > 100 / 讨论人数 > 3（公共词汇）/ 连续讨论 > 10 次。

我觉得这个设计最妙的地方在于，**它是「涌现式知识」**。词汇表自下而上从实际对话中长出来，阈值过滤确保稳定性，人工审核兜底。不是我们坐在办公室里拍脑袋想出来的词表，是用户自己教会的。

### 对应实现

术语标准化结果通过 API 响应的 `term_mappings` 字段返回。

```json
{
  "term_mappings": [
    {"original": "RAG", "standard": "检索增强生成", "source": "vocabulary-table"}
  ]
}
```

- **API 响应字段**，`term_mappings`（详见 [API 文档](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.zh-CN.md#term_mappings-元素结构)）
- **词汇表服务**，`VocabularyTable`（`src/user_input_normalization/vocabulary/table.py`）
- **离线分析器**，`OfflineAnalyzer`（`src/user_input_normalization/vocabulary/offline_analyzer.py`）
- **存储接口**，`VocabStore`（`src/user_input_normalization/storage/base.py`）
- **配置项**，`min_total_count=100`、`min_discussant_count=3`、`min_consecutive_count=10`（`config.py`）

---

## 形容词可量化，把「性价比」变成机器能懂的数字

### 问题

任何没量化的形容词都会导致 Agent 输出漂移。

很多 Agent 处理得太粗暴了，「性价比高」直接等于「价格低」，质量、性能、品牌、售后全忽略。这就像你去买车，说「我要性价比高的」，销售直接给你指了辆最便宜的，你什么感受？

### 量化策略

「性价比」其实有两种合理解释。

| 策略 | 含义 | 工具参数 |
|------|------|----------|
| 同等价格更好质量 | 同价位下，质量/配置/服务更好 | `price_range: [150, 200], quality_rank: top 30%` |
| 同等质量更低价格 | 同质量下，价格更便宜 | `price_range: [100, 150], quality_rank: top 50%` |

### 对应实现

量化结果通过 API 响应的 `quantifiable_adjectives` 字段返回，需要在请求中传入 `observation`（比如当前价格）才能完成量化。

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

- **API 请求字段**，`observation`（传入当前价格等上下文，触发 deep 阶段量化）
- **API 响应字段**，`quantifiable_adjectives`（详见 [API 文档](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.zh-CN.md#quantifiable_adjectives-元素结构)）
- **量化引擎**，`QuantificationEngine`（`src/user_input_normalization/quantification/engine.py`）
- **内置规则**，`DEFAULT_RULES`（6 条规则，性价比、划算、再高级一点等）（`quantification/rules.py`）
- **深度规范化**，`DeepNormalizer.quantify_adjectives()`（`deep_normalization/normalizer.py`）

---

## 三层上下文整合，所有规范化措施的一句话总结

### 核心思想

> **所有的规范化措施都可以归结为一句话，深度整合上下文**。

| 层级 | 内容 | 来源 | 用途 |
|------|------|------|------|
| 用户画像 | 行业、偏好、历史行为 | 用户画像存储 | 消歧倾向 |
| 关键事实 | 喜好、观点、态度、已认可点 | 关键事实存储 | 跨轮复用、补全依据 |
| 对话历史 | 摘要 + 细节 | 短期记忆 + RAG 长期记忆 | 指代消解证据、省略补全来源 |

### 对应实现

三层上下文在 `PreNormalizer.normalize()` 内部自动组装，通过 `user_id` 和 `session_id` 关联。

- **整合器**，`ContextIntegrator.assemble()`（`src/user_input_normalization/context/integrator.py`）
- **上下文模型**，`ContextBundle`（user_profile + key_facts + dialogue_summary + recalled_details）
- **优先级冲突**，`ContextIntegrator.resolve_conflict()`（关键事实 > 用户画像 > 对话历史）
- **存储接口**，`UserProfileStore`、`KeyFactStore`、`DialogueHistoryStore`（`storage/base.py`）
- **可观测性**，`ContextIntegrator.explain_influence()` 返回各层影响报告

---

## 完整性校验，给规范化装一个守门员

### 为什么需要校验

规范化不是「调一次大模型就完了」。大模型可能遗漏，主谓宾不完整、代词没完全消解、形容词没量化，这些它都可能漏掉。

### 三项校验 + 路由

```
校验通过 -> 流入意图识别
缺主谓宾 / 代词未消解 -> 触发澄清机制（paused_for_clarification: true）
形容词未量化 -> 路由至 deep-normalization（stage_reached: "deep"）
```

### 对应实现

校验结果影响 API 响应的 `stage_reached` 和 `paused_for_clarification` 字段。

- **校验器**，`CompletenessChecker`（`src/user_input_normalization/pre_normalization/completeness_checker.py`）
- **校验方法**，`check_and_route()` 返回 `(校验结果, 路由目标, 路由原因)`
- **校验项**，`spo_complete`（主谓宾）、`pronouns_resolved`（代词）、`adjectives_quantified`（形容词）
- **API 影响**，未量化形容词走 `stage_reached: "deep"`，代词未消解走 `paused_for_clarification: true`

---

## few-shot 检索注入，让系统越用越聪明

### 问题

few-shot 例子能显著提升规范化质量，但全量注入会撑爆上下文窗口。

### 检索注入 + 线上沉淀

```
用户输入 -> 向量化 -> 检索 top-k 相似例子 -> 注入 Prompt
线上遇到古怪输入 -> 成功消解 -> 整理为 few-shot 例子入库 -> 下次可参考
```

**这就是「数据飞轮」**。线上运行越久，例子库越丰富，规范化质量越高。系统会自己变聪明，不需要你手动喂。

### 对应实现

few-shot 注入在 `PreNormalizer.normalize()` 内部自动执行，对 API 调用方完全透明。

- **例子存储**，`FewShotStore`（`src/user_input_normalization/storage/base.py`）
- **检索注入**，`PreNormalizer._build_user_prompt()` 调用 `FewShotStore.search(top_k=3)`
- **线上沉淀**，`PreNormalizer._sink_to_fewshot()` 将古怪输入自动入库
- **格式化**，`format_fewshot()`（`pre_normalization/prompts.py`）
- **配置项**，`top_k=3`、`enabled=True`（`config.py`）

---

## 聊到这，先做个小结

模块一到此结束，用户输入规范化这块的核心思路其实就五句话。

**深度整合上下文**，所有的措施都围绕这一句话。

**两阶段分工**，能立即解决的在 pre，需要工具的在 deep。

**显式约束**，职责边界、完整性校验、澄清阈值，防止大模型越权和瞎猜。

**结构化输出**，消解表格、主谓宾、量化字段，让下游能程序化消费。

**自我演进**，词汇表迭代、few-shot 沉淀，系统越用越聪明。

接下来聊模块二，三层意图识别。这块的思路也五句话。

**瀑布式升级**，从便宜快的开始，按需升级（Code -> Flash -> Pro）。

**多信号置信度**，绝不只信大模型自报的 confidence，融合规则命中、向量相似度、历史准确率。

**槽位填充是识别的一部分**，意图 + 参数 + 约束是一个结构化输出，不是独立阶段。

**不确定性的双出口**，不支持的意图给出原因拒识，不清晰的意图触发有界澄清循环。

**闭环评估**，测试集 + 下一轮隐式信号回填指标，持续改进。

### 原理与接口映射总表

#### 模块一，用户输入规范化

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

#### 模块二，三层意图识别

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

## 为什么「直接调大模型做意图识别」是错的

### 面试场景

很多朋友可能不知道，面试的时候我问候选人「你的 Agent 怎么做意图识别？」，十有八九回答「调个大模型分类」。

**在工程上这是错的。不是不对，是不够**。原因有三个。

**成本与时延**，单次深度推理调用可能耗时 5 到 15 秒。但你想想看，大多数用户输入，「继续」、「下一个」、「退款」，都是无歧义的，根本不需要深度推理。用大炮打蚊子，又慢又贵。

**置信度不是概率**，大模型自报的 `confidence` 是「自评得分」，不是校准过的概率。把它当真值，错误答案会被过度接受。模型说「我 90% 确定是退款意图」，你就真信了 90%？

**没有双出口**，单次 LLM 调用只产生一个答案。但工程上你需要显式处理两种截然不同的失败，「不支持」（意图不在候选列表中）和「不清晰」（低置信度或缺必填槽位）。一个答案搞不定两种失败。

### 对应实现

本框架引入三层瀑布结构，同时解决上述三个问题。

- **服务入口**，`POST /recognize`（详见 [API 文档](https://github.com/xiaoyesoso/XIntent/blob/main/docs/API.zh-CN.md#4-post-recognize--意图识别)）
- **编排器**，`IntentRecognitionPipeline`（`src/intent_recognition/pipeline.py`）
- **核心设计**，瀑布式架构、独立阶段、双出口

---

## 三层瀑布式架构，从便宜到贵的正确升级姿势

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

**L1 代码层**，三种识别方式，页面引导（UI 事件到意图）、关键字/正则匹配、规则引擎（上下文状态 + 规范化输入）。返回第一个命中或「未命中」信号。**零 LLM 调用，< 10ms。**

**L2 轻量 LLM**，Flash/Mini 模型 + 结构化 Prompt（候选意图 + 槽位 + few-shot）。置信度路由，`>=0.85` 接受，`0.6-0.85` 澄清，`<0.6` 升级到 L3。

**L3 深度 LLM**，深度推理模型处理 5 类复杂场景，复杂表达（病句/倒装）、跨轮上下文依赖、意图切换、多意图分解、隐式信息补全。

### 为什么是瀑布，不是并行

你想想看，三层并行会让 API 成本直接变 3 倍，而且 Flash 模型的置信度和 Pro 模型的置信度根本没法直接比较。瀑布只在未命中或低置信度时升级，所以大多数请求止步于 L1 或 L2。

说真的，**80% 的请求在 L1 就搞定了**，根本不需要花 LLM 的钱。

### 优雅降级

如果 L3 失败了（超时、API 报错），流水线会回退到 L2 的结果并打 warning 日志，而不是整个请求崩掉。用户感知不到，但你能从日志里看到。这种优雅降级在生产环境太重要了。

### 对应实现

- **L1 代码层**，`src/intent_recognition/code_layer/`（`classifier.py`、`page_guidance.py`、`keyword_matcher.py`、`rule_engine.py`）
- **L2 轻量 LLM**，`src/intent_recognition/lightweight_llm/`（`classifier.py`、`confidence_router.py`、`multi_signal.py`、`prompts.py`）
- **L3 深度 LLM**，`src/intent_recognition/deep_llm/`（`classifier.py`、`prompts.py`）
- **编排器**，`IntentRecognitionPipeline.recognize()`（`src/intent_recognition/pipeline.py`）
- **配置开关**，`enable_code_layer`、`enable_lightweight_llm`、`enable_deep_llm`（`src/intent_recognition/config.py`）

---

## 置信度路由与多信号融合，别只信大模型自己说的

### LLM 置信度的问题

大模型自报的置信度是「自评得分」，反映的是模型内部的确定性，不是校准过的概率。

两种失败模式特别要命。

**错答时过度自信**，模型对一个错分类的意图报 0.9 置信度，你信了，结果给用户走了错误流程。

**对答时不够自信**，模型对一个正确意图报 0.5 置信度，触发不必要的升级，白白浪费一次 L3 调用。

### 多信号融合

所以我们不只看 LLM 置信度，融合四个信号。

| 信号 | 权重 | 来源 |
|------|------|------|
| LLM 置信度 | 0.5 | 模型输出 |
| 规则命中 | 0.2 | 代码层关键字/正则在同一意图上命中 |
| 向量相似度 | 0.2 | 与正样本例子的 embedding 相似度 |
| 历史准确率 | 0.1 | 评估存储中按意图统计的准确率 |

融合后的置信度按以下规则路由。

- `>= 0.85`，接受
- `>= 0.6 且 < 0.85`，澄清（问用户）
- `< 0.6`，升级到 L3

### 对应实现

- **置信度路由器**，`ConfidenceRouter`（`src/intent_recognition/lightweight_llm/confidence_router.py`）
- **多信号融合**，`MultiSignalFuser`（`src/intent_recognition/lightweight_llm/multi_signal.py`）
- **配置**，`ConfidenceConfig`（`src/intent_recognition/config.py`），`accept_threshold=0.85`、`clarify_threshold=0.6`、各信号权重字段

---

## 槽位填充，硬约束、软约束和跨轮累积

### 槽位填充是识别的一部分

**意图识别不只是「什么意图」，而是「什么意图 + 什么参数 + 什么约束」**。槽位填充被集成进识别流水线，不是独立阶段。

你想想看，用户说「帮我推荐一款 3000 元以内的华为手机」，意图是 product_recommendation，预算 3000、品牌华为，这些都是意图的一部分。拆开来做没道理。

### 硬约束 vs 软约束

| 约束类型 | 示例 | 语义 |
|---------|------|------|
| 硬约束 | "不超过3000"、"必须华为"、"不要二手" | 必须满足；用于过滤结果 |
| 软约束 | "最好轻薄"、"优先蓝色"、"续航越长越好" | 尽量满足；用于排序或加权 |

这里有意思了，硬约束和软约束必须分开。硬约束是过滤条件，不满足的直接排除。软约束是排序权重，满足不了也没关系但不能因此排除。

本框架用正则模式处理常见的中文约束表达，复杂情况回退到 LLM 抽取。约束以规范化表达式（如 `price<=3000`）+ 原始文本返回，便于追溯。

### 跨轮累积

在 ReAct/TAO 循环中，用户会跨轮迭代细化自己的请求。

```
第 1 轮: "帮我推荐一款3000元以内的手机"  -> {category: 手机, budget_max: 3000}
第 2 轮: "预算可以到4000"                -> {category: 手机, budget_max: 4000}  （最新优先）
第 3 轮: "最好是华为"                    -> {category: 手机, budget_max: 4000, brand: 华为}  （累积）
```

**最新优先策略**，当同一槽位在新轮次中出现，新值覆盖旧值。冲突会被记录用于调试。

**约束累积**，约束不覆盖，唯一的表达式追加进列表。`has_constraint_conflict()` 检测相互矛盾的硬约束（如 `price<100` 与 `price>200`）。

### 对应实现

- **槽位抽取器**，`SlotExtractor`（`src/intent_recognition/slot_filling/extractor.py`）
- **约束抽取器**，`ConstraintExtractor`（`src/intent_recognition/slot_filling/constraints.py`）
- **跨轮合并器**，`CrossTurnSlotMerger`（`src/intent_recognition/slot_filling/cross_turn.py`）
- **槽位状态存储**，`SlotStateStore`（`src/intent_recognition/storage/base.py`，内存实现在 `memory.py`）
- **API 字段**，`slots`、`missing_slots`、`hard_constraints`、`soft_constraints`

---

## 拒识与澄清，两种截然不同的失败要分开处理

### 两种失败，别混在一起

| 模式 | 触发条件 | 响应 |
|------|---------|------|
| 不支持（拒识） | 意图不在候选列表 | `rejection_reason` 列出支持的意图 |
| 不清晰（澄清） | 低置信度或缺必填槽位 | `clarification_question` 询问用户 |

一句「我不知道」会把两种情况混在一起。用户分不清是该重新表述（不支持）还是补充信息（不清晰）。体验太差了。

### 拒识，告诉用户什么是支持的

当意图不在候选列表时，拒识原因包含三部分。

1. 不支持的内容（原样回显）
2. 支持的意图列表，含描述和正样本
3. 来自负样本的重定向提示（如 `iPhone 16的电池容量是多少 -> product_query`）

你想想看，这比单纯说「不支持」好太多了。用户不仅知道这个不行，还知道哪些行，甚至知道类似的问题该怎么问。

### 澄清，带收敛追踪的有界循环

澄清不是无限的。框架按 session 追踪连续澄清次数。

达到 `max_consecutive_clarifications`（默认 3）后，系统降级为「最佳猜测」模式，接受置信度最高的候选，而不是再次询问。

如果用户的下一轮解除了澄清（比如补上了缺失的槽位），连续计数器重置。

这个设计很重要，**不能让用户陷入无限澄清的死循环里**。3 次之后就必须给出一个答案，哪怕不是最优的。

### 对应实现

- **处理器**，`RejectionClarificationHandler`（`src/intent_recognition/rejection_clarification/handler.py`）
- **配置**，`ClarificationConfig.max_consecutive_clarifications=3`（`src/intent_recognition/config.py`）
- **API 字段**，`rejection_reason`、`need_clarification`、`clarification_question`

---


## 下一轮隐式评估，不问用户也能知道对不对

### 显式反馈的问题

每次意图识别后问用户「我理解对了吗？」是糟糕的 UX。大多数用户不会答，答的人又有偏差，倾向于报错而不报对。你想想看，用着用着突然弹个框问你「我理解对了吗？」，你是不是想关掉这个应用？

### 隐式信号检测

所以我们不问用户，而是在**下一轮的用户输入**中检测失败信号。

```
第 N 轮: "帮我推荐手机"  -> intent: product_recommendation
第 N+1 轮: "你理解错了，我要查订单"  -> 检测到第 N 轮的失败信号
```

默认失败信号，`你理解错了`、`不是这个意思`、`不是，我不是`、`不对，我要的是`。

检测到失败信号时做两件事。

1. 上一轮的意图在评估存储中被标记为错误
2. 当前轮的意图重新识别（失败信号中常包含正确的意图）

这个设计太爽了，**你不需要打扰用户，就能自动发现上一轮的错误并纠正**。

### 收敛追踪

对澄清循环，框架追踪澄清后用户的任务是否进展。

- **已解除**，下一轮输入是正常请求（非失败信号，非又一次澄清），重置连续计数器
- **未解除**，下一轮输入是又一次澄清响应，递增连续计数器

### 对应实现

- **失败信号检测**，`RejectionClarificationHandler.detect_implicit_failure()`（`src/intent_recognition/rejection_clarification/handler.py`）
- **配置**，`IntentRecognitionConfig.failure_signals`（`src/intent_recognition/config.py`）
- **评估存储**，`EvaluationStore`（`src/intent_recognition/storage/base.py`）

---

## 意图边界定义与 few-shot 注入，给意图画清楚地盘

### 边界问题

意图之间有重叠。「iPhone 16的电池容量是多少」既可能是 `product_query`（查参数）也可能是 `product_recommendation`（按电池推荐）。没有显式边界，大模型会猜得不一致。今天猜这个，明天猜那个，用户懵了。

### 正样本 + 负样本

每个 `IntentDefinition` 包含两部分。

- **正样本**，明确属于该意图的输入
- **负样本**，看似相似但属于别的意图的输入，带重定向目标（如 `3000以内有什么手机推荐 -> product_recommendation`）

这些被注入 L2 Prompt 用于界定边界。就像教小孩认动物，不仅要给他看「这是猫」，还要给他看「这不是猫，这是狗」，边界才清楚。

### few-shot 检索注入

对易混淆的意图对，框架检索 top-k 个最相似的历史例子（来自 `FewShotStore`）注入 Prompt。这与模块一 few-shot 注入是同一机制，跨模块复用。

### 对应实现

- **意图注册表**，`IntentRegistry`（`src/intent_recognition/intent_registry.py`），存储带正/负样本的 `IntentDefinition`
- **Prompt 构造器**，`build_prompt()`（`lightweight_llm/prompts.py`），注入候选意图、槽位、边界、few-shot
- **few-shot 存储**，`FewShotStore`（与模块一共享，`src/user_input_normalization/storage/base.py`）
- **配置**，`fewshot_top_k=3`、`fewshot_enabled=True`（`src/intent_recognition/config.py`）

---

## 分层意图与决策树，100 个意图怎么塞进 Prompt

### 上下文窗口问题

当意图空间增长到 20+ 个时，把它们全列在 LLM Prompt 中会撑爆上下文窗口并降低准确率。100 个意图全塞进去，大模型看着都头晕。

### 分层调度

本框架支持父/子意图关系。

```python
IntentDefinition(name="refund", parent_intent="order_query", ...)
```

启用分层调度时，分三步走。

1. L1/L2 先识别父意图（如 `order_query`）
2. 候选列表收窄到 `order_query` 的子意图（如 `refund`、`exchange`、`cancel`）
3. L1/L2 在收窄后的列表上再跑一次

这把扁平的 100 意图问题变成两个 10 意图问题。是不是一下子清爽了？

### 何时启用

分层调度**默认关闭**（`enable_hierarchical=False`），因为它会增加时延（两次识别）。只有当意图空间大到影响 L2 准确率时才启用。小意图空间用不着，别过度设计。

### 对应实现

- **意图注册表**，`IntentRegistry`（`src/intent_recognition/intent_registry.py`），`parent_intent` 字段、`get_children()` 方法
- **配置开关**，`enable_hierarchical`（`src/intent_recognition/config.py`）
- **默认意图**，服务端 `_build_default_registry()`（`server.py`）将 `refund` 注册为 `order_query` 的子意图

---

## 评估指标，4 层指标体系与 9 场景测试集

### 4 层指标

| 层级 | 指标 | 衡量内容 |
|------|------|---------|
| 1. Top-K 准确率 | Top-1、Top-3 | 正确意图是否在 top-K 中 |
| 2. 按意图 | 每意图准确率 | 哪些意图容易被混为哪些 |
| 3. 拒识/澄清 | 拒识准确率、误拒率、漏拒率、澄清触发准确率、澄清后收敛率 | 双出口是否工作正常 |
| 4. 槽位填充 | 槽位准确率、槽位召回率、必填槽位完整率、槽位更新准确率、约束识别准确率 | 参数抽取是否正确 |

### 基准目标

- **领域专用 Agent**，0.99（本框架的目标市场）
- **通用 Agent**，0.85（基线）

说真的，0.99 这个目标不是随便定的。领域专用 Agent 的意图空间有限、用户群体固定，**99% 是可以做到的**。通用 Agent 面对无限的意图空间，85% 已经是很高的基线了。

### 9 场景测试集

本框架内置测试集，覆盖 9 类场景。

1. 清晰/简单意图（L1/L2 命中）
2. 复杂表达（需 L3）
3. 多意图输入（分解）
4. 跨轮上下文依赖
5. 意图切换
6. 槽位填充（缺必填槽位）
7. 拒识（不支持的意图）
8. 澄清（不清晰的意图）
9. 分层调度

这 9 个场景不是拍脑袋想的，是从真实生产环境中归纳出来的。每个场景都对应一类失败模式，覆盖了系统能遇到的大部分情况。

### 对应实现

- **指标计算器**，`MetricsCalculator`（`src/intent_recognition/evaluation/metrics.py`）
- **测试集**，`TestSet`（`src/intent_recognition/evaluation/test_set.py`），9 种场景类型、样例用例
- **测试运行器**，`TestRunner`（`src/intent_recognition/evaluation/runner.py`），对流水线跑测试集，生成报告
- **配置**，`EvaluationConfig`（`src/intent_recognition/config.py`），`top_k=3`、基准目标

---

## 准确率提升，从 85% 到 99% 的八把刀

回到意图识别这块，三层瀑布已经能解决大部分问题。但生产环境中你会发现，总有那么一些边界 case，瀑布搞不定。

我们花了大量时间分析生产环境的失败案例，最后归纳出八类准确率问题。这八个扩展机制就是针对这八类问题的八把刀，**每把刀切一类特定失败模式**。

大部分默认关闭，因为它们都有成本（多意图分解除外，默认开启）。你需要哪把开哪把，或者用 `accuracy_preset` 一键开一组。

### 问题与方案对照

| 问题 | 方案 | 配置 |
|---------|----------|--------|
| 意图空间过大 -> L2 混淆 | 检索式候选收窄 | `retrieval.enable` |
| 静态 few-shot 不足 | 动态 few-shot 注入 | `dynamic_fewshot.dynamic_enabled` |
| 意图边界重叠 | 意图正交性检查 | `orthogonality.enable_check` |
| 伪多意图（流程描述） | 真多意图分解 | `multi_intent.enable` |
| 代码层未命中（无关键字匹配） | 向量匹配兜底 | `vector_fallback.enable` |
| 同一任务不必要的重复识别 | 意图复用与回滚 | `reuse_strategy.enable` |
| 单识别器不确定 | 多识别器仲裁 | `arbiter.enable` |
| 需要领域专用模型 | 微调集成点 | `fine_tuning.enable` |

### 检索式候选收窄

你想想看，意图空间一大，L2 的 Prompt 里候选意图列表就长，大模型看着一堆选项容易犯迷糊。检索式候选收窄的思路是，在 LLM 分类之前先用检索把候选收窄到 Top-N。

用向量相似度、LLM 粗分类或混合检索都行。N 值动态调整，意图空间大就多给几个候选，小就少给。这样 L2 看到的选项少了，准确率自然就上去了。

- **配置**，`RetrievalConfig(enable=False, method="vector"|"llm_coarse"|"hybrid", top_n=10, dynamic_n=True)`
- **按请求覆盖**，在 `POST /agent/intent` 和 `POST /recognize` 上传 `enable_retrieval: true|false`
- **实现**，`src/intent_recognition/lightweight_llm/candidate_retriever.py`

### 动态 few-shot 注入

静态 few-shot 是固定的，对所有输入都注入同一批例子。但有些输入比较特殊，静态例子不够用。动态 few-shot 注入的思路是，除了静态 few-shot，再动态检索一批和当前输入最相似的历史例子注入进去。

用 Jaccard token 重叠度从 `FewShotStore` 中查找。这个和检索式候选收窄的思路类似，但检索的是 few-shot 例子而不是候选意图。

- **配置**，`DynamicFewShotConfig(dynamic_enabled=False, dynamic_top_k=3, static_kind_tag="static")`
- **实现**，`src/intent_recognition/lightweight_llm/dynamic_fewshot.py`

### 意图正交性检查

意图定义之间会有重叠，两个意图的正样本太像了，大模型分不清。意图正交性检查检测意图定义之间的重叠（相似度 > 阈值），并提供 `split_intent()` 和 `merge_with_param()` 操作让你维护一个干净、正交的意图空间。

这个更多是治理工具，帮你发现「这俩意图是不是该合并」或者「这个意图是不是该拆分」。

- **配置**，`OrthogonalityConfig(enable_check=False, overlap_threshold=0.7)`
- **实现**，`IntentRegistry.detect_overlap()`、`split_intent()`、`merge_with_param()`（`src/intent_recognition/intent_registry.py`）

### 真多意图分解

用户有时候一句话里包含多个意图，比如「推荐手机然后结账」。但这里有个坑，有些看起来像多意图的其实是流程描述，不是真正的独立意图。

真多意图分解区分真多意图请求与流程描述，返回子意图之间的依赖关系以及拓扑排序后的待执行列表。这样你就知道先做什么后做什么，而不是一股脑全扔给用户。

- **配置**，`MultiIntentConfig(enable=True, sequential_execution=True, filter_process_description=True)`
- **响应字段**，`relations: [{src, dst, constraints}]`、`pending_intents: ["intent_a", "intent_b"]`
- **实现**，`src/intent_recognition/deep_llm/classifier.py`（解析 + 拓扑排序）

### 向量匹配兜底

L1 代码层靠关键字、正则、规则匹配，但总有没命中的时候。用户输入不带任何关键字，L1 就抓瞎了。

向量匹配兜底的解法是，L1 没命中时回退到向量相似度匹配。用 128 维哈希向量化器（纯 Python 实现，无 numpy 依赖）和余弦相似度，在预建的向量映射里找最相似的意图。

这个设计面试的时候特别能聊，因为它展示了「**不依赖 LLM 也能做语义匹配**」的思路。

- **配置**，`VectorFallbackConfig(enable=False, similarity_threshold=0.92, top_k=1)`
- **按请求覆盖**，`enable_vector_fallback: true|false`
- **实现**，`src/intent_recognition/code_layer/vector_matcher.py`
- **source 取值**，`"vector-fallback"`

### 意图复用与回滚

用户跨轮继续同一任务时，比如第 1 轮说「推荐手机」，第 2 轮说「预算可以到 4000」，第 2 轮的意图其实和第 1 轮一样。如果每次都跑一遍瀑布，既浪费又可能跑出不一样的结果。

意图复用与回滚直接复用上一轮的意图（confidence=1.0, source="reused"），跳过整个瀑布。但在出现隐式失败信号或意图切换标记时回滚，重新跑瀑布。

意图切换标记包括「换个话题」「不是这个」「我要问别的」「切换」「换一个」「不是说这个」「我想问的是」。

- **配置**，`ReuseStrategyConfig(enable=False, rollback_on_failure_signal=True, rollback_on_tool_failure_count=3)`
- **按请求覆盖**，`reuse_previous_intent: true|false`
- **意图切换标记**，`换个话题`、`不是这个`、`我要问别的`、`切换`、`换一个`、`不是说这个`、`我想问的是`
- **实现**，`src/intent_recognition/intent_reuse_strategy.py`
- **source 取值**，`"reused"`

### 多识别器仲裁

单一识别器不确定的时候怎么办？多识别器仲裁的思路是，并行跑多个识别器（向量、规则、轻量 LLM），然后投票或加权评分做仲裁。

投票就是多数服从，加权评分就是给每个识别器一个权重，算总分。适用于那种单个识别器都拿不准、但综合起来能判断的场景。

- **配置**，`ArbiterConfig(enable=False, mode="vote"|"weighted_score", recognizers=["vector","rule","lightweight_llm"], weights={"vector":0.8,"rule":0.6,"lightweight_llm":1.0})`
- **实现**，`src/intent_recognition/multi_recognizer_arbiter.py`
- **source 取值**，`"arbiter-vote"` 或 `"arbiter-weighted"`

### 微调集成点

当通用模型不够用的时候，你需要领域专用模型。微调集成点提供训练数据导出流水线（JSONL 格式）和 `model_tier="fine_tuned"` LLM 客户端层级，读取 `FINE_TUNED_MODEL` 环境变量。

注意，这里只是设计层面的集成点，不执行实际训练。导出的数据可以拿去外部微调领域专用模型，然后接回来用。

- **配置**，`FineTuningConfig(enable=False, model=None)`
- **实现**，`src/intent_recognition/training_data_exporter.py`、评估运行器中的 `evaluate_fine_tuned()`
- **环境变量**，`FINE_TUNED_MODEL` 用于微调模型名

### 按请求覆盖模式

检索式候选收窄、向量匹配兜底、意图复用与回滚这三项能力都支持通过 `POST /agent/intent` 和 `POST /recognize` 请求体的按请求覆盖。

| 字段 | 类型 | 默认值 | 效果 |
|-------|------|---------|--------|
| `enable_retrieval` | `bool \| null` | `null` | 覆盖检索式候选收窄（`null`=使用配置，`true`/`false`=强制）|
| `enable_vector_fallback` | `bool \| null` | `null` | 覆盖向量匹配兜底 |
| `reuse_previous_intent` | `bool \| null` | `null` | 覆盖意图复用与回滚 |

覆盖会作用于流水线配置的快照，并在请求完成后恢复，因此不会跨共享单例流水线的请求泄漏。这个设计很重要，不然一个请求开了某个开关，后续所有请求都被影响了，那就出事了。

### 准确率预设

`accuracy_preset` 配置字段（默认 `"balanced"`）提供了一次性启用多项能力的快捷方式。

- `"balanced"`（默认），所有扩展关闭，依赖三层瀑布
- `"high_accuracy"`，启用检索式候选收窄 + 向量匹配兜底 + 意图复用与回滚以获得最高准确率（成本更高）
- `"low_cost"`，全部关闭（同 balanced，显式声明）

我是真的觉得这个预设设计很实用。大部分场景用 balanced 就够了，需要极限准确率的时候切 high_accuracy，成本敏感的时候切 low_cost。一个配置搞定，不用一个个开关去拨。

## 面试洞察，六个来自真实面试反馈的设计决策

顺着上面的再聊聊。

准确率提升那八个扩展是我们在生产环境中摸爬滚打总结出来的。面试洞察这六个设计决策不一样，来自面试反馈。

我们拿这个系统去面试，面试官问的问题、课后习题的思路、复用模板里的写法，反过来帮我们发现了很多之前没想到的设计盲点。这六项默认全部开启，因为它们是增量式的，丰富输出而不改变既有行为。

### 证据分级，每个槽位值都要有「出处」

每个槽位值和事实都标注证据等级，这是证据分级的核心。

- **`verified`（已验证）**，来自当前输入、活跃上下文或已确认的 `KeyFact`。硬操作（退款、支付）可以直接执行。
- **`provisional`（暂定）**，来自历史、用户画像、推测或默认值。硬操作必须升级证据（用户确认 / Observation 校验）或披露假设。

你想想看，这就像法庭上的证据规则。直接证据和间接证据的效力不一样，你不能用道听途说来定罪。Agent 也一样，用户当前输入里明确说的和从历史推测的，可信度不一样。

流水线通过 `_collect_evidence()` 收集证据，通过 `_upgrade_provisional_to_verified()` 升级暂定证据，并在 `evidence.require_verified_for_hard_ops=True` 且意图属于 `evidence.high_risk_intents` 时通过 `_check_hard_op_evidence()` 执行硬操作前校验。

- **配置**，`EvidenceConfig(enable_grading=True, require_verified_for_hard_ops=True, high_risk_intents=[])`
- **实现**，`IntentRecognitionPipeline._collect_evidence()` / `_upgrade_provisional_to_verified()` / `_check_hard_op_evidence()`，位于 `src/intent_recognition/pipeline.py`
- **模型**，`Evidence`、`EvidenceGrade`、`SlotValue`，位于 `src/intent_recognition/models.py`

### sub_tasks vs independent_intents 边界，意图的职责是选择业务流程

**意图的职责是选择业务流程。** 只有能独立触发*不同*业务流程的目标才算 `independent_intents`，主流程内的步骤（比价、查配送、查参数）应归入 `sub_tasks`、槽位或约束。

- `independent_intents` 进入多意图治理（过程性过滤、`relations`、`pending_intents`、顺序执行）。
- `sub_tasks` 作为主流程内的执行步骤记录，**不进入** `relations`、`pending_intents` 或过程性过滤。

这个边界特别重要。如果没有这个区分，「推荐手机然后比价然后查配送」会被当成三个独立意图，但其实比价和查配送只是推荐手机这个主流程里的步骤。把它们当成 sub_tasks 记录就好，不需要进入多意图治理。

深度 LLM 提示词（规则 6-10）和轻量 LLM（`detect_boundary_simple()` 配合 `_SUB_TASK_PATTERNS` / `_INDEPENDENT_MARKERS`）共同强制此边界。

- **配置**，`BoundaryConfig(enable_sub_tasks=True, strict_mode=False)`
- **实现**，`src/intent_recognition/deep_llm/prompts.py`（规则 6-10）、`src/intent_recognition/deep_llm/classifier.py`、`src/intent_recognition/lightweight_llm/classifier.py`

### 五因子路由仲裁，置信度是信号不是概率

**置信度是信号，不是概率**。这句话我在面试的时候反复强调。

当 L2 置信度落在模糊区间 `[clarify_threshold, accept_threshold)` 时，按优先级顺序执行五因子仲裁。

| # | 因子 | 失败动作 |
|---|------|---------|
| 1 | 规则校验 | 调整置信度（±0.05） |
| 2 | 槽位完整性 | 澄清 |
| 3 | 硬约束风险（高风险意图 + 暂定必填槽位） | 澄清 |
| 4 | 候选间距（Top1-Top2 < 阈值） | 升级到 L3 |
| 5 | 置信度 + 历史准确率 | 历史差且调整后置信度 < 接受阈值则澄清 |

第一个强制澄清/升级的失败因子胜出。因子 5 复用多信号融合分数，不替代它。

这里有个关键设计点，因子的优先级顺序不能乱。规则校验在前，槽位完整性在后，再是硬约束风险。为什么？因为如果槽位完整性排在风险后面，一个高风险意图缺了必填槽位，系统会先报风险原因而不是「缺必填槽位」。对用户来说，「缺必填槽位」比「高风险」更可操作，因为缺槽位他可以补，风险他不知道该怎么办。

对于 L2/L3 不一致，`arbitrate_l2_l3()` 规则，L3 高置信度 + 槽位完整就接受 L3，两者均低于澄清阈值就强制澄清，否则对 L3 结果执行五因子仲裁。

- **配置**，`ArbitrationConfig(enable_five_factor=True, candidate_gap_threshold=0.1, risk_aware_clarify=True, high_risk_intents=[])`
- **实现**，`src/intent_recognition/lightweight_llm/confidence_router.py`（`ArbitrationInput`、`ArbitrationDecision`、`arbitrate()`、`arbitrate_l2_l3()`）
- **输出**，`IntentRecognitionResult` 的 `arbitration_breakdown` 字段（与 `signals` 分离，因为 `signals` 类型为 `dict[str, float]`）

### 扩展评估指标，8 项细粒度指标 + 线上反馈回流

在四层指标基础上，扩展评估指标新增 8 项细粒度指标和线上反馈回流。

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

**线上反馈回流**，`TestSet.import_online_samples()` 将生产失败样本（通过 `TestRunner.collect_online_failures()` 利用失败信号收集）回灌到离线测试集。

这个闭环太重要了。线上失败样本自动收集，回灌到测试集，下次跑测试就能覆盖这些 case。**系统越用越强，不需要人工去捞 case**。

- **配置**，`ExtendedEvaluationConfig` 逐项开关
- **实现**，`src/intent_recognition/evaluation/metrics.py`、`src/intent_recognition/evaluation/test_set.py`、`src/intent_recognition/evaluation/runner.py`

### 方案演进方法论，从 v0 到 v2 的进化路径

框架分三个阶段演进，每个阶段解决前一阶段的准确率瓶颈。

| 阶段 | 架构 | 关键机制 |
|------|------|---------|
| **v0** | 单深度 LLM，无规范化 | 暴力意图识别 |
| **v1-stage1** | + L2 轻量 LLM | 降成本、置信度路由 |
| **v1-stage2** | + L1 代码层 + 多信号融合 | 三层瀑布、<10ms 快路径 |
| **v1-stage3** | + 高级机制 | 检索、动态 fewshot、向量回退、复用、仲裁、微调 |
| **v2** | + 面试洞察 | 证据分级、边界、仲裁、扩展评估、协议 |

`TestRunner.assess_evolution_stage(config)` 检查配置并返回当前阶段字符串。可用于面试中表达系统成熟度，或用于功能预期门控。

**60 秒面试表达骨架**（填空模板），

> 我们将意图识别建模为 ___ 层瀑布，代码层负责 <10ms 的关键字/正则命中，轻量 LLM 处理模糊中间地带，深度 LLM 应对 ___ 复杂场景。置信度被视为 ___（信号），因此在接受前执行五因子 ___（仲裁）。每个槽位值携带证据 ___（等级），已验证或暂定，硬操作要求 ___（已验证）证据否则必须披露假设。我们用 ___（9）项指标评估，包括混淆矩阵和线上失败反馈回流，系统按 v0 -> v1 -> v2 阶段演进。

- **实现**，`src/intent_recognition/evaluation/runner.py::assess_evolution_stage()`

### 统一输出协议，10 字段交接契约

10 字段协议作为规范化与意图识别之间、以及意图识别与下游 ReAct/TAO 循环之间的交接契约。

| # | 字段 | 来源 | 默认值 |
|---|------|------|--------|
| 1 | `normalized_query` | 规范化输出（跳过时为原始输入） | `/recognize` 上为 `""` |
| 2 | `intent` | 识别结果 | `null` |
| 3 | `sub_tasks` | 流程内步骤 | `[]` |
| 4 | `independent_intents` | 多流程目标 | `[]` |
| 5 | `slots` | 槽位抽取器 | `{}` |
| 6 | `missing_slots` | 槽位抽取器 | `[]` |
| 7 | `hard_constraints` | 约束抽取器 | `[]` |
| 8 | `soft_constraints` | 约束抽取器 | `[]` |
| 9 | `verified_evidence` | 证据收集器 | `[]` |
| 10 | `provisional_evidence` | 证据收集器 | `[]` |

`sub_intents` 作为 `independent_intents` 的**已弃用别名**保留（通过 `model_validator(mode="after")` 同步），以兼容旧客户端。老代码读 `sub_intents` 还能跑，但新代码应该用 `independent_intents`。

- **配置**，`ProtocolConfig(enable_structured_output=True, deprecate_sub_intents=False)`
- **实现**，`src/user_input_normalization/server.py` 中的 `AgentIntentResponse` / `AgentIntentNormalizationDetail` / `RecognizeResponse`

---

回到开头那个场景。

你跟 Agent 说「那个第二个再便宜点的帮我看看」，Agent 懵了。

现在你知道为什么了。不是 Agent 蠢，是这句话里藏着至少四个问题，「那个」是指代、「第二个」是序号引用、「再便宜点」是未量化的形容词、「帮我看看」是省略了具体动作。每一个都能让 Agent 的理解偏一截，四个叠在一起，彻底跑偏。

XIntent 做的事情，就是在用户和大模型之间，加了「两道翻译」。第一道把人话翻译成机器能消费的结构化输入，第二道用瀑布式的分层策略判断用户到底想干什么。大部分请求在代码层就搞定了，不用调大模型，10 毫秒以内返回。真正复杂的才往上走，一层一层升级，直到深度 LLM 兜底。

我有时候觉得，传统软件用表单和下拉框把用户输入框得死死的，那是一种「降低表达能力来换取准确性」的妥协。Agent 时代不一样了，用户终于可以像跟人说话一样跟机器说话，但代价是，系统得自己去理解那些模糊、跳跃、依赖上下文的自然语言。

这不是一个技术问题，是一个时代问题。

UI 消失了，理解诞生了。

这篇文章聊了从输入规范化到三层瀑布、从准确率提升到面试洞察的全部设计过程。如果你也在搞 Agent，正在被意图识别折磨，可以去 GitHub 看看完整代码，地址是 https://github.com/xiaoyesoso/XIntent 。

650 个测试全部通过，MockLLM 模式开箱即用，接上你自己的 API Key 就能跑真实模型。

代码在那，思路也在那，随你拿去用。
