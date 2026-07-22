"""Input problem classification rule definitions (corresponding to task 2.1).

Defines detection rules for six categories of input problems:
- Anaphora (ANAPHORA): this, that, the first one, the one just mentioned
- Missing (MISSING): missing subject/object/constraints
- Expression (EXPRESSION): colloquial, jumpy, broken sentences, word-order issues, mid-sentence changes
- Semantic (SEMANTIC): non-standard terminology, jargon, internal codes, ambiguous words
- Subjective (SUBJECTIVE): best value for money, a bit more advanced
- External fact (EXTERNAL_FACT): requires real-time data/tool returns

Rules use regex patterns + keyword lists; a match assigns a label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import InputProblemType


@dataclass
class ClassificationRule:
    """A single classification rule.

    Attributes:
        pattern: Compiled regex pattern
        description: Rule description (for explainability)
        sub_type: Sub-type labeled on match (e.g. "序号指代", "缺宾语")
    """

    pattern: re.Pattern[str]
    description: str
    sub_type: str | None = None


# ---------------------------------------------------------------------------
# 1. Anaphora (ANAPHORA) rules
# ---------------------------------------------------------------------------

ANAPHORA_RULES: list[ClassificationRule] = [
    ClassificationRule(
        pattern=re.compile(r"第[一二三四五六七八九十百千\d]+(?:个|种|项|条|步)"),
        description="序号指代：第二个/第三种/第 5 项",
        sub_type="序号指代",
    ),
    ClassificationRule(
        pattern=re.compile(r"这(?:个|些|种|样|那种)"),
        description="近指：这个/这些/这种",
        sub_type="近指",
    ),
    ClassificationRule(
        pattern=re.compile(r"那(?:个|些|种|样|种)"),
        description="远指：那个/那些/那种",
        sub_type="远指",
    ),
    ClassificationRule(
        pattern=re.compile(r"刚才(?:那个|这个|说的|提到的)"),
        description="时间指代：刚才那个/刚才说的",
        sub_type="时间指代",
    ),
    ClassificationRule(
        pattern=re.compile(r"前(?:一个|面那个|面说的|面提到的)"),
        description="时间指代：前一个/前面那个",
        sub_type="时间指代",
    ),
    ClassificationRule(
        pattern=re.compile(r"上(?:一个|次那个|次说的|次提到的)"),
        description="时间指代：上一个/上次那个",
        sub_type="时间指代",
    ),
    ClassificationRule(
        pattern=re.compile(r"它(?:们)?(?:是|的|怎么|适合|可以)"),
        description="代词指代：它/它们/它的",
        sub_type="代词指代",
    ),
    ClassificationRule(
        pattern=re.compile(r"他(?:们)?(?:是|的|怎么|适合|可以)"),
        description="代词指代：他/他们",
        sub_type="代词指代",
    ),
    ClassificationRule(
        pattern=re.compile(r"其(?:中|它|他)"),
        description="代词指代：其中/其它",
        sub_type="代词指代",
    ),
]


# ---------------------------------------------------------------------------
# 2. Missing (MISSING) rules
# ---------------------------------------------------------------------------

MISSING_RULES: list[ClassificationRule] = [
    ClassificationRule(
        pattern=re.compile(r"^[^，。、的是在].{0,15}多少[？?]?\s*$"),
        description="缺主语 + 多少：市场占有率多少？",
        sub_type="缺主语",
    ),
    ClassificationRule(
        pattern=re.compile(r"^[^，。、的是在].{0,15}怎么样[？?]?\s*$"),
        description="缺主语 + 怎么样：方案怎么样？",
        sub_type="缺主语",
    ),
    ClassificationRule(
        pattern=re.compile(r"^[^，。、的是在].{0,15}如何[？?]?\s*$"),
        description="缺主语 + 如何：如何优化？",
        sub_type="缺主语",
    ),
    ClassificationRule(
        pattern=re.compile(r"帮我(?:优化|看看|查一下|处理|分析|整理)一下[。.]?\s*$"),
        description="缺宾语：帮我优化一下。",
        sub_type="缺宾语",
    ),
    ClassificationRule(
        pattern=re.compile(r"^(?:帮我|请|麻烦)?(?:看|查|改|调|优化|分析)一下[。.]?\s*$"),
        description="缺宾语：看一下/查一下（无动作对象）",
        sub_type="缺宾语",
    ),
    ClassificationRule(
        # Short yes/no question: allow 0-4 char subject before 可以/不行/适合/能/行 + 吗
        # Covers "适合吗？" (0-char subject) and "它适合吗？" (1-char subject)
        pattern=re.compile(r"^.{0,4}(?:可以|不行|适合|能|行)[吗嘛][？?]?\s*$"),
        description="短句是非问：适合吗？/可以吗？（缺主语或缺约束）",
        sub_type="缺主语",
    ),
    ClassificationRule(
        pattern=re.compile(r"^(?:再|多)?(?:加|补|给)(?:点|一些|几个)[。.]?\s*$"),
        description="缺宾语：再加点/多给一些",
        sub_type="缺宾语",
    ),
    ClassificationRule(
        # Vague request: 帮我看看/查一下 (even with an object, still lacks explicit constraints, e.g. "what aspect")
        # Corresponds to spec scenario: "帮我看看那个 RAG 的情况。" also matches MISSING
        pattern=re.compile(r"帮我(?:看看|看一下|查一下|了解一下|了解下)"),
        description="请求模糊：帮我看看/查一下（缺明确约束）",
        sub_type="缺约束",
    ),
]


# ---------------------------------------------------------------------------
# 3. Expression (EXPRESSION) rules
# ---------------------------------------------------------------------------

EXPRESSION_RULES: list[ClassificationRule] = [
    ClassificationRule(
        pattern=re.compile(r"不是[，,]?\s*(?:我说的是|我意思是|我的意思是)"),
        description="临时改口：不是，我说的是另一个",
        sub_type="临时改口",
    ),
    ClassificationRule(
        pattern=re.compile(r"这(?:个|种)(?:不太行|不行|不太好|没法用|不能用)"),
        description="改口否定：这个不太行",
        sub_type="临时改口",
    ),
    ClassificationRule(
        pattern=re.compile(r"换(?:一个|个|成|种)(?:更像|更接近|更合适)?"),
        description="改口换方案：换一个/换个更像面试能讲的",
        sub_type="临时改口",
    ),
    ClassificationRule(
        pattern=re.compile(r"^\s*(?:对|嗯|啊|哦|哎|那个)[，,。.]"),
        description="口语化开头：对，/嗯。/那个，",
        sub_type="口语化",
    ),
    ClassificationRule(
        pattern=re.compile(r"(?:然后|就是|那个|这个|怎么说呢)[，,]\s*(?:然后|就是|那个|这个)"),
        description="口语化跳跃：然后，然后/就是，就是",
        sub_type="跳跃",
    ),
    ClassificationRule(
        pattern=re.compile(r"(?:算了|不对|不是)[，,。.]\s*"),
        description="改口修正：算了/不对/不是",
        sub_type="临时改口",
    ),
    ClassificationRule(
        pattern=re.compile(r"^\s*(?:我|你)?(?:刚才)?说的(?:是|不是)?(?:那个|这个|另一个|别的)"),
        description="跳跃指代：我说的是那个/你说的是另一个",
        sub_type="跳跃",
    ),
]


# ---------------------------------------------------------------------------
# 4. Semantic (SEMANTIC) rules
# ---------------------------------------------------------------------------

# Known jargon / abbreviations / internal code vocabulary
SEMANTIC_KNOWN_TERMS: list[str] = [
    # Internet jargon
    "抓手", "赋能", "闭环", "对齐", "拉通", "颗粒度", "组合拳", "方法论",
    "势能", "心智", "矩阵", "赛道", "护城河", "飞轮", "杠杆", "沉淀",
    "中台", "打法", "破局", "链路", "顶层设计", "降本增效", "复盘",
    # Internet job levels
    "P8", "P9", "P7", "P6", "P10", "M3", "M4", "M2",
    # Technical abbreviations
    "RAG", "CRUD", "DDL", "DML", "DCL", "TCL", "CAP", "ACID", "BASE",
    "DDD", "TDD", "BDD", "DDoS", "WAF", "CDN", "DNS", "SLB", "RPC",
    "MQ", "IOC", "AOP", "ORM", "BFF", "SSR", "CSR", "NSG",
    # Business jargon
    "GMV", "ROI", "UV", "PV", "DAU", "MAU", "ARPU", "LTV", "CAC",
    "SKU", "SPU", "SOP", "OKR", "KPI", "MVP", "PRD", "BRD", "MRD",
    # Chinese abbreviations
    "大模型", "中台", "前中后台", "上云", "出海",
]

# Build regex matching a single term (with word boundaries; for English terms, require non-letter before/after)
SEMANTIC_RULES: list[ClassificationRule] = [
    ClassificationRule(
        pattern=re.compile(rf"(?:^|[^a-zA-Z]){re.escape(term)}(?:[^a-zA-Z]|$)"),
        description=f"已知非标准术语/黑话：{term}",
        sub_type="黑话/缩写",
    )
    for term in SEMANTIC_KNOWN_TERMS
]
# Supplement: all-caps abbreviations (>=2 letters, suspected abbreviation)
SEMANTIC_RULES.append(
    ClassificationRule(
        pattern=re.compile(r"\b[A-Z]{2,6}\b"),
        description="疑似英文缩写（全大写 2-6 字母）",
        sub_type="缩写",
    )
)
# Supplement: Chinese + English mix (e.g. "PRD 文档", "CRM 系统"), not in jargon vocabulary
SEMANTIC_RULES.append(
    ClassificationRule(
        pattern=re.compile(r"[A-Z]{2,6}[\s]*[一-龥]"),
        description="中英混合术语：CRM 系统 / PRD 文档",
        sub_type="中英混合",
    )
)


# ---------------------------------------------------------------------------
# 5. Subjective (SUBJECTIVE) rules
# ---------------------------------------------------------------------------

SUBJECTIVE_RULES: list[ClassificationRule] = [
    ClassificationRule(
        pattern=re.compile(r"最(?:有|具)?性价比"),
        description="主观判断词：最(有/具)性价比",
        sub_type="性价比",
    ),
    ClassificationRule(
        pattern=re.compile(r"更(?:有|具)?性价比"),
        description="主观判断词：更(有/具)性价比",
        sub_type="性价比",
    ),
    ClassificationRule(
        pattern=re.compile(r"再(?:高级|便宜|好|强|简单|快|稳定)一点"),
        description="主观程度词：再高级一点/再便宜一点",
        sub_type="程度副词",
    ),
    ClassificationRule(
        pattern=re.compile(r"更(?:好|高级|便宜|强|简单|快|稳定|合适|有性价比)"),
        description="主观比较词：更好/更高级/更便宜/更有性价比",
        sub_type="比较词",
    ),
    ClassificationRule(
        pattern=re.compile(r"最(?:划算|优|好|便宜|贵|快|稳定)"),
        description="主观最高级词：最划算/最优/最好",
        sub_type="最高级",
    ),
    ClassificationRule(
        pattern=re.compile(r"(?:大|小|高|低|多|少)一点"),
        description="主观程度词：大一点/小一点",
        sub_type="程度副词",
    ),
    ClassificationRule(
        pattern=re.compile(r"(?:有点|稍微)(?:太)?(?:贵|便宜|大|小|高|低|复杂|简单)"),
        description="主观感受词：有点贵/稍微复杂",
        sub_type="感受词",
    ),
    ClassificationRule(
        pattern=re.compile(r"(?:看起来|感觉|觉得)(?:比较|有点)?(?:好|高级|便宜|差)"),
        description="主观感受词：看起来高级/感觉有点差",
        sub_type="感受词",
    ),
]


# ---------------------------------------------------------------------------
# 6. External fact (EXTERNAL_FACT) rules
# ---------------------------------------------------------------------------

# Combination of time-sensitivity keywords and data-demand keywords
EXTERNAL_FACT_TIME_KEYWORDS = r"(?:最近|现在|当前|最新|实时|目前|当下|今日|今天)"
EXTERNAL_FACT_QUERY_KEYWORDS = r"(?:哪个|哪个|最|多少|有哪些|是什么|怎么样)"

EXTERNAL_FACT_RULES: list[ClassificationRule] = [
    ClassificationRule(
        pattern=re.compile(
            rf"{EXTERNAL_FACT_TIME_KEYWORDS}(?:[^，。]{{0,15}})?{EXTERNAL_FACT_QUERY_KEYWORDS}"
        ),
        description="时效+数据需求：最近哪个框架更火/现在最便宜的是哪个",
        sub_type="实时数据",
    ),
    ClassificationRule(
        pattern=re.compile(r"(?:最近|现在|当前|最新|目前)(?:[^，。]{0,15})?(?:火|热|流行|便宜|贵|好)"),
        description="时效+趋势判断：最近哪个框架更火/现在什么流行",
        sub_type="实时数据",
    ),
    ClassificationRule(
        pattern=re.compile(r"(?:实时|在线)(?:[^，。]{0,15})?(?:价格|数据|行情|汇率|库存)"),
        description="实时数据：实时价格/在线行情",
        sub_type="实时数据",
    ),
    ClassificationRule(
        pattern=re.compile(r"(?:最新|今天|今日)(?:[^，。]{0,15})?(?:新闻|消息|公告|数据|版本)"),
        description="最新信息：最新版本/今日新闻",
        sub_type="最新信息",
    ),
    ClassificationRule(
        pattern=re.compile(r"(?:哪个|什么)(?:[^，。]{0,15})?(?:最便宜|最贵|最好|最火|最热)"),
        description="需检索比对：哪个最便宜/什么最好",
        sub_type="检索结果",
    ),
    ClassificationRule(
        pattern=re.compile(r"(?:排名|排行|榜单)(?:[^，。]{0,15})?(?:是什么|有哪些|怎么样)"),
        description="需工具返回：排名/榜单",
        sub_type="工具返回",
    ),
]


# ---------------------------------------------------------------------------
# Summary: full classification rule table
# ---------------------------------------------------------------------------

CLASSIFICATION_RULES: dict[InputProblemType, list[ClassificationRule]] = {
    InputProblemType.ANAPHORA: ANAPHORA_RULES,
    InputProblemType.MISSING: MISSING_RULES,
    InputProblemType.EXPRESSION: EXPRESSION_RULES,
    InputProblemType.SEMANTIC: SEMANTIC_RULES,
    InputProblemType.SUBJECTIVE: SUBJECTIVE_RULES,
    InputProblemType.EXTERNAL_FACT: EXTERNAL_FACT_RULES,
}


def get_all_rules() -> dict[InputProblemType, list[ClassificationRule]]:
    """Get all classification rules."""
    return CLASSIFICATION_RULES


def match_rules(
    text: str,
) -> dict[InputProblemType, list[tuple[ClassificationRule, str]]]:
    """Apply all classification rules to the input text.

    Args:
        text: User input text

    Returns:
        Match results: {InputProblemType: [(rule, matched_span), ...]}
    """
    matches: dict[InputProblemType, list[tuple[ClassificationRule, str]]] = {}
    for problem_type, rules in CLASSIFICATION_RULES.items():
        for rule in rules:
            m = rule.pattern.search(text)
            if m:
                matches.setdefault(problem_type, []).append((rule, m.group(0)))
    return matches
