"""Input problem classification module (corresponding to task 2).

Provides classification capability for six categories of input problems:
- Anaphora (ANAPHORA)
- Missing (MISSING)
- Expression (EXPRESSION)
- Semantic (SEMANTIC)
- Subjective (SUBJECTIVE)
- External fact (EXTERNAL_FACT)
"""

from .classifier import (
    ClassificationResult,
    InputClassifier,
    create_default_classifier,
)
from .rules import (
    ANAPHORA_RULES,
    CLASSIFICATION_RULES,
    EXTERNAL_FACT_RULES,
    EXPRESSION_RULES,
    MISSING_RULES,
    SEMANTIC_KNOWN_TERMS,
    SEMANTIC_RULES,
    SUBJECTIVE_RULES,
    ClassificationRule,
    get_all_rules,
    match_rules,
)

__all__ = [
    "InputClassifier",
    "ClassificationResult",
    "ClassificationRule",
    "CLASSIFICATION_RULES",
    "ANAPHORA_RULES",
    "MISSING_RULES",
    "EXPRESSION_RULES",
    "SEMANTIC_RULES",
    "SEMANTIC_KNOWN_TERMS",
    "SUBJECTIVE_RULES",
    "EXTERNAL_FACT_RULES",
    "get_all_rules",
    "match_rules",
    "create_default_classifier",
]
