"""Adjective quantification module (corresponding to Group 6 / D13).

Transforms subjective judgment words into executable tool-call parameters.
"""

from .engine import QuantificationEngine
from .rules import DEFAULT_RULES, get_alternative_rule, get_rule

__all__ = [
    "QuantificationEngine",
    "DEFAULT_RULES",
    "get_rule",
    "get_alternative_rule",
]
