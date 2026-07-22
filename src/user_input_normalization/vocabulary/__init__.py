"""Vocabulary module (corresponding to Group 8 / D10 / D11 / D12).

Two-level vocabulary (public + personal) + self-iteration mechanism + industry profiling.
"""

from .offline_analyzer import OfflineAnalyzer
from .table import VocabularyTable

__all__ = [
    "VocabularyTable",
    "OfflineAnalyzer",
]
