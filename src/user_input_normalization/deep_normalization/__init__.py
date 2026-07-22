"""Deep normalization module (corresponding to Group 5 / D1 / D13).

Executes within the ReAct loop, handling inputs that pre-normalization cannot complete:
1. Judgment word quantification ("性价比" -> tool parameters)
2. External fact resolution ("现在最便宜的" -> tool return)
3. Observation-dependent backtracking resolution
4. Context window management
5. Result writeback to key fact storage
"""

from .normalizer import DeepNormalizer

__all__ = ["DeepNormalizer"]
