"""Attribute + retrieval anaphora resolution module (corresponding to Group 9 / D9).

For long time-span, attribute anaphora (e.g. "上次推荐的看樱花的那个地方"),
uses two-step inference + compensation mechanism to complete anaphora resolution.

Flow:
1. Attribute extraction (task 9.1): extract attribute keywords from user input
2. Vector retrieval recall (task 9.2): use attribute keywords to search the dialogue history store
3. Two-step inference (task 9.3): after recall, the LLM infers the referent based on window content
4. Retrieval-failure compensation mechanism (task 9.4): extract attributes to trigger a tool call, then re-infer
5. Confidence assessment (task 9.5)
6. Resolution result writeback (task 9.6)
7. Recall quality monitoring (task 9.7)
"""

from .resolver import AttributeResolver

__all__ = ["AttributeResolver"]
