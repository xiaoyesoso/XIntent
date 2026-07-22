"""Completeness checker (corresponds to D16).

Check items:
1. Whether subject-predicate-object is complete
2. Whether pronouns are fully resolved
3. Whether adjectives and extremes have output "quantifiable" content

Inputs that fail the check are marked as incomplete and trigger subsequent processing.
"""

from __future__ import annotations

from ..models import (
    CompletenessCheck,
    CompletenessStatus,
    NormalizationResult,
    NormalizationStage,
)


class CompletenessChecker:
    """Completeness checker (corresponds to D16)."""

    def check(self, result: NormalizationResult) -> CompletenessCheck:
        """Perform completeness check.

        Returns:
            CompletenessCheck: check result
        """
        # 1. Subject-predicate-object completeness
        spo_complete = self._check_spo(result)

        # 2. Pronouns fully resolved
        pronouns_resolved = self._check_pronouns(result)

        # 3. Adjectives quantified
        adjectives_quantified = self._check_adjectives(result)

        # Determine check result status
        if spo_complete and pronouns_resolved and adjectives_quantified:
            status = CompletenessStatus.COMPLETE
        elif not spo_complete:
            status = CompletenessStatus.INCOMPLETE_MISSING_ARGUMENT
        elif not pronouns_resolved:
            status = CompletenessStatus.INCOMPLETE_PRONOUN_UNRESOLVED
        else:
            status = CompletenessStatus.INCOMPLETE_ADJECTIVE_UNQUANTIFIED

        return CompletenessCheck(
            spo_complete=spo_complete,
            pronouns_resolved=pronouns_resolved,
            adjectives_quantified=adjectives_quantified,
            result=status,
        )

    def check_and_route(
        self, result: NormalizationResult
    ) -> tuple[CompletenessCheck, NormalizationStage | None, str | None]:
        """Check and determine routing.

        Returns:
            (check result, routing target, routing reason)
            - Routing target None means check passed, no routing needed
            - Routing target DEEP means deep normalization is needed
            - Routing reason is the description of why routing was triggered
        """
        check = self.check(result)

        if check.result == CompletenessStatus.COMPLETE:
            return check, None, None

        if check.result == CompletenessStatus.INCOMPLETE_ADJECTIVE_UNQUANTIFIED:
            return check, NormalizationStage.DEEP, "形容词未量化，路由至 deep-normalization"

        # Missing object / unresolved pronoun -> needs clarification (not routed to deep, handled by clarification)
        return check, None, f"校验未通过：{check.result.value}，需触发澄清机制"

    @staticmethod
    def _check_spo(result: NormalizationResult) -> bool:
        """Check whether subject-predicate-object is complete."""
        spo = result.spo
        # Subject and predicate are required; object depends on predicate type
        if not spo.predicate:
            return False
        if not spo.subject:
            return False
        # Some predicates don't need an object (e.g. "下雨"), but normalized input usually should have one
        # Lenient check here: having subject and predicate is considered basically complete
        return True

    @staticmethod
    def _check_pronouns(result: NormalizationResult) -> bool:
        """Check whether pronouns are fully resolved.

        Checks whether unresolved pronouns still exist in the normalized input.
        """
        # Pronoun patterns
        import re

        pronoun_patterns = [
            r"那个",
            r"这个",
            r"那些",
            r"这些",
            r"第[一二三四五六七八九十\d]+[个种项]",
            r"刚才(那个|那个)",
            r"前(一个|面那个)",
            r"上(一个|次那个)",
            r"另一个",
            r"其他的",
        ]

        normalized = result.normalized_input
        for pattern in pronoun_patterns:
            if re.search(pattern, normalized):
                # Check whether it has been handled in the resolution table
                # If the pronoun has a corresponding entry in pronoun_resolutions, it is considered handled
                # But if the pronoun still appears in the normalized input, it means it is not fully resolved
                return False
        return True

    @staticmethod
    def _check_adjectives(result: NormalizationResult) -> bool:
        """Check whether adjectives have been quantified."""
        for adj in result.quantifiable_adjectives:
            if not adj.quantified:
                return False
        return True
