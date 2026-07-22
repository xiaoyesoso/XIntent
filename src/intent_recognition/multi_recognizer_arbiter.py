"""Multi-recognizer arbitration (D23).

Runs multiple recognizers (e.g. ``vector``, ``rule``, ``lightweight_llm``)
on the same input and produces a single arbitrated result.

Two arbitration modes are supported:

- ``"vote"``: each recognizer casts one vote (its predicted intent). The
  intent with the strict majority wins; ties produce ``None`` so the
  caller can escalate to clarification.
- ``"weighted_score"``: each recognizer contributes
  ``weight * confidence`` to the intent it predicted. The intent with
  the highest cumulative score wins; the reported confidence is the
  score normalized by the total weight of participating recognizers.

Both modes set ``source=ARBITER_VOTE`` / ``ARBITER_WEIGHTED`` so callers
can distinguish arbitrated results from a single-recognizer hit.

The arbiter is opt-in (``ArbiterConfig.enable=False`` by default) and
deliberately stateless with respect to the pipeline: it only consumes
recognizer callables and produces results.
"""

from __future__ import annotations

from typing import Any, Callable

from .config import ArbiterConfig
from .models import IntentRecognitionResult, RecognitionSource


# Type alias: a recognizer callable takes (text, context) and returns an
# IntentRecognitionResult or None when it has no opinion.
Recognizer = Callable[[str, dict[str, Any] | None], IntentRecognitionResult | None]


class MultiRecognizerArbiter:
    """D23: Arbitrate between multiple recognizers.

    Parameters
    ----------
    config:
        :class:`ArbiterConfig` (enable flag, mode, recognizer name list,
        per-recognizer weights).
    recognizers:
        Optional initial ``{name: callable}`` mapping. Recognizers can
        also be registered later via :meth:`register`.
    """

    def __init__(
        self,
        config: ArbiterConfig,
        recognizers: dict[str, Recognizer] | None = None,
    ) -> None:
        self._config = config
        self._recognizers: dict[str, Recognizer] = dict(recognizers or {})

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, name: str, recognizer: Recognizer) -> None:
        """Register (or overwrite) a recognizer by ``name``."""
        self._recognizers[name] = recognizer

    def validate_recognizers(self) -> None:
        """Ensure every name in ``config.recognizers`` is registered.

        Raises
        ------
        ValueError
            When any configured recognizer name is not registered.
        """
        for name in self._config.recognizers:
            if name not in self._recognizers:
                raise ValueError(f"Unknown recognizer: {name}")

    # ------------------------------------------------------------------
    # Arbitration
    # ------------------------------------------------------------------
    def arbitrate(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> IntentRecognitionResult | None:
        """Run all configured recognizers and arbitrate between them.

        Returns ``None`` when:
        - the arbiter is disabled, or
        - fewer than 2 recognizers are configured (no arbitration
          possible), or
        - no recognizer produces a result, or
        - vote mode ends in a tie.

        Otherwise returns an :class:`IntentRecognitionResult` with the
        arbitrated intent and ``source=ARBITER_VOTE`` or
        ``source=ARBITER_WEIGHTED``.
        """
        if not self._config.enable:
            return None

        # Need at least 2 recognizers for arbitration to be meaningful.
        if len(self._config.recognizers) < 2:
            return None

        # Validate up-front so callers get a clear error for misconfigured
        # recognizer names rather than a silent skip.
        self.validate_recognizers()

        # Run each configured recognizer and collect (name, result) pairs
        # for recognizers that returned a non-None result. Recognizer
        # exceptions are treated as "no opinion" so a misbehaving
        # recognizer cannot crash the arbiter.
        participating: list[tuple[str, IntentRecognitionResult]] = []
        for name in self._config.recognizers:
            recognizer = self._recognizers[name]
            try:
                result = recognizer(text, context)
            except Exception:
                continue
            if result is not None and result.intent is not None:
                participating.append((name, result))

        if not participating:
            return None

        if len(participating) == 1:
            # Only one recognizer has an opinion -> no arbitration needed.
            return participating[0][1]

        if self._config.mode == "vote":
            return self._vote(participating)
        if self._config.mode == "weighted_score":
            return self._weighted_score(participating)

        # Unknown mode: behave defensively (no arbitration).
        return None

    # ------------------------------------------------------------------
    # Internal: voting modes
    # ------------------------------------------------------------------
    def _vote(
        self,
        participating: list[tuple[str, IntentRecognitionResult]],
    ) -> IntentRecognitionResult | None:
        """Plurality vote. Ties produce ``None``."""
        # Group recognizers by the intent they predicted.
        votes: dict[str, list[IntentRecognitionResult]] = {}
        for _, result in participating:
            votes.setdefault(result.intent, []).append(result)

        # Find the intent(s) with the most votes.
        max_count = max(len(results) for results in votes.values())
        winners = [intent for intent, results in votes.items() if len(results) == max_count]

        if len(winners) != 1:
            # Tie (or all-zero degenerate case): no clear winner.
            return None

        winning_results = votes[winners[0]]
        avg_confidence = sum(r.confidence for r in winning_results) / len(winning_results)
        return IntentRecognitionResult(
            intent=winners[0],
            confidence=avg_confidence,
            source=RecognitionSource.ARBITER_VOTE,
            layer_reached=1,
            signals={
                "arbiter_votes": float(max_count),
                "arbiter_participants": float(len(participating)),
            },
        )

    def _weighted_score(
        self,
        participating: list[tuple[str, IntentRecognitionResult]],
    ) -> IntentRecognitionResult | None:
        """Weighted-score arbitration.

        For each intent, accumulate ``weight * confidence`` from every
        recognizer that voted for it. The highest cumulative score wins
        and the reported confidence is

            score / sum(weights of participating recognizers)

        clamped to ``[0, 1]``.
        """
        scores: dict[str, float] = {}
        supporting: dict[str, list[IntentRecognitionResult]] = {}
        total_weight = 0.0

        for name, result in participating:
            weight = self._config.weights.get(name, 0.0)
            total_weight += weight
            scores[result.intent] = scores.get(result.intent, 0.0) + weight * result.confidence
            supporting.setdefault(result.intent, []).append(result)

        if not scores:
            return None

        # Highest cumulative score wins. Ties (rare for floats) fall back
        # to insertion order via max() which is fine for determinism.
        winner = max(scores, key=lambda intent: scores[intent])
        winning_score = scores[winner]

        if total_weight <= 0.0:
            confidence = 0.0
        else:
            confidence = min(winning_score / total_weight, 1.0)

        return IntentRecognitionResult(
            intent=winner,
            confidence=confidence,
            source=RecognitionSource.ARBITER_WEIGHTED,
            layer_reached=1,
            signals={
                "arbiter_score": winning_score,
                "arbiter_total_weight": total_weight,
            },
        )
