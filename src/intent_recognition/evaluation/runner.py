"""Test runner (D15, D16).

Runs a recognition function over a :class:`TestSet`, collects per-case
results, and produces evaluation reports via :class:`MetricsCalculator`.
"""

from __future__ import annotations

from typing import Any, Callable

from ..models import IntentRecognitionResult
from .metrics import MetricsCalculator
from .test_set import TestSet

# A recognition function takes the user-text (and optionally the full case
# dict) and returns an IntentRecognitionResult.
RecognitionFn = Callable[..., IntentRecognitionResult]


class TestRunner:
    """Run a recognizer against a test set and compute metrics.

    Parameters
    ----------
    recognize:
        Callable that accepts the user text (and optionally ``case=``)
        and returns an :class:`IntentRecognitionResult`.
    """

    # Mark as non-test class so pytest does not try to collect it.
    __test__ = False

    def __init__(self, recognize: RecognitionFn) -> None:
        self._recognize = recognize

    # ------------------------------------------------------------------
    # Single case
    # ------------------------------------------------------------------
    def run_single(self, case: dict[str, Any]) -> dict[str, Any]:
        """Run one case and produce a result dict for metrics.

        The returned dict is suitable for direct ingestion by
        :class:`MetricsCalculator`.
        """
        text = case.get("text", "")
        # Prefer the richer calling convention when the callable accepts it.
        try:
            result = self._recognize(text, case=case)
        except TypeError:
            result = self._recognize(text)

        expected_intent = case.get("expected_intent")
        predicted_intent = result.intent

        # An intent match: both equal (including both None for rejection).
        intent_matches = predicted_intent == expected_intent

        # Slot bookkeeping
        expected_slots = case.get("expected_slots") or {}
        predicted_slots = result.slots or {}

        correct_slots = _count_correct_slots(predicted_slots, expected_slots)
        expected_slot_count = _count_slots(expected_slots)
        extracted_slot_count = _count_slots(predicted_slots)

        should_reject = bool(case.get("should_reject", False))
        system_rejected = predicted_intent is None
        correctly_rejected = should_reject and system_rejected
        wrongly_rejected = (not should_reject) and system_rejected
        missed_rejection = should_reject and (not system_rejected)

        should_clarify = bool(case.get("should_clarify", False))
        clarified = bool(result.need_clarification)
        correctly_clarified = should_clarify and clarified
        # Convergence is provided by the case when available
        converged = bool(case.get("converged", False)) if clarified else False

        # Required-slot completeness: case may assert this directly, otherwise
        # infer from missing_slots on the result.
        if "required_slots_complete" in case:
            required_complete = bool(case["required_slots_complete"])
        else:
            required_complete = not result.missing_slots

        # Constraint bookkeeping (optional)
        correct_constraints = case.get("correct_constraints", 0)
        total_constraints = case.get("total_constraints", 0)
        if not total_constraints:
            # Fall back to counting constraints on the result
            total_constraints = (
                len(result.hard_constraints) + len(result.soft_constraints)
            )
            # Assume correctness only when counts match expected (best effort)
            correct_constraints = total_constraints if intent_matches else 0

        # Slot updates (optional, cross-turn cases)
        slot_updates_correct = case.get("slot_updates_correct", 0)
        slot_updates_total = case.get("slot_updates_total", 0)

        task_succeeded = case.get(
            "task_succeeded", intent_matches and not result.missing_slots
        )

        # Top-k predictions: prefer case-provided, else fall back to
        # ``sub_intents`` on the result, else a singleton.
        topk = case.get("topk_predictions")
        if not topk:
            topk = (
                list(result.sub_intents)
                if result.sub_intents
                else ([predicted_intent] if predicted_intent else [])
            )

        return {
            "case_id": case.get("case_id"),
            "scenario_type": case.get("scenario_type"),
            "text": text,
            "predicted_intent": predicted_intent,
            "expected_intent": expected_intent,
            "predicted_slots": predicted_slots,
            "expected_slots": expected_slots,
            "topk_predictions": topk,
            "is_correct": intent_matches and not wrongly_rejected,
            "should_reject": should_reject,
            "correctly_rejected": correctly_rejected,
            "wrongly_rejected": wrongly_rejected,
            "missed_rejection": missed_rejection,
            "should_clarify": should_clarify,
            "clarified": clarified,
            "correctly_clarified": correctly_clarified,
            "converged": converged,
            "extracted_slots": predicted_slots,
            "correct_slots": correct_slots,
            "required_slots_complete": required_complete,
            "slot_updates_correct": slot_updates_correct,
            "slot_updates_total": slot_updates_total,
            "correct_constraints": correct_constraints,
            "total_constraints": total_constraints,
            "task_succeeded": task_succeeded,
            "confidence": result.confidence,
        }

    # ------------------------------------------------------------------
    # Whole test set
    # ------------------------------------------------------------------
    def run(self, test_set: TestSet) -> list[dict[str, Any]]:
        """Run every case in ``test_set`` and return the collected results."""
        return [self.run_single(case) for case in test_set.cases]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute metrics for a list of result dicts."""
        return MetricsCalculator(results).generate_report()

    def generate_report(self, test_set: TestSet) -> dict[str, Any]:
        """Run the test set and return a full report with metrics + coverage."""
        results = self.run(test_set)
        metrics = MetricsCalculator(results).generate_report()
        coverage = test_set.coverage_report()
        return {
            "metrics": metrics,
            "coverage": coverage,
            "results": results,
        }

    # ------------------------------------------------------------------
    # Fine-tuning comparison (D24)
    # ------------------------------------------------------------------
    def evaluate_fine_tuned(
        self,
        fine_tuned_runner: "TestRunner",
        test_set: TestSet,
    ) -> dict[str, Any]:
        """Compare a fine-tuned runner against this baseline on the same test set.

        Used as the integration point for D24: after offline fine-tuning on
        data exported by :class:`TrainingDataExporter`, the fine-tuned model
        is wrapped in a second ``TestRunner`` and compared to the baseline
        on a fixed evaluation set. The returned dict drives the deploy vs
        rollback decision.

        Args:
            fine_tuned_runner: A ``TestRunner`` whose recognizer uses the
                fine-tuned LLM client.
            test_set: The evaluation test set.

        Returns:
            Dict with ``before_accuracy``, ``after_accuracy``,
            ``improvement`` (after - before), and ``recommendation``
            (``"deploy"`` when ``improvement > 0`` else ``"rollback"``).
        """
        # Baseline (self)
        baseline_results = self.run(test_set)
        baseline_report = self.evaluate(baseline_results)
        # MetricsCalculator.generate_report() puts top1_accuracy at the top
        # level of the report dict (see metrics.py).
        baseline_accuracy = baseline_report.get("top1_accuracy", 0.0)

        # Fine-tuned
        ft_results = fine_tuned_runner.run(test_set)
        ft_report = fine_tuned_runner.evaluate(ft_results)
        ft_accuracy = ft_report.get("top1_accuracy", 0.0)

        improvement = ft_accuracy - baseline_accuracy
        recommendation = "deploy" if improvement > 0 else "rollback"

        return {
            "before_accuracy": baseline_accuracy,
            "after_accuracy": ft_accuracy,
            "improvement": improvement,
            "recommendation": recommendation,
        }

    # ------------------------------------------------------------------
    # D29 task 5.9: Online failure-sample collection
    # ------------------------------------------------------------------
    def collect_online_failures(
        self,
        cases: list[dict[str, Any]],
        results: list[dict[str, Any]] | None = None,
        failure_signals: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Collect online failure samples for feedback to the test set.

        A case is flagged as a failure when either:
        - ``results`` is provided and the corresponding result has
          ``is_correct=False`` (intent mismatch / wrong rejection), OR
        - the case text contains one of the D11 implicit-failure signals
          (``你理解错了`` / ``不是这个意思`` / ...) in ``failure_signals``.

        Returned samples are in the shape expected by
        :meth:`TestSet.import_online_samples` (``text`` +
        ``expected_intent`` + optional annotations).

        Args:
            cases: Online cases (each must contain ``text``).
            results: Optional parallel list of result dicts (same length).
                When omitted, only failure-signal detection is applied.
            failure_signals: Override failure-signal phrases; defaults to
                the D11 phrases used by the pipeline.

        Returns:
            List of failure samples ready to feed back into a test set.
        """
        if failure_signals is None:
            failure_signals = [
                "你理解错了",
                "不是这个意思",
                "不是，我不是",
                "不对，我要的是",
            ]
        failures: list[dict[str, Any]] = []
        for idx, case in enumerate(cases):
            if not isinstance(case, dict):
                continue
            text = case.get("text") or case.get("input") or ""
            if not text:
                continue
            is_failure = False
            # 1. Result-driven failure (mismatch / wrong rejection)
            if results is not None and idx < len(results):
                r = results[idx] or {}
                if not r.get("is_correct", True):
                    is_failure = True
            # 2. Implicit-failure-signal-driven failure (D11)
            if not is_failure:
                for sig in failure_signals:
                    if sig in text:
                        is_failure = True
                        break
            if is_failure:
                failures.append({
                    "text": text,
                    "expected_intent": case.get("expected_intent"),
                    "should_reject": case.get("should_reject", False),
                    "should_clarify": case.get("should_clarify", False),
                    "source": "online_failure",
                    "original_case_id": case.get("case_id"),
                })
        return failures

    # ------------------------------------------------------------------
    # D30 task 6.3: Solution evolution stage assessment
    # ------------------------------------------------------------------
    def assess_evolution_stage(
        self,
        config: Any | None = None,
    ) -> str:
        """Assess the current solution-evolution stage (D30 task 6.3).

        Returns one of:
        - ``"v0"``: Single deep LLM only, no normalization, no code layer.
        - ``"v1-stage1"``: L2 lightweight LLM introduced, L1 code layer off.
        - ``"v1-stage2"``: L1 code layer + L2 lightweight LLM + L3 deep LLM
          (three-layer waterfall). D5 multi-signal fusion enabled.
        - ``"v1-stage3"``: v1-stage2 plus D17-D24 advanced mechanisms
          (retrieval, dynamic fewshot, vector fallback, reuse strategy,
          multi-recognizer arbiter, fine-tuning integration point).
        - ``"v2"``: v1-stage3 plus D26-D31 interview-insights mechanisms
          (evidence grading, sub-task boundary, five-factor arbitration,
          extended evaluation, structured output protocol).

        Args:
            config: ``IntentRecognitionConfig`` to inspect. When ``None``,
                the runner has no config reference; callers should pass the
                pipeline's config explicitly for an accurate assessment.
        """
        if config is None:
            # No config to inspect: assume the most basic stage
            return "v0"

        # v0: only deep LLM enabled, everything else off
        if (
            not getattr(config, "enable_code_layer", False)
            and getattr(config, "enable_lightweight_llm", False) is False
            and getattr(config, "enable_deep_llm", True)
        ):
            return "v0"

        # v1-stage1: L2 on, L1 off
        if (
            not getattr(config, "enable_code_layer", False)
            and getattr(config, "enable_lightweight_llm", True)
            and getattr(config, "enable_deep_llm", True)
        ):
            return "v1-stage1"

        # v1-stage2: three-layer waterfall on
        three_layer = (
            getattr(config, "enable_code_layer", False)
            and getattr(config, "enable_lightweight_llm", True)
            and getattr(config, "enable_deep_llm", True)
        )
        if not three_layer:
            return "v1-stage1"

        # v1-stage3: any D17-D24 extension enabled
        d17_d24_enabled = (
            getattr(getattr(config, "retrieval", None), "enable", False)
            or getattr(getattr(config, "dynamic_fewshot", None), "dynamic_enabled", False)
            or getattr(getattr(config, "vector_fallback", None), "enable", False)
            or getattr(getattr(config, "reuse_strategy", None), "enable", False)
            or getattr(getattr(config, "arbiter", None), "enable", False)
            or getattr(getattr(config, "fine_tuning", None), "enable", False)
        )

        # v2: any D26-D31 extension enabled (evidence grading, five-factor
        # arbitration, structured output, extended evaluation, etc.)
        # NOTE: D26-D31 default ON (additive), so this check MUST come before
        # the D17-D24 gate below — a default config skips v1-stage3 and lands
        # directly at v2.
        d26_d31_enabled = (
            getattr(getattr(config, "evidence", None), "enable_grading", False)
            or getattr(getattr(config, "arbitration", None), "enable_five_factor", False)
            or getattr(getattr(config, "boundary", None), "enable_sub_tasks", False)
            or getattr(getattr(config, "protocol", None), "enable_structured_output", False)
        )
        if d26_d31_enabled:
            return "v2"
        if d17_d24_enabled:
            return "v1-stage3"
        return "v1-stage2"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _count_slots(value: Any) -> int:
    """Count non-None slots in a dict (or items in a list)."""
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return sum(1 for v in value.values() if v is not None)
    return 0


def _count_correct_slots(
    predicted: dict[str, Any], expected: dict[str, Any]
) -> int:
    """Count slots whose predicted value equals the expected value."""
    if not expected:
        return 0
    correct = 0
    for k, v_expected in expected.items():
        if v_expected is None:
            continue
        v_predicted = predicted.get(k) if isinstance(predicted, dict) else None
        if v_predicted is not None and v_predicted == v_expected:
            correct += 1
    return correct
