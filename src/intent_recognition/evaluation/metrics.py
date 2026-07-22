"""Evaluation metrics calculator (D15, D16).

Computes Top-1 / Top-K accuracy, per-intent accuracy, rejection metrics,
clarification metrics, slot-filling metrics, and overall task-success rate
from a list of test-result dicts.
"""

from __future__ import annotations

from typing import Any


def _safe_div(num: float, den: float) -> float:
    """Safe division that returns 0.0 when the denominator is zero."""
    if den == 0:
        return 0.0
    return num / den


class MetricsCalculator:
    """Compute evaluation metrics from collected test results.

    Each result dict SHOULD contain (all optional but recommended):

    - ``predicted_intent`` (str | None): the intent the system predicted,
      ``None`` means the system rejected the input.
    - ``expected_intent`` (str | None): the ground-truth intent,
      ``None`` means the input should have been rejected.
    - ``is_correct`` (bool): whether the prediction matches the expectation.
    - ``topk_predictions`` (list[str]): ordered top-k predictions.
    - ``should_reject`` (bool): whether the case is a rejection scenario.
    - ``correctly_rejected`` (bool): whether the system rejected correctly.
    - ``wrongly_rejected`` (bool): whether the system rejected when it
      should have recognized an intent.
    - ``should_clarify`` (bool): whether the case should trigger clarification.
    - ``correctly_clarified`` (bool): whether the system clarified correctly.
    - ``clarified`` (bool): whether the system triggered clarification.
    - ``converged`` (bool): whether the user converged after clarification.
    - ``extracted_slots`` (list[str] | dict): slots the system extracted.
    - ``expected_slots`` (list[str] | dict): slots that should be extracted.
    - ``correct_slots`` (int): count of correctly extracted slots.
    - ``required_slots_complete`` (bool): whether all required slots are present.
    - ``slot_updates_correct`` (int): correct cross-turn slot updates.
    - ``slot_updates_total`` (int): total cross-turn slot updates.
    - ``correct_constraints`` (int): correctly identified constraints.
    - ``total_constraints`` (int): total constraints in the case.
    - ``task_succeeded`` (bool): whether the end-to-end task succeeded.
    """

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self._results: list[dict[str, Any]] = list(results or [])

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def add_result(self, result: dict[str, Any]) -> None:
        """Append a single test result."""
        self._results.append(result)

    def add_results(self, results: list[dict[str, Any]]) -> None:
        """Append multiple test results."""
        self._results.extend(results)

    @property
    def results(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the stored results."""
        return list(self._results)

    # ------------------------------------------------------------------
    # Accuracy metrics (D15)
    # ------------------------------------------------------------------
    def top1_accuracy(self) -> float:
        """Fraction of cases where ``predicted_intent == expected_intent``."""
        total = len(self._results)
        if total == 0:
            return 0.0
        correct = sum(1 for r in self._results if r.get("is_correct"))
        return _safe_div(correct, total)

    def topk_accuracy(self, k: int = 3) -> float:
        """Fraction of cases where ``expected_intent`` is in the top-k."""
        total = len(self._results)
        if total == 0:
            return 0.0
        hit = 0
        for r in self._results:
            if r.get("is_correct"):
                hit += 1
                continue
            topk = r.get("topk_predictions") or []
            expected = r.get("expected_intent")
            if expected and expected in topk[:k]:
                hit += 1
        return _safe_div(hit, total)

    def per_intent_accuracy(self) -> dict[str, float]:
        """Per-intent accuracy breakdown keyed by expected intent name.

        Cases with ``expected_intent is None`` (i.e. should-reject cases)
        are excluded from this breakdown; they are handled by
        :meth:`rejection_metrics`.
        """
        buckets: dict[str, list[bool]] = {}
        for r in self._results:
            expected = r.get("expected_intent")
            if not expected:
                continue
            buckets.setdefault(expected, []).append(bool(r.get("is_correct")))
        return {
            intent: _safe_div(sum(1 for ok in oks if ok), len(oks))
            for intent, oks in buckets.items()
        }

    # ------------------------------------------------------------------
    # Rejection metrics (D15)
    # ------------------------------------------------------------------
    def rejection_metrics(self) -> dict[str, float]:
        """Return rejection precision / false-rejection / missed-rejection.

        - ``拒识准确率`` (rejection_precision):
            correctly rejected / total rejected
        - ``误拒率`` (false_rejection_rate):
            wrongly rejected / total non-rejectable
        - ``漏拒率`` (missed_rejection_rate):
            should reject but didn't / total rejectable
        """
        total_rejected = 0
        correctly_rejected = 0
        total_non_rejectable = 0
        wrongly_rejected = 0
        total_rejectable = 0
        missed_rejection = 0

        for r in self._results:
            should_reject = bool(r.get("should_reject", False))
            predicted = r.get("predicted_intent")
            system_rejected = predicted is None

            if should_reject:
                total_rejectable += 1
                if system_rejected:
                    total_rejected += 1
                    correctly_rejected += 1
                else:
                    missed_rejection += 1
            else:
                total_non_rejectable += 1
                if system_rejected:
                    total_rejected += 1
                    wrongly_rejected += 1

        return {
            "拒识准确率": _safe_div(correctly_rejected, total_rejected),
            "误拒率": _safe_div(wrongly_rejected, total_non_rejectable),
            "漏拒率": _safe_div(missed_rejection, total_rejectable),
        }

    # ------------------------------------------------------------------
    # Clarification metrics (D15)
    # ------------------------------------------------------------------
    def clarification_metrics(self) -> dict[str, float]:
        """Return clarification precision and post-clarification convergence.

        - ``澄清触发准确率`` (clarification_precision):
            correctly triggered / total triggered
        - ``澄清后收敛率`` (convergence_rate):
            resolved after clarification / total clarified
        """
        total_triggered = 0
        correctly_triggered = 0
        total_clarified = 0
        converged = 0

        for r in self._results:
            should_clarify = r.get("should_clarify", False)
            clarified = bool(r.get("clarified", False))
            if clarified:
                total_triggered += 1
                if should_clarify:
                    correctly_triggered += 1
                # Track convergence only for cases that actually clarified
                total_clarified += 1
                if r.get("converged", False):
                    converged += 1

        return {
            "澄清触发准确率": _safe_div(correctly_triggered, total_triggered),
            "澄清后收敛率": _safe_div(converged, total_clarified),
        }

    # ------------------------------------------------------------------
    # Slot filling metrics (D15)
    # ------------------------------------------------------------------
    def slot_filling_metrics(self) -> dict[str, float]:
        """Return the five slot-filling metrics.

        - ``槽位准确率`` (slot_precision): correct slots / extracted slots
        - ``槽位召回率`` (slot_recall): extracted slots / expected slots
        - ``必填槽位完整率`` (required_slot_completeness):
            sessions with all required / total sessions
        - ``槽位更新准确率`` (slot_update_accuracy):
            correct updates / total updates
        - ``约束识别准确率`` (constraint_accuracy):
            correct constraints / total constraints
        """
        total_extracted = 0
        total_expected = 0
        total_correct_slots = 0
        sessions_with_required = 0
        total_sessions = 0
        total_updates = 0
        correct_updates = 0
        total_constraints = 0
        correct_constraints = 0

        for r in self._results:
            extracted = r.get("extracted_slots")
            expected = r.get("expected_slots")
            correct_slots = r.get("correct_slots", 0)

            n_extracted = self._count_slots(extracted)
            n_expected = self._count_slots(expected)
            n_correct = correct_slots if isinstance(correct_slots, int) else 0

            total_extracted += n_extracted
            total_expected += n_expected
            total_correct_slots += n_correct

            # Required-slot completeness is per-session
            if r.get("required_slots_complete") is not None:
                total_sessions += 1
                if r.get("required_slots_complete"):
                    sessions_with_required += 1
            elif "required_slots_complete" in r:
                total_sessions += 1
                if r.get("required_slots_complete"):
                    sessions_with_required += 1

            cu = r.get("slot_updates_correct", 0)
            tu = r.get("slot_updates_total", 0)
            if isinstance(cu, int):
                correct_updates += cu
            if isinstance(tu, int):
                total_updates += tu

            cc = r.get("correct_constraints", 0)
            tc = r.get("total_constraints", 0)
            if isinstance(cc, int):
                correct_constraints += cc
            if isinstance(tc, int):
                total_constraints += tc

        return {
            "槽位准确率": _safe_div(total_correct_slots, total_extracted),
            "槽位召回率": _safe_div(total_correct_slots, total_expected),
            "必填槽位完整率": _safe_div(sessions_with_required, total_sessions),
            "槽位更新准确率": _safe_div(correct_updates, total_updates),
            "约束识别准确率": _safe_div(correct_constraints, total_constraints),
        }

    @staticmethod
    def _count_slots(value: Any) -> int:
        """Count slots from either a list or a dict representation."""
        if value is None:
            return 0
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            # Non-None values count as extracted/expected slots
            return sum(1 for v in value.values() if v is not None)
        return 0

    # ------------------------------------------------------------------
    # Overall task success (D15)
    # ------------------------------------------------------------------
    def task_success_rate(self) -> float:
        """Fraction of cases where the end-to-end task succeeded."""
        total = 0
        succeeded = 0
        for r in self._results:
            if "task_succeeded" in r:
                total += 1
                if r.get("task_succeeded"):
                    succeeded += 1
        if total == 0:
            # Fall back to Top-1 when task_succeeded is not annotated
            return self.top1_accuracy()
        return _safe_div(succeeded, total)

    # ------------------------------------------------------------------
    # D29: Extended evaluation metrics
    # ------------------------------------------------------------------
    def compute_confusion_matrix(self) -> dict[str, dict[str, int]]:
        """Return an N x N confusion matrix (D29 task 5.1).

        Outer key = expected intent (string, ``"__reject__"`` for None).
        Inner key = predicted intent (same convention).
        Cell value = count of cases with that (expected, predicted) pair.
        Labels not present in the data are omitted (true sparse matrix).
        """
        matrix: dict[str, dict[str, int]] = {}
        for r in self._results:
            expected = r.get("expected_intent") or "__reject__"
            predicted = r.get("predicted_intent") or "__reject__"
            matrix.setdefault(expected, {})
            matrix[expected][predicted] = matrix[expected].get(predicted, 0) + 1
        return matrix

    def compute_hard_sample_accuracy(self) -> float:
        """Return accuracy computed over hard samples only (D29 task 5.2).

        A case is "hard" when ``is_hard=True``. Cases without the flag are
        excluded from both numerator and denominator. Falls back to Top-1
        accuracy over all cases when no hard samples are annotated.
        """
        hard_total = 0
        hard_correct = 0
        for r in self._results:
            if r.get("is_hard"):
                hard_total += 1
                if r.get("is_correct"):
                    hard_correct += 1
        if hard_total == 0:
            return self.top1_accuracy()
        return _safe_div(hard_correct, hard_total)

    def compute_rejection_accuracy(self) -> dict[str, float]:
        """Return rejection precision / false-reject / missed-reject (D29 5.3).

        Distinct from :meth:`rejection_metrics` (which uses Chinese keys) by
        using English keys for programmatic access:
        - ``false_reject_rate``: 误拒率 (wrongly rejected / non-rejectable)
        - ``missed_reject_rate``: 漏拒率 (missed / rejectable)
        - ``rejection_precision``: 拒识准确率 (correctly rejected / total rejected)
        """
        m = self.rejection_metrics()
        return {
            "rejection_precision": m["拒识准确率"],
            "false_reject_rate": m["误拒率"],
            "missed_reject_rate": m["漏拒率"],
        }

    def compute_clarification_convergence_rate(self) -> float:
        """Return first-clarification convergence rate (D29 task 5.4).

        Fraction of clarification-triggered cases that converged on the
        *first* clarification round (i.e. ``clarification_rounds <= 1``).
        Cases without the ``clarification_rounds`` field are treated as
        ``1`` round if they converged (preserving backward compat).
        """
        total_clarified = 0
        first_round_converged = 0
        for r in self._results:
            if not r.get("clarified"):
                continue
            total_clarified += 1
            rounds = r.get("clarification_rounds", 1)
            if r.get("converged") and rounds <= 1:
                first_round_converged += 1
        return _safe_div(first_round_converged, total_clarified)

    def compute_slot_recall(self) -> float:
        """Return slot recall (D29 task 5.5).

        ``correct_slots / expected_slots`` across all cases. Mirrors the
        existing ``槽位召回率`` in :meth:`slot_filling_metrics` but exposed
        as a standalone method.
        """
        total_correct = 0
        total_expected = 0
        for r in self._results:
            n_expected = self._count_slots(r.get("expected_slots"))
            n_correct = r.get("correct_slots", 0)
            if isinstance(n_correct, int):
                total_correct += n_correct
            total_expected += n_expected
        return _safe_div(total_correct, total_expected)

    def compute_slot_completeness(self) -> float:
        """Return required-slot completeness (D29 task 5.5).

        Fraction of sessions where all required slots were filled.
        """
        total_sessions = 0
        complete_sessions = 0
        for r in self._results:
            if "required_slots_complete" in r:
                total_sessions += 1
                if r.get("required_slots_complete"):
                    complete_sessions += 1
        return _safe_div(complete_sessions, total_sessions)

    def compute_constraint_identification_rate(self) -> float:
        """Return hard+soft constraint identification rate (D29 task 5.6).

        ``correct_constraints / total_constraints`` across all cases. Falls
        back to ``1.0`` when no constraints are annotated (nothing to miss).
        """
        total_constraints = 0
        correct_constraints = 0
        for r in self._results:
            tc = r.get("total_constraints", 0)
            cc = r.get("correct_constraints", 0)
            if isinstance(tc, int):
                total_constraints += tc
            if isinstance(cc, int):
                correct_constraints += cc
        if total_constraints == 0:
            return 1.0
        return _safe_div(correct_constraints, total_constraints)

    def compute_state_update_accuracy(self) -> float:
        """Return multi-turn state update accuracy (D29 task 5.7).

        ``slot_updates_correct / slot_updates_total`` across cross-turn
        cases. Falls back to ``1.0`` when no state updates are annotated.
        """
        total_updates = 0
        correct_updates = 0
        for r in self._results:
            tu = r.get("slot_updates_total", 0)
            cu = r.get("slot_updates_correct", 0)
            if isinstance(tu, int):
                total_updates += tu
            if isinstance(cu, int):
                correct_updates += cu
        if total_updates == 0:
            return 1.0
        return _safe_div(correct_updates, total_updates)

    # ------------------------------------------------------------------
    # Full report (D15, D16, D29)
    # ------------------------------------------------------------------
    def generate_report(self) -> dict[str, Any]:
        """Build a full report with all metric categories and counts.

        D29 extends the D15 report with optional extended metrics. The
        extended metrics are always computed; callers can ignore them.
        """
        return {
            "total_cases": len(self._results),
            "top1_accuracy": self.top1_accuracy(),
            "topk_accuracy": self.topk_accuracy(),
            "per_intent_accuracy": self.per_intent_accuracy(),
            "rejection_metrics": self.rejection_metrics(),
            "clarification_metrics": self.clarification_metrics(),
            "slot_filling_metrics": self.slot_filling_metrics(),
            "task_success_rate": self.task_success_rate(),
            # D29 extended metrics
            "confusion_matrix": self.compute_confusion_matrix(),
            "hard_sample_accuracy": self.compute_hard_sample_accuracy(),
            "rejection_accuracy": self.compute_rejection_accuracy(),
            "clarification_convergence_rate": self.compute_clarification_convergence_rate(),
            "slot_recall": self.compute_slot_recall(),
            "slot_completeness": self.compute_slot_completeness(),
            "constraint_identification_rate": self.compute_constraint_identification_rate(),
            "state_update_accuracy": self.compute_state_update_accuracy(),
        }
