"""Evaluation metrics & test runner tests (Group 8: D15, D16)."""

import json
from pathlib import Path

import pytest

from intent_recognition import IntentRecognitionResult
from intent_recognition.evaluation import (
    MetricsCalculator,
    TestRunner,
    TestSet,
)
from intent_recognition.evaluation.test_set import SCENARIO_TYPES


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _make_result(
    *,
    intent: str | None = "product_recommendation",
    slots: dict | None = None,
    confidence: float = 0.9,
    need_clarification: bool = False,
    missing_slots: list[str] | None = None,
) -> IntentRecognitionResult:
    return IntentRecognitionResult(
        intent=intent,
        confidence=confidence,
        slots=slots or {},
        need_clarification=need_clarification,
        missing_slots=missing_slots or [],
    )


# ----------------------------------------------------------------------
# Top-1 / Top-K accuracy
# ----------------------------------------------------------------------
class TestTopKAccuracy:
    def test_top1_accuracy_85_of_100(self):
        results = [
            {"is_correct": True, "expected_intent": "a", "predicted_intent": "a"}
            for _ in range(85)
        ] + [
            {"is_correct": False, "expected_intent": "a", "predicted_intent": "b"}
            for _ in range(15)
        ]
        calc = MetricsCalculator(results)
        assert calc.top1_accuracy() == pytest.approx(0.85)

    def test_top1_empty_results(self):
        assert MetricsCalculator([]).top1_accuracy() == 0.0

    def test_topk_accuracy_90_of_100(self):
        results = [
            {
                "is_correct": True,
                "expected_intent": "a",
                "predicted_intent": "a",
                "topk_predictions": ["a", "b", "c"],
            }
            for _ in range(85)
        ] + [
            # 5 wrong but expected is in top-3
            {
                "is_correct": False,
                "expected_intent": "b",
                "predicted_intent": "a",
                "topk_predictions": ["a", "b", "c"],
            }
            for _ in range(5)
        ] + [
            # 10 wrong and not in top-3
            {
                "is_correct": False,
                "expected_intent": "z",
                "predicted_intent": "a",
                "topk_predictions": ["a", "b", "c"],
            }
            for _ in range(10)
        ]
        calc = MetricsCalculator(results)
        assert calc.topk_accuracy(k=3) == pytest.approx(0.90)

    def test_topk_with_k_value(self):
        results = [
            {
                "is_correct": False,
                "expected_intent": "c",
                "predicted_intent": "a",
                "topk_predictions": ["a", "b", "c", "d"],
            }
        ]
        calc = MetricsCalculator(results)
        # k=2 misses, k=3 hits
        assert calc.topk_accuracy(k=2) == 0.0
        assert calc.topk_accuracy(k=3) == 1.0

    def test_topk_with_is_correct_always_hits(self):
        # is_correct cases always count as hits
        results = [
            {
                "is_correct": True,
                "expected_intent": "a",
                "predicted_intent": "a",
                "topk_predictions": [],
            }
        ]
        assert MetricsCalculator(results).topk_accuracy(k=3) == 1.0


# ----------------------------------------------------------------------
# Per-intent accuracy
# ----------------------------------------------------------------------
class TestPerIntentAccuracy:
    def test_per_intent_breakdown(self):
        results = [
            # product_recommendation: 9 correct, 1 wrong -> 0.9
            *(
                {"is_correct": True, "expected_intent": "product_recommendation",
                 "predicted_intent": "product_recommendation"}
                for _ in range(9)
            ),
            {"is_correct": False, "expected_intent": "product_recommendation",
             "predicted_intent": "order_query"},
            # order_query: 7 correct, 3 wrong -> 0.7
            *(
                {"is_correct": True, "expected_intent": "order_query",
                 "predicted_intent": "order_query"}
                for _ in range(7)
            ),
            *(
                {"is_correct": False, "expected_intent": "order_query",
                 "predicted_intent": "product_recommendation"}
                for _ in range(3)
            ),
        ]
        per = MetricsCalculator(results).per_intent_accuracy()
        assert per["product_recommendation"] == pytest.approx(0.9)
        assert per["order_query"] == pytest.approx(0.7)

    def test_per_intent_excludes_rejection_cases(self):
        # expected_intent=None should be excluded
        results = [
            {"is_correct": True, "expected_intent": None, "predicted_intent": None,
             "should_reject": True},
            {"is_correct": True, "expected_intent": "a", "predicted_intent": "a"},
        ]
        per = MetricsCalculator(results).per_intent_accuracy()
        assert None not in per
        assert per == {"a": 1.0}


# ----------------------------------------------------------------------
# Rejection metrics
# ----------------------------------------------------------------------
class TestRejectionMetrics:
    def test_rejection_metrics_basic(self):
        # 5 correctly rejected, 2 wrongly rejected
        # Should-reject cases: 5 (all correctly rejected)
        # Should-not-reject cases: 100 (2 wrongly rejected)
        results = [
            # 5 correctly rejected
            *(
                {"should_reject": True, "predicted_intent": None}
                for _ in range(5)
            ),
            # 2 wrongly rejected (should have recognized an intent)
            *(
                {"should_reject": False, "predicted_intent": None}
                for _ in range(2)
            ),
            # 98 correctly recognized
            *(
                {"should_reject": False, "predicted_intent": "a",
                 "expected_intent": "a"}
                for _ in range(98)
            ),
        ]
        m = MetricsCalculator(results).rejection_metrics()
        # rejection_precision = 5 / 7
        assert m["拒识准确率"] == pytest.approx(5 / 7)
        # false_rejection_rate = 2 / 100
        assert m["误拒率"] == pytest.approx(2 / 100)
        # missed_rejection_rate = 0 / 5
        assert m["漏拒率"] == pytest.approx(0.0)

    def test_rejection_metrics_missed(self):
        # Should reject 5 but only rejected 3 -> 2 missed
        results = [
            {"should_reject": True, "predicted_intent": None},   # correct
            {"should_reject": True, "predicted_intent": None},   # correct
            {"should_reject": True, "predicted_intent": None},   # correct
            {"should_reject": True, "predicted_intent": "a"},    # missed
            {"should_reject": True, "predicted_intent": "a"},    # missed
        ]
        m = MetricsCalculator(results).rejection_metrics()
        assert m["漏拒率"] == pytest.approx(2 / 5)
        assert m["拒识准确率"] == pytest.approx(1.0)

    def test_rejection_metrics_empty(self):
        m = MetricsCalculator([]).rejection_metrics()
        assert m["拒识准确率"] == 0.0
        assert m["误拒率"] == 0.0
        assert m["漏拒率"] == 0.0


# ----------------------------------------------------------------------
# Clarification metrics
# ----------------------------------------------------------------------
class TestClarificationMetrics:
    def test_clarification_metrics_basic(self):
        # 3 correctly triggered (and 2 of them converged)
        # 1 incorrectly triggered (should not clarify)
        results = [
            # 3 correct: should_clarify=True, clarified=True
            {"should_clarify": True, "clarified": True, "converged": True},
            {"should_clarify": True, "clarified": True, "converged": True},
            {"should_clarify": True, "clarified": True, "converged": False},
            # 1 incorrect: should_clarify=False, clarified=True
            {"should_clarify": False, "clarified": True, "converged": False},
            # Non-clarified cases (do not count)
            {"should_clarify": True, "clarified": False},
            {"should_clarify": False, "clarified": False},
        ]
        m = MetricsCalculator(results).clarification_metrics()
        # precision = 3 / 4 (3 correct out of 4 triggered)
        assert m["澄清触发准确率"] == pytest.approx(3 / 4)
        # convergence = 2 / 4 (2 converged out of 4 clarified)
        assert m["澄清后收敛率"] == pytest.approx(2 / 4)

    def test_clarification_metrics_no_triggers(self):
        results = [
            {"should_clarify": True, "clarified": False},
            {"should_clarify": False, "clarified": False},
        ]
        m = MetricsCalculator(results).clarification_metrics()
        assert m["澄清触发准确率"] == 0.0
        assert m["澄清后收敛率"] == 0.0


# ----------------------------------------------------------------------
# Slot filling metrics
# ----------------------------------------------------------------------
class TestSlotFillingMetrics:
    def test_slot_precision_recall(self):
        # 100 cases, each with 3 expected slots and varying extraction
        results = []
        for _ in range(70):
            # 70 cases: extracted 3, all correct
            results.append({
                "extracted_slots": {"a": 1, "b": 2, "c": 3},
                "expected_slots": {"a": 1, "b": 2, "c": 3},
                "correct_slots": 3,
                "required_slots_complete": True,
            })
        for _ in range(30):
            # 30 cases: extracted 2 of 3, both correct
            results.append({
                "extracted_slots": {"a": 1, "b": 2},
                "expected_slots": {"a": 1, "b": 2, "c": 3},
                "correct_slots": 2,
                "required_slots_complete": False,
            })
        m = MetricsCalculator(results).slot_filling_metrics()
        # precision = (70*3 + 30*2) / (70*3 + 30*2) = 1.0
        assert m["槽位准确率"] == pytest.approx(1.0)
        # recall = (70*3 + 30*2) / (100*3) = 270/300 = 0.9
        assert m["槽位召回率"] == pytest.approx(0.9)
        # required completeness = 70 / 100
        assert m["必填槽位完整率"] == pytest.approx(0.7)

    def test_slot_update_accuracy(self):
        results = [
            {"slot_updates_correct": 9, "slot_updates_total": 10},
            {"slot_updates_correct": 8, "slot_updates_total": 10},
        ]
        m = MetricsCalculator(results).slot_filling_metrics()
        assert m["槽位更新准确率"] == pytest.approx(17 / 20)

    def test_constraint_accuracy(self):
        results = [
            {"correct_constraints": 75, "total_constraints": 80},
            {"correct_constraints": 30, "total_constraints": 40},
        ]
        m = MetricsCalculator(results).slot_filling_metrics()
        # 105 / 120 = 0.875
        assert m["约束识别准确率"] == pytest.approx(105 / 120)

    def test_slot_filling_empty(self):
        m = MetricsCalculator([]).slot_filling_metrics()
        assert m["槽位准确率"] == 0.0
        assert m["槽位召回率"] == 0.0
        assert m["必填槽位完整率"] == 0.0
        assert m["槽位更新准确率"] == 0.0
        assert m["约束识别准确率"] == 0.0

    def test_slot_filling_with_list_slots(self):
        # list-style slots also supported
        results = [
            {
                "extracted_slots": ["a", "b"],
                "expected_slots": ["a", "b", "c"],
                "correct_slots": 2,
            }
        ]
        m = MetricsCalculator(results).slot_filling_metrics()
        # precision = 2 / 2
        assert m["槽位准确率"] == pytest.approx(1.0)
        # recall = 2 / 3
        assert m["槽位召回率"] == pytest.approx(2 / 3)


# ----------------------------------------------------------------------
# Task success rate
# ----------------------------------------------------------------------
class TestTaskSuccessRate:
    def test_task_success_rate_with_annotation(self):
        results = [
            {"task_succeeded": True},
            {"task_succeeded": True},
            {"task_succeeded": False},
            {"task_succeeded": True},
        ]
        rate = MetricsCalculator(results).task_success_rate()
        assert rate == pytest.approx(0.75)

    def test_task_success_rate_fallback_to_top1(self):
        # When task_succeeded not annotated, fall back to Top-1
        results = [
            {"is_correct": True},
            {"is_correct": False},
        ]
        rate = MetricsCalculator(results).task_success_rate()
        assert rate == pytest.approx(0.5)


# ----------------------------------------------------------------------
# Report generation
# ----------------------------------------------------------------------
class TestReportGeneration:
    def test_report_includes_all_metric_categories(self):
        results = [
            {
                "is_correct": True,
                "expected_intent": "a",
                "predicted_intent": "a",
                "topk_predictions": ["a"],
                "should_reject": False,
                "should_clarify": False,
                "clarified": False,
                "extracted_slots": {"a": 1},
                "expected_slots": {"a": 1},
                "correct_slots": 1,
                "required_slots_complete": True,
                "task_succeeded": True,
            }
        ]
        report = MetricsCalculator(results).generate_report()
        assert "total_cases" in report
        assert "top1_accuracy" in report
        assert "topk_accuracy" in report
        assert "per_intent_accuracy" in report
        assert "rejection_metrics" in report
        assert "clarification_metrics" in report
        assert "slot_filling_metrics" in report
        assert "task_success_rate" in report
        # Spot-check a value
        assert report["total_cases"] == 1
        assert report["top1_accuracy"] == pytest.approx(1.0)

    def test_report_rejection_keys_are_chinese(self):
        report = MetricsCalculator([]).generate_report()
        assert set(report["rejection_metrics"].keys()) == {
            "拒识准确率", "误拒率", "漏拒率"
        }
        assert set(report["clarification_metrics"].keys()) == {
            "澄清触发准确率", "澄清后收敛率"
        }
        assert set(report["slot_filling_metrics"].keys()) == {
            "槽位准确率", "槽位召回率", "必填槽位完整率",
            "槽位更新准确率", "约束识别准确率",
        }


# ----------------------------------------------------------------------
# Test set & coverage
# ----------------------------------------------------------------------
class TestTestSet:
    def test_all_nine_scenario_types(self):
        assert len(SCENARIO_TYPES) == 9
        expected = {
            "standard", "colloquial", "missing_slot", "rejection",
            "confusing", "cross_turn", "constraint", "intent_switch",
            "multi_intent",
        }
        assert set(SCENARIO_TYPES) == expected

    def test_add_case_validates_scenario_type(self):
        ts = TestSet()
        ts.add_case({"text": "推荐手机", "scenario_type": "standard"})
        with pytest.raises(ValueError):
            ts.add_case({"text": "x", "scenario_type": "unknown_type"})
        with pytest.raises(ValueError):
            ts.add_case({"text": "x"})  # missing scenario_type

    def test_get_cases_filtered(self):
        ts = TestSet()
        ts.add_case({"text": "a", "scenario_type": "standard"})
        ts.add_case({"text": "b", "scenario_type": "colloquial"})
        ts.add_case({"text": "c", "scenario_type": "standard"})
        assert len(ts.get_cases()) == 3
        standard = ts.get_cases("standard")
        assert len(standard) == 2
        assert all(c["scenario_type"] == "standard" for c in standard)

    def test_validate_coverage_complete(self):
        ts = TestSet()
        for t in SCENARIO_TYPES:
            ts.add_case({"text": "x", "scenario_type": t})
        counts = ts.validate_coverage()
        assert set(counts.keys()) == set(SCENARIO_TYPES)
        assert all(c >= 1 for c in counts.values())

    def test_validate_coverage_detects_missing(self):
        ts = TestSet()
        ts.add_case({"text": "x", "scenario_type": "standard"})
        counts = ts.validate_coverage()
        # All other types should report 0
        assert counts["standard"] == 1
        assert counts["colloquial"] == 0
        assert counts["intent_switch"] == 0

    def test_coverage_report_missing_scenarios(self):
        ts = TestSet()
        ts.add_case({"text": "x", "scenario_type": "standard"})
        report = ts.coverage_report()
        assert report["total"] == 1
        assert "colloquial" in report["missing_scenarios"]
        assert "standard" not in report["missing_scenarios"]
        assert report["percentages"]["standard"] == pytest.approx(1.0)

    def test_to_json_and_from_json_roundtrip(self, tmp_path: Path):
        ts = TestSet()
        ts.add_case({"text": "推荐手机", "scenario_type": "standard",
                     "expected_intent": "product_recommendation"})
        ts.add_case({"text": "查订单", "scenario_type": "cross_turn",
                     "expected_intent": "order_query"})
        path = tmp_path / "test_set.json"
        ts.to_json(str(path))
        assert path.exists()
        loaded = TestSet.from_json(str(path))
        assert len(loaded) == 2
        assert loaded.get_cases("standard")[0]["text"] == "推荐手机"
        assert loaded.get_cases("cross_turn")[0]["expected_intent"] == "order_query"

    def test_from_json_bare_list(self, tmp_path: Path):
        path = tmp_path / "bare.json"
        path.write_text(
            json.dumps([{"text": "x", "scenario_type": "standard"}]),
            encoding="utf-8",
        )
        loaded = TestSet.from_json(str(path))
        assert len(loaded) == 1

    def test_from_json_invalid(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"not_cases": []}), encoding="utf-8")
        with pytest.raises(ValueError):
            TestSet.from_json(str(path))


# ----------------------------------------------------------------------
# Test runner end-to-end
# ----------------------------------------------------------------------
class TestTestRunner:
    def test_run_single_basic(self):
        def recognize(text, case=None):
            return _make_result(intent="product_recommendation", slots={"category": "手机"})

        runner = TestRunner(recognize)
        case = {
            "case_id": "c1",
            "text": "推荐手机",
            "scenario_type": "standard",
            "expected_intent": "product_recommendation",
            "expected_slots": {"category": "手机"},
        }
        result = runner.run_single(case)
        assert result["case_id"] == "c1"
        assert result["predicted_intent"] == "product_recommendation"
        assert result["expected_intent"] == "product_recommendation"
        assert result["is_correct"] is True
        assert result["correct_slots"] == 1
        assert result["task_succeeded"] is True

    def test_run_single_rejection(self):
        def recognize(text, case=None):
            # System rejects unsupported input
            return IntentRecognitionResult(intent=None, rejection_reason="unsupported")

        runner = TestRunner(recognize)
        case = {
            "case_id": "c2",
            "text": "生成动画",
            "scenario_type": "rejection",
            "expected_intent": None,
            "should_reject": True,
        }
        result = runner.run_single(case)
        assert result["predicted_intent"] is None
        assert result["correctly_rejected"] is True
        assert result["wrongly_rejected"] is False
        assert result["is_correct"] is True

    def test_run_single_wrong_rejection(self):
        def recognize(text, case=None):
            # System wrongly rejects a valid input
            return IntentRecognitionResult(intent=None, rejection_reason="unsupported")

        runner = TestRunner(recognize)
        case = {
            "case_id": "c3",
            "text": "推荐手机",
            "scenario_type": "rejection",
            "expected_intent": "product_recommendation",
            "should_reject": False,
        }
        result = runner.run_single(case)
        assert result["wrongly_rejected"] is True
        assert result["correctly_rejected"] is False
        assert result["is_correct"] is False

    def test_run_single_clarification(self):
        def recognize(text, case=None):
            return IntentRecognitionResult(
                intent="product_recommendation",
                need_clarification=True,
                clarification_question="您希望推荐哪类商品？",
                missing_slots=["category"],
            )

        runner = TestRunner(recognize)
        case = {
            "case_id": "c4",
            "text": "推荐",
            "scenario_type": "missing_slot",
            "expected_intent": "product_recommendation",
            "should_clarify": True,
            "converged": True,
        }
        result = runner.run_single(case)
        assert result["clarified"] is True
        assert result["correctly_clarified"] is True
        assert result["converged"] is True
        # Missing slots -> not complete
        assert result["required_slots_complete"] is False

    def test_run_end_to_end_with_metrics(self):
        def recognize(text, case=None):
            intent = case.get("expected_intent") if case else None
            if "wrong" in text:
                intent = "wrong_intent"
            return _make_result(intent=intent, slots={"category": "手机"})

        ts = TestSet()
        # 8 correct + 2 wrong -> top1 = 0.8
        for i in range(8):
            ts.add_case({
                "case_id": f"c{i}",
                "text": f"推荐手机{i}",
                "scenario_type": "standard",
                "expected_intent": "product_recommendation",
                "expected_slots": {"category": "手机"},
            })
        for i in range(2):
            ts.add_case({
                "case_id": f"w{i}",
                "text": f"wrong{i}",
                "scenario_type": "standard",
                "expected_intent": "product_recommendation",
                "expected_slots": {"category": "手机"},
            })

        runner = TestRunner(recognize)
        results = runner.run(ts)
        assert len(results) == 10

        report = runner.evaluate(results)
        assert report["total_cases"] == 10
        assert report["top1_accuracy"] == pytest.approx(0.8)

    def test_generate_report_includes_coverage(self):
        def recognize(text, case=None):
            return _make_result(intent=case.get("expected_intent") if case else None)

        ts = TestSet()
        for t in SCENARIO_TYPES:
            ts.add_case({
                "case_id": f"case_{t}",
                "text": "x",
                "scenario_type": t,
                "expected_intent": "a",
                "expected_slots": {},
            })

        runner = TestRunner(recognize)
        report = runner.generate_report(ts)
        assert "metrics" in report
        assert "coverage" in report
        assert "results" in report
        assert len(report["results"]) == 9
        assert report["coverage"]["total"] == 9
        assert report["coverage"]["missing_scenarios"] == []

    def test_run_single_with_simple_callable(self):
        # Callable that does NOT accept case= kwarg
        def recognize(text):
            return _make_result(intent="a")

        runner = TestRunner(recognize)
        case = {
            "case_id": "c1",
            "text": "x",
            "scenario_type": "standard",
            "expected_intent": "a",
        }
        result = runner.run_single(case)
        assert result["predicted_intent"] == "a"
        assert result["is_correct"] is True


# ----------------------------------------------------------------------
# D29: Extended evaluation metrics (tasks 10.8 - 10.12)
# ----------------------------------------------------------------------
class TestD29ConfusionMatrix:
    """Task 10.8: N x N confusion matrix."""

    def test_confusion_matrix_basic(self):
        results = [
            {"expected_intent": "a", "predicted_intent": "a"},
            {"expected_intent": "a", "predicted_intent": "a"},
            {"expected_intent": "a", "predicted_intent": "b"},
            {"expected_intent": "b", "predicted_intent": "b"},
            {"expected_intent": "b", "predicted_intent": "a"},
        ]
        m = MetricsCalculator(results).compute_confusion_matrix()
        assert m["a"]["a"] == 2
        assert m["a"]["b"] == 1
        assert m["b"]["b"] == 1
        assert m["b"]["a"] == 1

    def test_confusion_matrix_uses_reject_label(self):
        results = [
            {"expected_intent": None, "predicted_intent": None},  # correct reject
            {"expected_intent": "a", "predicted_intent": None},   # wrong reject
        ]
        m = MetricsCalculator(results).compute_confusion_matrix()
        assert "__reject__" in m
        assert m["__reject__"]["__reject__"] == 1
        assert m["a"]["__reject__"] == 1

    def test_confusion_matrix_empty(self):
        m = MetricsCalculator([]).compute_confusion_matrix()
        assert m == {}


class TestD29ExtendedMetrics:
    """Task 10.9: hard sample, convergence, slot recall/completeness, constraint id, state update."""

    def test_hard_sample_accuracy(self):
        results = [
            {"is_hard": True, "is_correct": True},
            {"is_hard": True, "is_correct": False},
            {"is_hard": True, "is_correct": True},
            {"is_hard": False, "is_correct": False},  # excluded
        ]
        acc = MetricsCalculator(results).compute_hard_sample_accuracy()
        # 2 correct of 3 hard
        assert acc == pytest.approx(2 / 3)

    def test_hard_sample_accuracy_fallback_to_top1(self):
        # No hard samples -> fallback to Top-1
        results = [
            {"is_correct": True},
            {"is_correct": False},
        ]
        acc = MetricsCalculator(results).compute_hard_sample_accuracy()
        assert acc == pytest.approx(0.5)

    def test_clarification_convergence_rate_first_round(self):
        results = [
            {"clarified": True, "converged": True, "clarification_rounds": 1},
            {"clarified": True, "converged": True, "clarification_rounds": 2},  # not first-round
            {"clarified": True, "converged": False, "clarification_rounds": 1},
        ]
        rate = MetricsCalculator(results).compute_clarification_convergence_rate()
        # 1 first-round converged of 3 clarified
        assert rate == pytest.approx(1 / 3)

    def test_clarification_convergence_rate_default_rounds(self):
        # Cases without clarification_rounds default to 1 if converged
        results = [
            {"clarified": True, "converged": True},  # default 1 round -> first-round
            {"clarified": True, "converged": False},
        ]
        rate = MetricsCalculator(results).compute_clarification_convergence_rate()
        assert rate == pytest.approx(0.5)

    def test_slot_recall_standalone(self):
        results = [
            {"correct_slots": 3, "expected_slots": {"a": 1, "b": 2, "c": 3}},
            {"correct_slots": 2, "expected_slots": {"a": 1, "b": 2, "c": 3}},
        ]
        recall = MetricsCalculator(results).compute_slot_recall()
        # (3 + 2) / (3 + 3) = 5/6
        assert recall == pytest.approx(5 / 6)

    def test_slot_completeness(self):
        results = [
            {"required_slots_complete": True},
            {"required_slots_complete": True},
            {"required_slots_complete": False},
        ]
        comp = MetricsCalculator(results).compute_slot_completeness()
        assert comp == pytest.approx(2 / 3)

    def test_constraint_identification_rate(self):
        results = [
            {"correct_constraints": 4, "total_constraints": 5},
            {"correct_constraints": 3, "total_constraints": 5},
        ]
        rate = MetricsCalculator(results).compute_constraint_identification_rate()
        assert rate == pytest.approx(7 / 10)

    def test_constraint_identification_rate_no_constraints(self):
        # No constraints annotated -> 1.0 (nothing to miss)
        results = [{"is_correct": True}]
        rate = MetricsCalculator(results).compute_constraint_identification_rate()
        assert rate == 1.0

    def test_state_update_accuracy(self):
        results = [
            {"slot_updates_correct": 9, "slot_updates_total": 10},
            {"slot_updates_correct": 7, "slot_updates_total": 10},
        ]
        acc = MetricsCalculator(results).compute_state_update_accuracy()
        assert acc == pytest.approx(16 / 20)

    def test_state_update_accuracy_no_updates(self):
        results = [{"is_correct": True}]
        acc = MetricsCalculator(results).compute_state_update_accuracy()
        assert acc == 1.0


class TestD29RejectionAccuracy:
    """Task 10.10: false-reject / missed-reject statistics (English keys)."""

    def test_rejection_accuracy_english_keys(self):
        results = [
            {"should_reject": True, "predicted_intent": None},   # correct reject
            {"should_reject": True, "predicted_intent": None},   # correct reject
            {"should_reject": True, "predicted_intent": "a"},    # missed reject
            {"should_reject": False, "predicted_intent": None},  # false reject
            {"should_reject": False, "predicted_intent": "a", "expected_intent": "a"},
        ]
        m = MetricsCalculator(results).compute_rejection_accuracy()
        assert set(m.keys()) == {"rejection_precision", "false_reject_rate", "missed_reject_rate"}
        # rejection_precision = 2 / 3 (2 correct rejects out of 3 total rejects)
        assert m["rejection_precision"] == pytest.approx(2 / 3)
        # false_reject_rate = 1 / 2 (1 false reject out of 2 non-rejectable)
        assert m["false_reject_rate"] == pytest.approx(1 / 2)
        # missed_reject_rate = 1 / 3 (1 missed out of 3 rejectable)
        assert m["missed_reject_rate"] == pytest.approx(1 / 3)

    def test_rejection_accuracy_matches_chinese_keys(self):
        results = [
            {"should_reject": True, "predicted_intent": None},
            {"should_reject": False, "predicted_intent": None},
        ]
        calc = MetricsCalculator(results)
        en = calc.compute_rejection_accuracy()
        zh = calc.rejection_metrics()
        assert en["rejection_precision"] == zh["拒识准确率"]
        assert en["false_reject_rate"] == zh["误拒率"]
        assert en["missed_reject_rate"] == zh["漏拒率"]


class TestD29OnlineFeedback:
    """Task 10.11: online feedback loop via import_online_samples."""

    def test_import_online_samples_basic(self):
        ts = TestSet()
        samples = [
            {"text": "推荐手机", "expected_intent": "product_recommendation"},
            {"text": "查订单", "expected_intent": "order_query"},
        ]
        count = ts.import_online_samples(samples)
        assert count == 2
        all_cases = ts.get_cases()
        assert len(all_cases) == 2
        # tag is written into case_id prefix and source field
        assert all(c.get("source") == "online" for c in all_cases)
        assert all(c["case_id"].startswith("online_") for c in all_cases)

    def test_import_online_samples_with_default_intent(self):
        ts = TestSet()
        samples = [
            {"text": "推荐手机"},  # no expected_intent
            {"text": "查订单", "expected_intent": "order_query"},  # overrides default
        ]
        count = ts.import_online_samples(samples, expected_intent="product_recommendation")
        assert count == 2
        cases = ts.get_cases()
        # First case gets the default intent
        assert cases[0]["expected_intent"] == "product_recommendation"
        # Second case keeps its own intent
        assert cases[1]["expected_intent"] == "order_query"

    def test_import_online_samples_custom_scenario_and_tag(self):
        ts = TestSet()
        samples = [{"text": "x"}]
        ts.import_online_samples(samples, scenario_type="colloquial", tag="prod_v2")
        case = ts.get_cases()[0]
        assert case["scenario_type"] == "colloquial"
        # tag becomes the case_id prefix
        assert case["case_id"].startswith("prod_v2_")

    def test_collect_online_failures_by_result(self):
        def recognize(_text, case=None):
            return _make_result(intent=case.get("expected_intent") if case else None)

        runner = TestRunner(recognize)
        cases = [
            {"case_id": "c1", "text": "推荐手机", "scenario_type": "standard",
             "expected_intent": "product_recommendation"},
            {"case_id": "c2", "text": "查订单", "scenario_type": "standard",
             "expected_intent": "order_query"},
        ]
        # Mark c2 as a failure (wrong prediction)
        results = [
            {"case_id": "c1", "is_correct": True, "predicted_intent": "product_recommendation"},
            {"case_id": "c2", "is_correct": False, "predicted_intent": "wrong"},
        ]
        failures = runner.collect_online_failures(cases, results=results)
        # Only c2 should be collected
        assert len(failures) == 1
        # output stores original_case_id, not case_id
        assert failures[0]["original_case_id"] == "c2"
        assert failures[0]["source"] == "online_failure"

    def test_collect_online_failures_by_signal(self):
        def recognize(_text, _case=None):
            return _make_result(intent="a")

        runner = TestRunner(recognize)
        cases = [
            {"case_id": "c1", "text": "推荐手机", "scenario_type": "standard"},
            {"case_id": "c2", "text": "你理解错了", "scenario_type": "standard"},
        ]
        failures = runner.collect_online_failures(
            cases, failure_signals=["你理解错了"],
        )
        # c2 contains the failure signal
        assert len(failures) == 1
        assert failures[0]["original_case_id"] == "c2"


class TestD30EvolutionStage:
    """Task 10.12: assess_evolution_stage() returns correct stage."""

    def _disable_d26_d31(self, cfg):
        """Turn off all D26-D31 features to isolate earlier stages."""
        cfg.evidence.enable_grading = False
        cfg.arbitration.enable_five_factor = False
        cfg.boundary.enable_sub_tasks = False
        cfg.protocol.enable_structured_output = False
        return cfg

    def test_v0_single_deep_llm_only(self):
        from intent_recognition import IntentRecognitionConfig
        cfg = IntentRecognitionConfig()
        cfg.enable_code_layer = False
        cfg.enable_lightweight_llm = False
        cfg.enable_deep_llm = True
        runner = TestRunner(lambda text: _make_result())
        stage = runner.assess_evolution_stage(cfg)
        assert stage == "v0"

    def test_v1_stage1_l2_only_no_code_layer(self):
        from intent_recognition import IntentRecognitionConfig
        cfg = IntentRecognitionConfig()
        cfg.enable_code_layer = False
        cfg.enable_lightweight_llm = True
        cfg.enable_deep_llm = True
        runner = TestRunner(lambda text: _make_result())
        stage = runner.assess_evolution_stage(cfg)
        assert stage == "v1-stage1"

    def test_v1_stage2_three_layer_waterfall(self):
        from intent_recognition import IntentRecognitionConfig
        cfg = IntentRecognitionConfig()
        cfg.enable_code_layer = True
        cfg.enable_lightweight_llm = True
        cfg.enable_deep_llm = True
        # D17-D24 all OFF, D26-D31 all OFF
        self._disable_d26_d31(cfg)
        runner = TestRunner(lambda text: _make_result())
        stage = runner.assess_evolution_stage(cfg)
        assert stage == "v1-stage2"

    def test_v1_stage3_advanced_mechanisms_on(self):
        from intent_recognition import IntentRecognitionConfig
        cfg = IntentRecognitionConfig()
        cfg.enable_code_layer = True
        cfg.enable_lightweight_llm = True
        cfg.enable_deep_llm = True
        cfg.retrieval.enable = True  # turn on a D17-D24 mechanism
        # D26-D31 must be OFF to stay at v1-stage3
        self._disable_d26_d31(cfg)
        runner = TestRunner(lambda text: _make_result())
        stage = runner.assess_evolution_stage(cfg)
        assert stage == "v1-stage3"

    def test_v2_interview_insights_on(self):
        from intent_recognition import IntentRecognitionConfig
        cfg = IntentRecognitionConfig()
        cfg.enable_code_layer = True
        cfg.enable_lightweight_llm = True
        cfg.enable_deep_llm = True
        cfg.retrieval.enable = True
        cfg.evidence.enable_grading = True  # D26 mechanism
        runner = TestRunner(lambda text: _make_result())
        stage = runner.assess_evolution_stage(cfg)
        assert stage == "v2"
