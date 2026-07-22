"""Tests for D24: fine-tuning integration (training data export + comparison)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from intent_recognition import IntentRecognitionResult, RecognitionSource
from intent_recognition.evaluation import TestRunner, TestSet
from intent_recognition.storage import MemoryIntentHistoryStore
from intent_recognition.training_data_exporter import TrainingDataExporter


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _make_result(
    *,
    intent: str | None = "product_recommendation",
    confidence: float = 0.9,
    slots: dict[str, Any] | None = None,
    layer_reached: int = 1,
    source: RecognitionSource = RecognitionSource.CODE_LAYER,
) -> IntentRecognitionResult:
    return IntentRecognitionResult(
        intent=intent,
        confidence=confidence,
        slots=slots or {},
        source=source,
        layer_reached=layer_reached,
    )


def _seed_history(
    store: MemoryIntentHistoryStore,
    session_id: str,
    results: list[IntentRecognitionResult],
    start_turn: int = 1,
) -> None:
    for i, result in enumerate(results):
        store.add(session_id, result, start_turn + i)


def _make_test_set(n_correct: int = 5, n_wrong: int = 0) -> TestSet:
    """Build a small test set with the given correct/wrong counts."""
    ts = TestSet()
    for i in range(n_correct):
        ts.add_case({
            "case_id": f"c{i}",
            "text": f"推荐手机{i}",
            "scenario_type": "standard",
            "expected_intent": "product_recommendation",
            "expected_slots": {},
        })
    for i in range(n_wrong):
        ts.add_case({
            "case_id": f"w{i}",
            "text": f"wrong{i}",
            "scenario_type": "standard",
            "expected_intent": "product_recommendation",
            "expected_slots": {},
        })
    return ts


def _make_runner(
    *,
    correct_intent: str = "product_recommendation",
    wrong_keyword: str = "wrong",
    wrong_intent: str = "wrong_intent",
) -> TestRunner:
    """Build a TestRunner whose recognizer flips intent when a keyword appears."""

    def recognize(text, case=None):
        if wrong_keyword in text:
            return _make_result(intent=wrong_intent)
        return _make_result(intent=correct_intent)

    return TestRunner(recognize)


# ----------------------------------------------------------------------
# TrainingDataExporter
# ----------------------------------------------------------------------
class TestTrainingDataExporter:
    def test_export_writes_jsonl_format(self, tmp_path: Path):
        store = MemoryIntentHistoryStore()
        _seed_history(store, "s1", [
            _make_result(intent="product_recommendation", slots={"category": "手机"}),
            _make_result(intent="order_query"),
        ])
        exporter = TrainingDataExporter(
            store, system_prompt_template="You are an intent recognizer."
        )

        out = tmp_path / "train.jsonl"
        count = exporter.export(str(out), session_ids=["s1"])

        assert count == 2
        assert out.exists()
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        for line in lines:
            record = json.loads(line)
            assert "messages" in record
            messages = record["messages"]
            # system + assistant turns are always present.
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are an intent recognizer."
            assert messages[-1]["role"] == "assistant"
            # Assistant content is valid JSON of the result.
            assistant_payload = json.loads(messages[-1]["content"])
            assert "intent" in assistant_payload
            assert "source" in assistant_payload

    def test_export_with_session_ids_filter(self, tmp_path: Path):
        store = MemoryIntentHistoryStore()
        _seed_history(store, "s1", [_make_result(intent="a")])
        _seed_history(store, "s2", [_make_result(intent="b"), _make_result(intent="b")])
        _seed_history(store, "s3", [_make_result(intent="c")])
        exporter = TrainingDataExporter(store)

        out = tmp_path / "filtered.jsonl"
        # Export only s2.
        count = exporter.export(str(out), session_ids=["s2"])

        assert count == 2
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            payload = json.loads(json.loads(line)["messages"][-1]["content"])
            assert payload["intent"] == "b"

    def test_export_without_session_ids_requires_list_sessions(self, tmp_path: Path):
        # MemoryIntentHistoryStore does NOT implement list_sessions(), so
        # exporting without session_ids must raise NotImplementedError.
        store = MemoryIntentHistoryStore()
        _seed_history(store, "s1", [_make_result(intent="a")])
        exporter = TrainingDataExporter(store)

        out = tmp_path / "all.jsonl"
        with pytest.raises(NotImplementedError):
            exporter.export(str(out), session_ids=None)

    def test_export_with_list_sessions_uses_it(self, tmp_path: Path):
        """When the store implements list_sessions(), it is used for export-all."""

        class ListableStore(MemoryIntentHistoryStore):
            def list_sessions(self) -> list[str]:
                return list(self._history.keys())

        store = ListableStore()
        _seed_history(store, "s1", [_make_result(intent="a")])
        _seed_history(store, "s2", [_make_result(intent="b")])
        exporter = TrainingDataExporter(store, system_prompt_template="sys")

        out = tmp_path / "all.jsonl"
        count = exporter.export(str(out), session_ids=None)

        assert count == 2
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        intents = sorted(
            json.loads(json.loads(line)["messages"][-1]["content"])["intent"]
            for line in lines
        )
        assert intents == ["a", "b"]

    def test_export_preserves_raw_text_when_stashed(self, tmp_path: Path):
        """When the result carries a ``_raw_text`` slot, it becomes the user turn."""
        store = MemoryIntentHistoryStore()
        _seed_history(store, "s1", [
            _make_result(
                intent="product_recommendation",
                slots={"_raw_text": "推荐手机", "category": "手机"},
            ),
        ])
        exporter = TrainingDataExporter(store, system_prompt_template="sys")

        out = tmp_path / "raw.jsonl"
        count = exporter.export(str(out), session_ids=["s1"])

        assert count == 1
        record = json.loads(out.read_text(encoding="utf-8").strip())
        messages = record["messages"]
        # system, user, assistant.
        assert len(messages) == 3
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "推荐手机"


# ----------------------------------------------------------------------
# TestRunner.evaluate_fine_tuned
# ----------------------------------------------------------------------
class TestEvaluateFineTuned:
    def test_evaluate_fine_tuned_returns_improvement_dict(self):
        # Baseline: 4 correct, 1 wrong -> top1 = 0.8
        # Fine-tuned: 5 correct, 0 wrong -> top1 = 1.0
        ts = _make_test_set(n_correct=4, n_wrong=1)
        baseline = _make_runner()  # flips to wrong_intent on "wrong"
        ft = _make_runner(wrong_keyword="__never__")  # never flips -> always correct

        report = baseline.evaluate_fine_tuned(ft, ts)

        assert set(report.keys()) == {
            "before_accuracy", "after_accuracy", "improvement", "recommendation"
        }
        assert report["before_accuracy"] == pytest.approx(0.8)
        assert report["after_accuracy"] == pytest.approx(1.0)
        assert report["improvement"] == pytest.approx(0.2)

    def test_recommendation_deploy_when_improvement_positive(self):
        ts = _make_test_set(n_correct=4, n_wrong=1)
        baseline = _make_runner()
        ft = _make_runner(wrong_keyword="__never__")

        report = baseline.evaluate_fine_tuned(ft, ts)

        assert report["improvement"] > 0
        assert report["recommendation"] == "deploy"

    def test_recommendation_rollback_when_improvement_negative(self):
        # Baseline: perfect (5/5) because wrong_keyword never matches.
        # Fine-tuned: 4/5 because it now flips a real case.
        ts = _make_test_set(n_correct=5, n_wrong=0)
        # Add one case whose text contains the wrong_keyword so the FT runner
        # flips it.
        ts.add_case({
            "case_id": "flip",
            "text": "wrong-flip",
            "scenario_type": "standard",
            "expected_intent": "product_recommendation",
            "expected_slots": {},
        })
        baseline = _make_runner(wrong_keyword="__never__")
        ft = _make_runner(wrong_keyword="wrong")

        report = baseline.evaluate_fine_tuned(ft, ts)

        assert report["improvement"] < 0
        assert report["recommendation"] == "rollback"

    def test_recommendation_rollback_when_no_improvement(self):
        # Identical runners -> identical accuracy -> improvement == 0 -> rollback.
        ts = _make_test_set(n_correct=5, n_wrong=0)
        baseline = _make_runner(wrong_keyword="__never__")
        ft = _make_runner(wrong_keyword="__never__")

        report = baseline.evaluate_fine_tuned(ft, ts)

        assert report["improvement"] == pytest.approx(0.0)
        # improvement is not strictly positive -> rollback.
        assert report["recommendation"] == "rollback"

    def test_evaluate_fine_tuned_does_not_mutate_baseline_runner(self):
        ts = _make_test_set(n_correct=3, n_wrong=0)
        baseline = _make_runner(wrong_keyword="__never__")

        # Snapshot baseline accuracy before the comparison.
        before_report = baseline.evaluate(baseline.run(ts))
        before_acc = before_report["top1_accuracy"]

        ft = _make_runner(wrong_keyword="__never__")
        baseline.evaluate_fine_tuned(ft, ts)

        # Re-run baseline; accuracy must be unchanged.
        after_report = baseline.evaluate(baseline.run(ts))
        assert after_report["top1_accuracy"] == pytest.approx(before_acc)
