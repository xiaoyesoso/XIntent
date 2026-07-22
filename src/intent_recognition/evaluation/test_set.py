"""Test set management with 9 scenario types (D16).

Scenario types:
- ``standard``: standard expressions
- ``colloquial``: colloquial / informal expressions
- ``missing_slot``: required slot missing
- ``rejection``: out-of-scope, should be rejected
- ``confusing``: easily-confused intents
- ``cross_turn``: cross-turn slot supplementation
- ``constraint``: hard/soft constraint extraction
- ``intent_switch``: intent switching between turns
- ``multi_intent``: multi-intent input
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCENARIO_TYPES: tuple[str, ...] = (
    "standard",
    "colloquial",
    "missing_slot",
    "rejection",
    "confusing",
    "cross_turn",
    "constraint",
    "intent_switch",
    "multi_intent",
)


class TestSet:
    """Manage test cases covering the 9 D16 scenario types.

    Each case is a dict. The convention is:

    - ``case_id`` (str): unique identifier
    - ``text`` (str): user input
    - ``expected_intent`` (str | None): expected intent, ``None`` for rejection
    - ``expected_slots`` (dict | list | None): expected slot values
    - ``scenario_type`` (str): one of :data:`SCENARIO_TYPES`
    - ``should_reject`` (bool): whether the case expects rejection
    - ``should_clarify`` (bool): whether the case expects clarification
    - ... any additional fields the recognizer needs
    """

    # Mark as non-test class so pytest does not try to collect it.
    __test__ = False

    def __init__(self, cases: list[dict[str, Any]] | None = None) -> None:
        self._cases: list[dict[str, Any]] = list(cases or [])

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add_case(self, case: dict[str, Any]) -> None:
        """Add a single test case.

        Raises ``ValueError`` if ``scenario_type`` is missing or not one of
        the 9 supported types.
        """
        scenario = case.get("scenario_type")
        if not scenario:
            raise ValueError("case must include 'scenario_type'")
        if scenario not in SCENARIO_TYPES:
            raise ValueError(
                f"unknown scenario_type {scenario!r}; "
                f"must be one of {SCENARIO_TYPES}"
            )
        self._cases.append(case)

    def add_cases(self, cases: list[dict[str, Any]]) -> None:
        """Add multiple test cases."""
        for case in cases:
            self.add_case(case)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------
    def get_cases(self, scenario_type: str | None = None) -> list[dict[str, Any]]:
        """Return all cases, optionally filtered by scenario type."""
        if scenario_type is None:
            return list(self._cases)
        return [c for c in self._cases if c.get("scenario_type") == scenario_type]

    @property
    def cases(self) -> list[dict[str, Any]]:
        """All cases (shallow copy)."""
        return list(self._cases)

    def __len__(self) -> int:
        return len(self._cases)

    # ------------------------------------------------------------------
    # Coverage (D16)
    # ------------------------------------------------------------------
    def validate_coverage(self) -> dict[str, int]:
        """Return a mapping ``scenario_type -> case_count`` for all 9 types.

        Missing scenarios have count ``0`` so callers can detect gaps.
        """
        counts = {t: 0 for t in SCENARIO_TYPES}
        for c in self._cases:
            t = c.get("scenario_type")
            if t in counts:
                counts[t] += 1
        return counts

    def coverage_report(self) -> dict[str, Any]:
        """Return counts plus percentages and a list of missing scenarios."""
        counts = self.validate_coverage()
        total = sum(counts.values())
        percentages = {
            t: (_safe_div(c, total) if total else 0.0)
            for t, c in counts.items()
        }
        missing = [t for t, c in counts.items() if c == 0]
        return {
            "total": total,
            "counts": counts,
            "percentages": percentages,
            "missing_scenarios": missing,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_json(self, path: str) -> None:
        """Save the test set to a JSON file."""
        data = {"cases": self._cases}
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_json(cls, path: str) -> "TestSet":
        """Load a test set from a JSON file.

        Accepts both ``{"cases": [...]}`` and a bare ``[...]`` layout.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "cases" in raw:
            cases = raw["cases"]
        elif isinstance(raw, list):
            cases = raw
        else:
            raise ValueError("invalid test-set JSON; expected list or {'cases': [...]}")
        return cls(cases=[dict(c) for c in cases])

    # ------------------------------------------------------------------
    # D29: Online feedback loop
    # ------------------------------------------------------------------
    def import_online_samples(
        self,
        samples: list[dict[str, Any]],
        expected_intent: str | None = None,
        scenario_type: str = "standard",
        tag: str = "online",
    ) -> int:
        """Import online samples into the test set (D29 task 5.8).

        Online samples are typically collected from production failure
        cases (e.g. via D11 implicit-failure-signal detection) and fed back
        into the offline test set so regressions can be caught.

        Each sample is normalized to the test-case schema:
        - ``text`` is taken from the sample's ``text`` (or ``input``) field
        - ``expected_intent`` is taken from the sample (or the
          ``expected_intent`` argument when the sample lacks one)
        - ``scenario_type`` defaults to ``"standard"``; pass the desired
          scenario to bucket the online samples differently
        - ``tag`` is written into ``case_id`` so online-origin samples can
          be filtered out later (``f"{tag}_{idx}"``)

        Args:
            samples: List of dicts (each must contain ``text`` or ``input``).
            expected_intent: Default expected intent when sample lacks one.
            scenario_type: Scenario label for the imported cases.
            tag: Prefix for the generated ``case_id`` values.

        Returns:
            Number of samples actually imported (samples without text are
            skipped).
        """
        if scenario_type not in SCENARIO_TYPES:
            raise ValueError(
                f"unknown scenario_type {scenario_type!r}; "
                f"must be one of {SCENARIO_TYPES}"
            )
        imported = 0
        for idx, sample in enumerate(samples):
            if not isinstance(sample, dict):
                continue
            text = sample.get("text") or sample.get("input")
            if not text:
                continue
            case = {
                "case_id": f"{tag}_{idx}",
                "text": text,
                "expected_intent": sample.get("expected_intent", expected_intent),
                "scenario_type": scenario_type,
                "should_reject": bool(sample.get("should_reject", False)),
                "should_clarify": bool(sample.get("should_clarify", False)),
                "source": "online",
                "raw": sample,
            }
            # Preserve any extra annotation fields from the sample
            for k, v in sample.items():
                if k not in case:
                    case[k] = v
            self._cases.append(case)
            imported += 1
        return imported


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den
