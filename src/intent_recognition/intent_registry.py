"""Intent registry for managing intent definitions (D12, D13, D19)."""

from __future__ import annotations

from .models import IntentDefinition, OverlapReport, SlotDefinition


class IntentRegistry:
    """Registry for intent definitions.

    Supports flat and hierarchical (parent-child) intent structures.
    D19: Adds orthogonality governance (overlap detection, split, merge_with_param).
    """

    # D19: split threshold (similarity above this suggests splitting)
    _SPLIT_THRESHOLD = 0.85
    # D19: merge threshold (similarity in [overlap_threshold, split_threshold] suggests merge)
    _MERGE_DEFAULT_THRESHOLD = 0.7
    # D19: example overlap threshold used by split_intent to pick overlapping examples
    _EXAMPLE_OVERLAP_THRESHOLD = 0.5

    def __init__(self) -> None:
        self._intents: dict[str, IntentDefinition] = {}
        self._children: dict[str, list[str]] = {}  # parent -> [child names]

    def register(self, definition: IntentDefinition) -> None:
        """Register an intent definition."""
        self._intents[definition.name] = definition
        if definition.parent_intent:
            if definition.parent_intent not in self._children:
                self._children[definition.parent_intent] = []
            self._children[definition.parent_intent].append(definition.name)

    def get(self, name: str) -> IntentDefinition | None:
        """Look up an intent by name."""
        return self._intents.get(name)

    def list_all(self) -> list[IntentDefinition]:
        """List all registered intents."""
        return list(self._intents.values())

    def list_names(self) -> list[str]:
        """List all intent names."""
        return list(self._intents.keys())

    def list_root_intents(self) -> list[IntentDefinition]:
        """List top-level intents (no parent)."""
        return [d for d in self._intents.values() if d.parent_intent is None]

    def list_children(self, parent: str) -> list[IntentDefinition]:
        """List child intents of a parent."""
        child_names = self._children.get(parent, [])
        return [self._intents[n] for n in child_names if n in self._intents]

    def has_intent(self, name: str) -> bool:
        """Check if an intent is registered."""
        return name in self._intents

    def build_prompt_description(self, intents: list[IntentDefinition] | None = None) -> str:
        """Build intent descriptions for LLM prompt (D12)."""
        target = intents if intents else self.list_all()
        lines: list[str] = []
        for d in target:
            lines.append(f"- {d.name}: {d.description}")
            if d.positive_examples:
                lines.append(f"  正例: {'; '.join(d.positive_examples)}")
            if d.negative_examples:
                lines.append(f"  反例: {'; '.join(d.negative_examples)}")
        return "\n".join(lines)

    def build_slot_description(self, intent_name: str) -> str:
        """Build slot descriptions for a specific intent."""
        d = self._intents.get(intent_name)
        if not d or not d.slots:
            return ""
        lines: list[str] = []
        for s in d.slots:
            req = "必选" if s.required else "可选"
            default = f", 默认={s.default}" if s.default is not None else ""
            lines.append(f"  - {s.name} ({req}{default}): {s.description}")
        return "\n".join(lines)

    def get_required_slots(self, intent_name: str) -> list[str]:
        """Get required slot names for an intent."""
        d = self._intents.get(intent_name)
        if not d:
            return []
        return [s.name for s in d.slots if s.required]

    def get_all_slots(self, intent_name: str) -> list[str]:
        """Get all slot names for an intent."""
        d = self._intents.get(intent_name)
        if not d:
            return []
        return [s.name for s in d.slots]

    # ------------------------------------------------------------------
    # D19: Orthogonality governance
    # ------------------------------------------------------------------

    def detect_overlap(self, threshold: float | None = None) -> list[OverlapReport]:
        """Detect pairwise overlap between all registered intents (D19).

        Computes Jaccard token overlap between each pair of intents based on
        ``description + positive_examples`` text. Returns OverlapReport for any
        pair whose similarity is >= threshold.

        Args:
            threshold: Similarity threshold. Defaults to ``_MERGE_DEFAULT_THRESHOLD``
                (0.7) when not provided.

        Returns:
            List of OverlapReport sorted by descending similarity.
        """
        if threshold is None:
            threshold = self._MERGE_DEFAULT_THRESHOLD
        names = self.list_names()
        reports: list[OverlapReport] = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                sim = self._compute_similarity(
                    self._intent_text(a), self._intent_text(b)
                )
                if sim >= threshold:
                    suggestion = (
                        "split" if sim > self._SPLIT_THRESHOLD else "merge_with_param"
                    )
                    reports.append(OverlapReport(
                        intent_a=a,
                        intent_b=b,
                        similarity=sim,
                        suggestion=suggestion,
                    ))
        reports.sort(key=lambda r: r.similarity, reverse=True)
        return reports

    def split_intent(
        self,
        intent_a: str,
        intent_b: str,
        new_intent_name: str,
        new_intent_description: str = "",
    ) -> IntentDefinition:
        """Split overlapping content of two intents into a new intent (D19).

        The new intent's positive_examples are examples from ``intent_a`` that
        are similar (token overlap > ``_EXAMPLE_OVERLAP_THRESHOLD``) to any text
        of ``intent_b`` (and vice versa). Those overlapping examples are removed
        from the original intents. The new intent is registered.

        Args:
            intent_a: First intent name.
            intent_b: Second intent name.
            new_intent_name: Name for the new split intent.
            new_intent_description: Optional description (defaults to a generated one).

        Returns:
            The newly created and registered IntentDefinition.
        """
        a = self._intents.get(intent_a)
        b = self._intents.get(intent_b)
        if a is None or b is None:
            raise ValueError(
                f"split_intent requires both intents to be registered: "
                f"{intent_a!r}, {intent_b!r}"
            )
        # Build target text per opposite intent
        b_text = self._intent_text(intent_b)
        a_text = self._intent_text(intent_a)
        b_tokens = self._tokenize(b_text)
        a_tokens = self._tokenize(a_text)

        # Collect overlapping examples from each side
        overlap_from_a: list[str] = []
        keep_from_a: list[str] = []
        for ex in a.positive_examples:
            ex_tokens = self._tokenize(ex)
            sim = self._jaccard(ex_tokens, b_tokens)
            if sim > self._EXAMPLE_OVERLAP_THRESHOLD:
                overlap_from_a.append(ex)
            else:
                keep_from_a.append(ex)

        overlap_from_b: list[str] = []
        keep_from_b: list[str] = []
        for ex in b.positive_examples:
            ex_tokens = self._tokenize(ex)
            sim = self._jaccard(ex_tokens, a_tokens)
            if sim > self._EXAMPLE_OVERLAP_THRESHOLD:
                overlap_from_b.append(ex)
            else:
                keep_from_b.append(ex)

        # Deduplicate while preserving order
        merged_examples: list[str] = []
        seen: set[str] = set()
        for ex in overlap_from_a + overlap_from_b:
            if ex not in seen:
                merged_examples.append(ex)
                seen.add(ex)

        description = new_intent_description or (
            f"Split overlap of {intent_a} and {intent_b}"
        )
        new_intent = IntentDefinition(
            name=new_intent_name,
            description=description,
            positive_examples=merged_examples,
        )
        # Strip overlapping examples from originals (in-place mutation)
        a.positive_examples = keep_from_a
        b.positive_examples = keep_from_b
        self.register(new_intent)
        return new_intent

    def merge_with_param(
        self,
        intents: list[str],
        new_intent_name: str,
        dispatch_param: str,
        new_intent_description: str = "",
    ) -> IntentDefinition:
        """Merge multiple similar intents into one with a dispatch param (D19).

        The new intent's positive_examples = union of all merged intents'
        examples. The new intent's slots = union of all slots plus a new
        SlotDefinition for ``dispatch_param``. Original intents are removed
        from the registry. The new intent is registered.

        Args:
            intents: List of intent names to merge.
            new_intent_name: Name for the merged intent.
            dispatch_param: Slot name used to dispatch to the original intent.
            new_intent_description: Optional description.

        Returns:
            The newly created and registered IntentDefinition.
        """
        if not intents:
            raise ValueError("merge_with_param requires at least one intent")
        merged_examples: list[str] = []
        seen_examples: set[str] = set()
        merged_slots: list[SlotDefinition] = []
        seen_slot_names: set[str] = set()
        for name in intents:
            d = self._intents.get(name)
            if d is None:
                raise ValueError(
                    f"merge_with_param: intent {name!r} is not registered"
                )
            for ex in d.positive_examples:
                if ex not in seen_examples:
                    merged_examples.append(ex)
                    seen_examples.add(ex)
            for s in d.slots:
                if s.name not in seen_slot_names:
                    merged_slots.append(s)
                    seen_slot_names.add(s.name)
        # Add dispatch param slot if not already present
        if dispatch_param not in seen_slot_names:
            merged_slots.append(SlotDefinition(
                name=dispatch_param,
                required=False,
                description="Dispatch parameter selecting the original merged intent",
            ))
        description = new_intent_description or (
            f"Merged intent of {', '.join(intents)}"
        )
        new_intent = IntentDefinition(
            name=new_intent_name,
            description=description,
            positive_examples=merged_examples,
            slots=merged_slots,
        )
        # Remove original intents
        for name in intents:
            self._remove_intent(name)
        self.register(new_intent)
        return new_intent

    # ------------------------------------------------------------------
    # D19: Internal helpers
    # ------------------------------------------------------------------

    def _remove_intent(self, name: str) -> None:
        """Remove an intent from the registry (and parent->child index)."""
        d = self._intents.pop(name, None)
        if d is not None and d.parent_intent:
            siblings = self._children.get(d.parent_intent, [])
            if name in siblings:
                siblings.remove(name)
            if not siblings:
                self._children.pop(d.parent_intent, None)

    def _intent_text(self, name: str) -> str:
        """Build the comparison text (description + positive examples) for an intent."""
        d = self._intents.get(name)
        if d is None:
            return ""
        parts = [d.description]
        parts.extend(d.positive_examples)
        return " ".join(parts)

    @classmethod
    def _compute_similarity(cls, text_a: str, text_b: str) -> float:
        """Compute Jaccard token-overlap similarity between two texts (D19)."""
        return cls._jaccard(cls._tokenize(text_a), cls._tokenize(text_b))

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize text: char-level for CJK + word-level for English.

        Same approach as MemoryFewShotStore._tokenize.
        """
        tokens: set[str] = set()
        for part in text.split():
            tokens.add(part.lower())
            for ch in part:
                if "\u4e00" <= ch <= "\u9fff":
                    tokens.add(ch)
        return tokens

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        """Jaccard similarity between two token sets."""
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union > 0 else 0.0
