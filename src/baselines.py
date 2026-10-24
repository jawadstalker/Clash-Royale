"""Simple, deterministic baselines for next-card prediction.

These baselines are intended to be evaluated on held-out matches before relying
on the LSTM. They use only training-match sequences when fitted.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable


def _normalize(counts: Counter, vocabulary: list[str]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {card: 0.0 for card in vocabulary}
    return {card: counts.get(card, 0) / total for card in vocabulary}


class FrequencyBaseline:
    """Predict the next card from its overall frequency in training sequences."""

    def __init__(self) -> None:
        self.probabilities: dict[str, float] = {}
        self.vocabulary: list[str] = []

    def fit(self, sequences: Iterable[list[str]]) -> "FrequencyBaseline":
        counts: Counter = Counter()
        for sequence in sequences:
            counts.update(sequence[1:])
        self.vocabulary = sorted(counts)
        self.probabilities = _normalize(counts, self.vocabulary)
        return self

    def predict_proba(self, history: list[str]) -> dict[str, float]:
        del history  # This baseline deliberately ignores context.
        return dict(self.probabilities)


class TransitionBaseline:
    """Predict from the last observed card; fall back to global frequency."""

    def __init__(self) -> None:
        self.vocabulary: list[str] = []
        self.global_probabilities: dict[str, float] = {}
        self.transitions: dict[str, dict[str, float]] = {}

    def fit(self, sequences: Iterable[list[str]]) -> "TransitionBaseline":
        global_counts: Counter = Counter()
        transition_counts: dict[str, Counter] = defaultdict(Counter)
        for sequence in sequences:
            global_counts.update(sequence[1:])
            for previous, target in zip(sequence, sequence[1:]):
                transition_counts[previous][target] += 1

        self.vocabulary = sorted(global_counts)
        self.global_probabilities = _normalize(global_counts, self.vocabulary)
        self.transitions = {
            previous: _normalize(counts, self.vocabulary)
            for previous, counts in transition_counts.items()
        }
        return self

    def predict_proba(self, history: list[str]) -> dict[str, float]:
        if history and history[-1] in self.transitions:
            return dict(self.transitions[history[-1]])
        return dict(self.global_probabilities)


def top_k(probabilities: dict[str, float], k: int = 3) -> list[tuple[str, float]]:
    """Return cards ordered by descending probability, with stable tie-breaking."""
    if k < 1:
        raise ValueError("k must be >= 1")
    return sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))[:k]
