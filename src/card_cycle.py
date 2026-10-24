"""Stateful, testable model of the 8-card Clash Royale deck cycle.

Provide the known deck in cycle order. The first four entries represent the
initial hand; entries may be None when a card is unknown. Playing a card from
the current hand moves it to the end of the cycle, bringing the next card into
the hand. This module does not infer a deck order from vision detections.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

Card = Optional[str]


class CardCycleError(ValueError):
    """Raised when a play cannot be reconciled with the known cycle."""


class CardCycleEngine:
    DECK_SIZE = 8
    HAND_SIZE = 4

    def __init__(self, deck_order: Sequence[Card]) -> None:
        if len(deck_order) != self.DECK_SIZE:
            raise ValueError("deck_order must contain exactly 8 slots")
        known = [card for card in deck_order if card is not None]
        duplicates = [card for card, count in Counter(known).items() if count > 1]
        if duplicates:
            raise ValueError(f"known cards must be unique; duplicates: {duplicates}")
        self._initial_cycle = list(deck_order)
        self.reset()

    def reset(self) -> None:
        """Restore the supplied cycle order and clear match history."""
        self._cycle = list(self._initial_cycle)
        self._played: list[str] = []

    @property
    def hand(self) -> tuple[Card, ...]:
        """Current four hand slots; None means the slot is not yet known."""
        return tuple(self._cycle[: self.HAND_SIZE])

    @property
    def next_cards(self) -> tuple[Card, ...]:
        """Next four cards in cycle order (not necessarily the opponent's plays)."""
        return tuple(self._cycle[self.HAND_SIZE :])

    @property
    def cycle_order(self) -> tuple[Card, ...]:
        return tuple(self._cycle)

    @property
    def played_history(self) -> tuple[str, ...]:
        return tuple(self._played)

    def play_card(self, card: str) -> tuple[Card, ...]:
        """Apply a confirmed card-play event and return the new hand.

        Unknown cards or cards not in the current hand are rejected rather
        than silently corrupting the cycle state.
        """
        if not card:
            raise CardCycleError("card must be a non-empty card id")
        try:
            index = self._cycle[: self.HAND_SIZE].index(card)
        except ValueError as exc:
            raise CardCycleError(
                f"{card!r} is not in the current known hand {self.hand!r}"
            ) from exc

        played = self._cycle.pop(index)
        self._cycle.append(played)
        self._played.append(card)
        return self.hand
