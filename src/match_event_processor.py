"""Apply explicit, confirmed card-play events to the card-cycle state.

Vision events from EventEngine are only card_detected. They must not be passed
directly to this processor: a caller needs separate evidence that the card was
played by the opponent and belongs to the current hand.
"""
from __future__ import annotations

from typing import Any

from card_cycle import CardCycleEngine, CardCycleError


class MatchEventProcessor:
    def __init__(self, cycle: CardCycleEngine) -> None:
        self.cycle = cycle
        self.processed_event_ids: set[str] = set()

    def process(self, event: dict[str, Any]) -> dict[str, Any]:
        event_id = str(event.get("event_id", "")).strip()
        event_type = event.get("event_type")
        card = event.get("card")

        if not event_id:
            raise ValueError("event_id is required")
        if event_id in self.processed_event_ids:
            return {
                "event_id": event_id,
                "status": "duplicate",
                "hand": list(self.cycle.hand),
            }
        if event_type != "card_played":
            return {
                "event_id": event_id,
                "status": "ignored_not_a_play",
                "hand": list(self.cycle.hand),
            }
        if not isinstance(card, str) or not card:
            raise ValueError("card_played events require a non-empty card")

        try:
            new_hand = self.cycle.play_card(card)
        except CardCycleError:
            raise
        self.processed_event_ids.add(event_id)
        return {
            "event_id": event_id,
            "status": "applied",
            "card": card,
            "hand": list(new_hand),
        }

    def reset(self) -> None:
        self.cycle.reset()
        self.processed_event_ids.clear()
