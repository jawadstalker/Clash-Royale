import unittest

from src.card_cycle import CardCycleEngine, CardCycleError
from src.match_event_processor import MatchEventProcessor


class MatchEventProcessorTests(unittest.TestCase):
    def setUp(self):
        cycle = CardCycleEngine(
            ["giant", "fireball", "knight", "archers", "zap", "musketeer", "log", "goblins"]
        )
        self.processor = MatchEventProcessor(cycle)

    def test_detection_event_does_not_mutate_cycle(self):
        result = self.processor.process({
            "event_id": "e1", "event_type": "card_detected", "card": "giant"
        })
        self.assertEqual(result["status"], "ignored_not_a_play")
        self.assertEqual(self.processor.cycle.hand[0], "giant")

    def test_confirmed_play_updates_cycle(self):
        result = self.processor.process({
            "event_id": "e1", "event_type": "card_played", "card": "giant"
        })
        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["hand"], ["fireball", "knight", "archers", "zap"])

    def test_duplicate_event_is_idempotent(self):
        event = {"event_id": "e1", "event_type": "card_played", "card": "giant"}
        self.processor.process(event)
        result = self.processor.process(event)
        self.assertEqual(result["status"], "duplicate")
        self.assertEqual(self.processor.cycle.played_history, ("giant",))

    def test_invalid_play_does_not_mark_event_processed(self):
        event = {"event_id": "e1", "event_type": "card_played", "card": "log"}
        with self.assertRaises(CardCycleError):
            self.processor.process(event)
        self.assertNotIn("e1", self.processor.processed_event_ids)


if __name__ == "__main__":
    unittest.main()
