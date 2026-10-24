import unittest

from src.card_cycle import CardCycleEngine, CardCycleError


class CardCycleEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = CardCycleEngine(
            ["giant", "fireball", "knight", "archers", "zap", "musketeer", "log", "goblins"]
        )

    def test_initial_hand_is_first_four_cards(self):
        self.assertEqual(
            self.engine.hand, ("giant", "fireball", "knight", "archers")
        )

    def test_played_card_moves_to_end_and_next_card_enters_hand(self):
        hand = self.engine.play_card("giant")
        self.assertEqual(hand, ("fireball", "knight", "archers", "zap"))
        self.assertEqual(self.engine.cycle_order[-1], "giant")
        self.assertEqual(self.engine.played_history, ("giant",))

    def test_playing_non_hand_card_does_not_mutate_state(self):
        before = self.engine.cycle_order
        with self.assertRaises(CardCycleError):
            self.engine.play_card("log")
        self.assertEqual(self.engine.cycle_order, before)
        self.assertEqual(self.engine.played_history, ())

    def test_unknown_slots_are_supported(self):
        engine = CardCycleEngine(["giant", None, "knight", None, "zap", None, "log", None])
        self.assertEqual(engine.hand, ("giant", None, "knight", None))
        self.assertEqual(engine.play_card("giant"), (None, "knight", None, "zap"))

    def test_reset_restores_initial_state(self):
        self.engine.play_card("giant")
        self.engine.reset()
        self.assertEqual(self.engine.hand, ("giant", "fireball", "knight", "archers"))
        self.assertEqual(self.engine.played_history, ())

    def test_deck_must_have_eight_slots_and_unique_known_cards(self):
        with self.assertRaises(ValueError):
            CardCycleEngine(["giant"])
        with self.assertRaises(ValueError):
            CardCycleEngine(["giant", "giant", None, None, None, None, None, None])


if __name__ == "__main__":
    unittest.main()
