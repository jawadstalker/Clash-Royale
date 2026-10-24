import unittest

from src.baselines import FrequencyBaseline, TransitionBaseline, top_k


class BaselineTests(unittest.TestCase):
    def test_frequency_baseline_ranks_common_targets(self):
        model = FrequencyBaseline().fit([
            ["a", "x", "x"],
            ["b", "x", "y"],
        ])
        predictions = top_k(model.predict_proba(["unknown"]), 2)
        self.assertEqual(predictions[0][0], "x")

    def test_transition_baseline_learns_last_card_transition(self):
        model = TransitionBaseline().fit([
            ["giant", "fireball", "knight"],
            ["giant", "fireball", "archers"],
            ["zap", "knight", "giant"],
        ])
        predictions = top_k(model.predict_proba(["giant"]), 2)
        self.assertEqual(predictions[0][0], "fireball")

    def test_unseen_transition_falls_back_to_global_distribution(self):
        model = TransitionBaseline().fit([["a", "b", "b"], ["c", "b", "a"]])
        self.assertEqual(model.predict_proba(["not-in-training"]), model.global_probabilities)

    def test_top_k_requires_positive_k(self):
        with self.assertRaises(ValueError):
            top_k({"a": 1.0}, 0)


if __name__ == "__main__":
    unittest.main()
