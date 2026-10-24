import unittest

from src.event_engine import EventEngine


class EventEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = EventEngine(min_confirmations=2, rearm_absence_seconds=1.0)

    def test_requires_multiple_frames(self):
        self.assertEqual(
            self.engine.process_frame([{"card": "giant", "conf": 0.9, "x": 0.4}], 1.0),
            [],
        )
        events = self.engine.process_frame(
            [{"card": "giant", "conf": 0.8, "x": 0.4}], 1.3
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "card_detected")
        self.assertAlmostEqual(events[0].confidence, 0.85)

    def test_persistent_detection_emits_only_once(self):
        detections = [{"card": "giant", "conf": 0.9, "x": 0.4}]
        self.engine.process_frame(detections, 1.0)
        self.assertEqual(len(self.engine.process_frame(detections, 1.2)), 1)
        for timestamp in (1.4, 1.6, 2.0, 2.5):
            self.assertEqual(len(self.engine.process_frame(detections, timestamp)), 0)

    def test_rearms_after_absence(self):
        detection = [{"card": "giant", "conf": 0.9, "x": 0.4}]
        self.engine.process_frame(detection, 1.0)
        self.engine.process_frame(detection, 1.2)
        self.engine.process_frame([], 1.5)
        self.engine.process_frame([], 2.3)
        self.assertEqual(self.engine.process_frame(detection, 2.4), [])
        events = self.engine.process_frame(detection, 2.5)
        self.assertEqual(len(events), 1)

    def test_low_confidence_and_invalid_location_are_ignored(self):
        self.assertEqual(
            self.engine.process_frame(
                [
                    {"card": "giant", "conf": 0.2, "x": 0.4},
                    {"card": "zap", "conf": 0.9, "x": 1.5},
                ],
                1.0,
            ),
            [],
        )

    def test_reset_clears_latches(self):
        detection = [{"card": "giant", "conf": 0.9, "x": 0.4}]
        self.engine.process_frame(detection, 1.0)
        self.engine.process_frame(detection, 1.2)
        self.engine.reset()
        self.assertEqual(self.engine.process_frame(detection, 2.0), [])
        self.assertEqual(len(self.engine.process_frame(detection, 2.2)), 1)


if __name__ == "__main__":
    unittest.main()
