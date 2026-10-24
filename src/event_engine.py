"""Turn frame detections into stable, deduplicated detection events.

A detection event is not automatically a card-play event: units can remain on
screen for seconds. Callers must explicitly confirm a play before updating the
opponent card cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class DetectionEvent:
    event_id: str
    timestamp: float
    event_type: str
    card: str
    confidence: float
    x: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "t": self.timestamp,
            "event_type": self.event_type,
            "card": self.card,
            "conf": self.confidence,
            "x": self.x,
        }


@dataclass
class _Candidate:
    count: int
    last_timestamp: float
    confidence_sum: float
    x: float | None


class EventEngine:
    """Confirm repeated detections across frames and suppress persistent sightings.

    Each frame is a sequence of dictionaries using the repository's existing
    log schema: {"t": seconds, "card": name, "conf": float, "x": optional}.
    Coordinates are normalized to [0, 1] when supplied.
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        min_confirmations: int = 2,
        max_gap_seconds: float = 1.0,
        same_location_tolerance: float = 0.12,
        rearm_absence_seconds: float = 1.0,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if min_confirmations < 1:
            raise ValueError("min_confirmations must be >= 1")
        if max_gap_seconds <= 0 or rearm_absence_seconds <= 0:
            raise ValueError("time thresholds must be positive")
        if same_location_tolerance < 0:
            raise ValueError("same_location_tolerance must be >= 0")

        self.min_confidence = min_confidence
        self.min_confirmations = min_confirmations
        self.max_gap_seconds = max_gap_seconds
        self.same_location_tolerance = same_location_tolerance
        self.rearm_absence_seconds = rearm_absence_seconds
        self.reset()

    def reset(self) -> None:
        self._frame_index = 0
        self._event_counter = 0
        self._candidates: dict[tuple[str, int], _Candidate] = {}
        self._last_seen: dict[tuple[str, int], float] = {}
        self._emitted_latches: set[tuple[str, int]] = set()

    def _location_key(self, card: str, x: float | None) -> tuple[str, int]:
        bucket = -1 if x is None else round(x / max(self.same_location_tolerance, 0.01))
        return card, bucket

    def process_frame(
        self, detections: Iterable[dict[str, Any]], timestamp: float
    ) -> list[DetectionEvent]:
        """Process one frame and return newly confirmed card-detection events."""
        if timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        self._frame_index += 1
        valid: list[tuple[tuple[str, int], str, float, float | None]] = []

        for detection in detections:
            card = str(detection.get("card", "")).strip()
            try:
                confidence = float(detection.get("conf", 0.0))
                x_value = detection.get("x")
                x = None if x_value is None else float(x_value)
                detected_at = float(detection.get("t", timestamp))
            except (TypeError, ValueError):
                continue
            if not card or confidence < self.min_confidence:
                continue
            if x is not None and not 0.0 <= x <= 1.0:
                continue
            if detected_at < 0 or abs(detected_at - timestamp) > self.max_gap_seconds:
                continue
            key = self._location_key(card, x)
            valid.append((key, card, confidence, x))

        seen_keys = {item[0] for item in valid}
        for key in list(self._candidates):
            if key not in seen_keys:
                del self._candidates[key]

        for key in list(self._emitted_latches):
            if key not in seen_keys:
                last_seen = self._last_seen.get(key)
                if last_seen is None or timestamp - last_seen >= self.rearm_absence_seconds:
                    self._emitted_latches.remove(key)
                    self._candidates.pop(key, None)

        events: list[DetectionEvent] = []
        for key, card, confidence, x in valid:
            candidate = self._candidates.get(key)
            if candidate is None or timestamp - candidate.last_timestamp > self.max_gap_seconds:
                candidate = _Candidate(0, timestamp, 0.0, x)
            candidate.count += 1
            candidate.last_timestamp = timestamp
            candidate.confidence_sum += confidence
            candidate.x = x
            self._candidates[key] = candidate
            self._last_seen[key] = timestamp

            if candidate.count == self.min_confirmations and key not in self._emitted_latches:
                self._event_counter += 1
                events.append(
                    DetectionEvent(
                        event_id=f"detection_{self._event_counter:06d}",
                        timestamp=timestamp,
                        event_type="card_detected",
                        card=card,
                        confidence=candidate.confidence_sum / candidate.count,
                        x=x,
                    )
                )
                self._emitted_latches.add(key)
        return events
