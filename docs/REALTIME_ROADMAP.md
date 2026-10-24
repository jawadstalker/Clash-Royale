# Real-time roadmap: card cycle and event confirmation

## Current modules

- `src/card_cycle.py`: stateful 8-slot deck cycle. It expects a known cycle order; unknown slots can be represented with `None`.
- `src/baselines.py`: frequency and last-card transition baselines.
- `src/evaluate_predictor.py`: match-level holdout evaluation.
- `src/event_engine.py`: confirms detections across adjacent frames and suppresses persistent sightings at the same card/location.
- `src/match_event_processor.py`: applies only explicit `card_played` events to the cycle and ignores raw `card_detected` events.

## Important limitation

A card visible on the arena is not necessarily evidence that the opponent just played that card: troops and buildings persist on screen. Therefore EventEngine deliberately emits `card_detected`, not `card_played`. Do not feed its events directly into the cycle engine. A future arena-delta, spawn-location, and temporal-tracking stage must establish that a new opponent card was played before emitting a `card_played` event.

## Tests and evaluation

From the repository root:

```bash
python -m unittest discover -s tests -v
python src/prepare_sequences.py
python src/evaluate_predictor.py --test-fraction 0.2 --seed 42
```

The evaluation command requires at least two match sequences in `data/sequences/sequences.json`.

## Next steps

1. Implement spatial/temporal association between detections in consecutive frames.
2. Determine whether a new opponent unit appeared and distinguish opponent units from the user's own cards.
3. Emit confirmed `card_played` events with stable event IDs.
4. Connect those events to `MatchEventProcessor` and the card-cycle engine.
5. Connect the event stream to next-card prediction and the live display.
