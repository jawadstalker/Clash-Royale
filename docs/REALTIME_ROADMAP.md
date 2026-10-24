# Real-time roadmap: card cycle and next-card prediction

This milestone adds a deterministic card-cycle state machine and evaluation
baselines before connecting vision detections to live predictions.

## Current additions

- `src/card_cycle.py`: tracks an 8-slot deck cycle. Supply the slots in known
  cycle order; the first four are treated as the current hand. Use `None` for
  unknown slots. Only confirmed plays from the current hand are accepted.
- `src/baselines.py`: frequency and last-card transition baselines.
- `src/evaluate_predictor.py`: match-level holdout evaluation for the baselines.
- `tests/`: unit tests for the cycle state and baseline behavior.

## First commands

From the repository root:

```bash
python -m unittest discover -s tests -v
python src/prepare_sequences.py
python src/evaluate_predictor.py --test-fraction 0.2 --seed 42
```

The evaluation command expects at least two match sequences in
`data/sequences/sequences.json`. It holds out entire matches to avoid leaking
near-duplicate contexts from one match across train and test.

## Important semantics

The current sequence model predicts the **next observed card play** from recent
plays. That is not necessarily the same as predicting which card is currently
in the opponent's hand or which card they will choose to play next. Keep these
targets separate in future evaluations.

The cycle engine does not infer missing cards or determine whether a detection
is a real play. Feed it only confirmed, deduplicated `card_played` events from
the future event engine. Do not pass every raw frame detection directly into
`play_card()`.

## Next steps

1. Fix and test raw-sighting deduplication against repeated sightings and
   separate plays of the same card.
2. Add an event engine that confirms a card play across frames and assigns a
   unique event ID.
3. Add replayable match event logs.
4. Connect capture -> detector -> event engine -> cycle -> prediction.
5. Evaluate on real held-out matches and report per-match metrics.
