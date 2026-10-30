# E1.2: airbench94 replicated
Frozen: 2026-10-29 (snapshot `refs/autoresearch/E1.2`).

- Hypothesis: `AIRBENCH94` (64/256/256, depth 2, 9.9 epochs, TTA level 2) reproduces 94.01 % on all 50k, and gives
  the first Goal A point (1,972,792 stored parameters, under the 2 M cap).
- Decision: the port's 2-conv path is right; the starting point of family A1's screen and of the method track.
- Runs: 2 seeds on 50k (`train_on=train+val`); 3 seeds on 45k/5k with `--save`.
- Gates: 50k mean within 0.3 pp of 94.01 % (TTA) and 93.2 % (no TTA).
- Compute: local 4060, ~1 min a run expected.
