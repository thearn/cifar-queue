# E0.3: B0 over 5 seeds on the 4060
Frozen: 2026-10-29 (snapshot `refs/autoresearch/E0.3`).

- Hypothesis: B0 (`results/g4-best.json`) reproduces its G4 figure (94.57 % test over 3 seeds) on the 4060 within
  noise; its seed sd is under 0.2 pp.
- Decision: the reference numbers and the noise for every later comparison; if sd > 0.2 pp, the measurement protocol
  is investigated before any candidate is judged.
- Configuration: `cifarq.train:train` with `results/g4-best.json` (width 48, schedule-free AdamW, 30 epochs, batch 256,
  whitening, flip TTA), seeds 0-4, `--save`.
- Command: `pipeline/tools/seeds.py E0.3 --fn cifarq.train:train --params results/g4-best.json --seeds 0 1 2 3 4 --save --targets local`
- Measurements: val and test accuracy per seed, train seconds; then E0.1's bench on the median-validation seed.
- Gate: completes; mean test within 0.3 pp of 94.57 %, or the difference explained (card, software) before going on.
- Compute: local 4060, ~3 min a seed expected (22 s on the G4), one at a time.
