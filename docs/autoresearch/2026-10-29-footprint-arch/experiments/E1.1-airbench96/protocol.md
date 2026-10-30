# E1.1: airbench96 replicated
Frozen: 2026-10-29 (snapshot `refs/autoresearch/E1.1`).

- Hypothesis: the port (`cifarq.airbench`, `AIRBENCH96` = `legacy/airbench96.py` at 4c1b6d1: 128/384/512, depth 3,
  37 epochs, cutout 12, translate 4, TTA level 2) reproduces 96.05 % (the paper; the script's own header: 96.03 % over
  400 runs) when trained on all 50,000 images.
- Decision: whether the port is right (every Goal A trick comes from it); a gap over 0.3 pp is diagnosed before
  anything else runs. Also gives the local 45k reference and the R1 teacher.
- Runs: (a) 1 seed, `train_on=train+val` (the published setting: 50k, test on the 10k); (b) 3 seeds on 45k/5k with
  `--save` (reference and teacher candidates).
- Commands:
  `seeds.py E1.1-50k --preset airbench96 --set train_on=train+val --seeds 0 --targets local`
  `seeds.py E1.1-45k --preset airbench96 --seeds 0 1 2 --save --targets local`
- Gates: (a) test (TTA) within 0.3 pp of 96.03 %, no-TTA within 0.3 pp of 95.6 %; (b) recorded; parameter count
  9,918,776 (already confirmed); footprint measured with E0.1's bench (expected: far outside Class S against B0).
- Compute: local 4060; ~5 min a run expected (34.7 s on an A100; the 4060 is roughly 6-8x slower), sequential.
