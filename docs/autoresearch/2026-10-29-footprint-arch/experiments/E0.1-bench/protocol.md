# E0.1: the footprint and runtime bench
Frozen: 2026-10-29, working tree of `layout` at e980ead plus the session's files (snapshot ref
`refs/autoresearch/E0.1` taken when the GPU measurements run); dataset sha256 6e01ae0c…

- Hypothesis: `cifarq.footprint` measures every axis of PLAN.md §2.2 repeatably on the local machine, and its classes
  reject larger and training-set-storing models.
- Failure addressed: no comparison of footprint or latency can be trusted without it.
- Decision: whether the bench is accepted as the session's evaluator (then frozen: later changes need a protocol
  revision and old/new numbers on the same models).
- Candidate: `src/cifarq/footprint.py`, `pipeline/tools/footprint_bench.py`.
- Fixed inputs: B0's saved network (from E0.3, seed with the median validation accuracy), the published width-64 model
  (`results/models/20261029-190902`), an airbench94 network (E1.2).
- Measurements and gates:
  1. `footprint_bench.py --repeat 5 B0`: coefficient of variation < 5 % on every p50 latency, < 1 % on every peak
     memory reading.
  2. `--ref B0 width64`: width64 classified `out`.
  3. `tests/test_footprint.py`: the training-set-storing model classified `out`; counts equal the trainer's.
- Already measured on the CPU (2026-10-29): gates 3 pass; width64 static = 6,573,130 params, 26,314,903 bytes, 379 M
  MACs; cpu1_b1 p50 11.0 ms; cpuall_b256 474 images/s (with an af2 job running on the GPU: CPU timings to be redone
  on a quiet machine).
- Compute: local only (4060 + CPU), nothing else running on the GPU, no queue job active during timing.
- Artifacts: results/footprint/<time>.json.
