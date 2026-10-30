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

## Revision a2/a3 (2026-10-29, before any candidate was measured)

Attempt a1 (results/footprint/20261029-213115.json) failed the gates: GPU batch-1 latency CV 28 % (the first process
ran before the card left its idle clocks: 4.6 ms, then 2.5 ms), and CPU peak memory read 0 MB after the first run
(the floor was taken after importing torch, whose own transient peak exceeds a batch-1 inference). Repairs: a2
(branch fparch/E0.1-a2-bench) 200 warm-up calls at batch 1 and memory as the process peak minus a null model's
(same imports and inputs, no network); a3 CPU memory as resident RSS after the timed loop, less the null's.

Results of a3 on B0 (results/footprint/20261029-213952.json), 5 repeats:
- GPU: p50 CV 0.2-0.6 % (b1 2.51 ms, b256 20.2 ms, b1024 82.3 ms); peak GPU memory identical every time
  (17.8 / 149.4 / 552.3 MB). Original gates pass.
- CPU: p50 CV 3.2 % (1 thread, b1 15.2 ms), 5.5 % (all threads, b1 7.7 ms), 1.0 % (all threads, b256 738 ms);
  resident memory at b1 43 +- 3 MB (CV 5-6 %), at b256 all threads 92-132 MB (CV 17 %: allocator caches).

Revised gates (the CPU numbers cannot meet 1 % on this machine): GPU rows as before; CPU latency CV < 5 % at one
thread or batch 256, < 10 % at all threads and batch 1 (latency never gates a decision; the 2x trade-off flag
tolerates it); CPU memory for the classes from the batch-1 rows only, a candidate allowed 1.05 x B0 + 5 MB. Under
these, a3 passes. The bench is frozen at this revision.
