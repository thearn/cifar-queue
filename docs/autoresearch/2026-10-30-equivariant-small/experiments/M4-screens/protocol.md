# M4-screens: levers on the flip-equivariant finalist at 100 epochs
Frozen: 2026-10-30, commit fa4b708 (eqsm/main; M1 and O4 merged). Dataset dataset-6e01ae0c4036. Run on the G4 (the
4060 is shared with another session; PLAN §4 allows this), grid `results/autoresearch/eqsm/m4-grid.json`.

- **Base and baseline:** the finalist (D1 D1, 128/112/144, 1.94 M, 237 M MACs, one-teacher KD w 0.12) at 100 epochs;
  its baseline is M3's `D1-e100` (same seeds 1-3, same code, same card): 95.62 / 95.79 / 96.02 % val at 1/2/6 views.
- **Arms (one variable each):** SE (`se=8`, +13k params); SRF (`srf=True`, widened to 160/136/176 for 1.94 M,
  353 M MACs); TIE (`tie=1`, +0.8k params, 299 M MACs); MUON (`optimizer='muon'`, Muon lr 0.24, momentum 0.6 for
  filters); FILT (`filter_keep=0.5`, 200 passes = 100 charged epochs, proxy 32/64/64 trained first for 200 epochs,
  not charged); KDW (`kd_weighting='disagree'`). The Fourier block (O1) is dropped: after the whitening layer's valid
  2×2 convolution the group maps are 15, 7 and 3 pixels, where a 3×3 convolution is already nearly global.
- **Gates (PLAN §2.1):** Promising at ≥ +0.15 pp at 1 view over the baseline (3 paired seeds); Null inside ± 0.10.
  The target decision (PLAN §1): if no lever, alone or combined, brings one view to ≥ 95.7 % at 100 epochs, the
  target falls back to six views.
- **Compute:** 18 trainings plus the 9-run PW control repair of M3 (param_cap 0), ~40 min, cap 8 units.
