# M3-h1-curves: equivariance against test-time augmentation at matched parameters and MACs
DRAFT: freezes at M2's readout (E\* filled in) on `eqsm/main` with M1 part 2 merged (`val_acc_by_views`).
Dataset dataset-6e01ae0c4036.

- **Hypothesis and mechanism (H1, PLAN §1):** building the symmetry into the network buys at one forward pass what
  the plain network gets from TTA views, so at equal epochs an equivariant arm at 1 view matches or beats the plain
  arm at 2 views (and nearly at 6), at fewer MACs, with the margin largest at short schedules.
- **Uncertainty addressed:** Adly 2026 (literature §2) predicts a tie at equal test-time compute on a controlled
  task; nothing measures it on natural images in a short schedule. Q1 (D4) and Q2 (BlurPool) are untested here.
- **Decision:** which equivariance pattern is the base for M4/M5; whether H1 is reported as a gain or as a null.
- **Arms** (the finalist's recipe, `results/autoresearch/eqsm/finalist-d1d1.json`, one-teacher KD at w 0.12; only the
  architecture changes; parameters and MACs a view from `footprint.count_macs` a4):

  | Arm | Widths | Equivariance | Params | MACs/view |
  |---|---|---|---|---|
  | P plain twin | 128/160/192 | none | 1,912,824 | 177 M |
  | D1 finalist | 128/112/144 | flips, groups 2-3 | 1,944,024 | 237 M |
  | D1B finalist + BlurPool | 128/112/144 | flips, groups 2-3, MaxBlurPool | 1,944,024 | 237 M (+ blur) |
  | D4a | 128/160/76 | flips + quarter turns, group 3 | 1,912,120 | 260 M |
  | D4b | 128/60/72 | flips + quarter turns, groups 2-3 | 1,970,208 | 597 M |
  | PW plain, matched MACs (over the cap: a control) | 148/188/224 | none | 2,600,936 | 234 M |

- **Comparisons read (validation, 3-seed means, `val_acc_by_views`):**
  - matched parameters: D1, D1B, D4a, D4b at 1 view against P at 1 view;
  - matched inference cost: each equivariant arm at 1 view against P at 2 views (354 M) and at 6 views (1,061 M);
  - matched MACs per view, parameters free: D1 at 1 view against PW at 1 view;
  - the flip TTA's residual value on each equivariant arm (its 2-view minus 1-view).
- **Budgets:** 20, 40 and E\* epochs. **Seeds:** 1-3 (paired across arms). **Test:** sealed.
- **Gates:** Promising for an arm at ≥ +0.15 pp at 1 view over P at 2 views at E\*; H1 null if no arm reaches it at any
  budget (PLAN §6). Tolerance from M2's seed spread.
- **Compute:** `colab:cifar-g4`, two jobs at a time (the card's default slots; one VM). 6 arms × 3 budgets × 3 seeds
  = 54 trainings, ~1-3 min each (D4b ~3× the MACs), ~60 min, cap 10 units (from the 15-unit reserve's margin over
  the ~6 planned). Balance before recorded at launch. `SOLVEQ_SESSION=eqsm`, watcher on.
- **Artifacts:** `results/autoresearch/M3-<arm>-e<E>/`; one ledger row per arm with both matched deltas.
