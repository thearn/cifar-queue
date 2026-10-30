# Experiments

One directory per experiment or attempt, `experiments/<ID>[-aN]/`. A repair gets a new attempt suffix; an earlier
attempt is never edited. Each directory holds `protocol.md`, which is frozen before launch, then `result.json`, logs
or job ids (`solveq` job directories stay under `results/jobs/`, referenced by path), and for searches the copied
study `.db` (with its sha256 when large artefacts are kept outside git).

## protocol.md template

```markdown
# <ID>: <title>
Frozen: <date>, commit <hash> (branch autoresearch/footprint-arch), dataset sha256 <hash>

- Hypothesis and mechanism:
- Failure or uncertainty addressed:
- Decision it informs (what each outcome changes):
- Baseline / candidate identity; the variable or bundle changed:
- Fixed inputs: split (configs/data.toml [split]), evaluator version (footprint.py commit), seeds:
- Measurements and gates (PLAN.md §2 references), tolerance, stop condition:
- Compute: target (local | colab:cifar-g4 | ...), slots from calibration, units cap, per-job minutes, --stall-min:
- GPU-use check: expected busy share, how measured (usage.json), action if < 80 %:
- Artifacts and recovery (--resume path, study checkpoint):
```

After the run, the experiment gets a row appended to `../ledger.tsv`, and `../STATUS.md` is updated.
