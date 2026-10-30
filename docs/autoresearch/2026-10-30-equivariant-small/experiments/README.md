# Experiments

One directory per experiment or attempt, `experiments/<ID>[-aN]/`. A repair gets a new attempt suffix; an earlier
attempt is never edited. Each holds `protocol.md`, frozen before launch (its commit named), then results, logs or job
directories by path (`results/jobs/...`), and for searches the study `.db` path.

## protocol.md template

```markdown
# <ID>: <title>
Frozen: <date>, commit <hash> (branch eqsm/...), dataset dataset-6e01ae0c4036

- Hypothesis and mechanism:
- Failure or uncertainty addressed:
- Decision it informs (what each outcome changes):
- Baseline / candidate identity; the variable or bundle changed:
- Matched-parameter and matched-MAC baselines (PLAN §2.2):
- Fixed inputs: split, evaluator version, seeds, epochs charged (PLAN §2.1), test sealed:
- Measurements and gates (PLAN §2 references), tolerance, stop condition:
- Compute: target, trainings at once, units cap, per-job minutes, --stall-min, balance before:
- GPU-use check: expected busy share, action if < 80 %:
- Artifacts and recovery:
```

After the run: a row in `../ledger.tsv`, `../STATUS.md` updated, the balance after recorded for VM jobs.
