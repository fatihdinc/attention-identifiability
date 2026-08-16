# Reproducibility guide

## Data boundary

Generated material is written under `data/`, which is intentionally excluded from Git. This includes teacher checkpoints, cached Gram bases, learned low-rank factors, per-seed records, aggregate tables, audits, manifests, and logs.

Set `ATTENTION_IDENTIFIABILITY_DATA` before running any command to move this tree to another disk. Every entry point uses the same resolved location.

## Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Pinned versions are also listed in `requirements.txt`.

## Validation gate

```bash
python scripts/run_experiment.py --stage validate --workers 4
```

This checks protocol hashes, fixed seed/rank grids, model architecture, deterministic generators, direct-$M$ equivalence, Gram identities, rank constraints, learned-factor initialization, and all unit tests. No scientific result should be interpreted if this gate fails.

## Stage order

### 1. Train teachers

```bash
python scripts/run_experiment.py --stage train --workers 4
```

Trains one joint ten-task teacher for each of 20 fixed seeds. Outputs are placed under:

```text
data/artifacts/ten_task_effective_score_20seeds_v1/confirmatory/checkpoints/
data/results/ten_task_effective_score_20seeds_v1/confirmatory/training/
```

### 2. Run Gram and SVD reconstructions

```bash
python scripts/run_experiment.py --stage reconstruct --workers 4
```

This creates task-specific Gram bases, evaluates four Gram reconstructions plus direct $M$-SVD, and aggregates 59,000 primary conditions (`20 seeds × 10 tasks × 5 methods × 59 ranks`).

### 3. Train functional low-rank matrices

```bash
python scripts/run_experiment.py --stage low-rank --workers 4
```

This trains the rank-factorized replacement matrices and records 4,800 evaluated conditions (`20 × 10 × 2 horizons × 12 ranks`). The combined main table therefore has 63,800 rows.

### 4. Run all-pairs controls

```bash
python scripts/run_experiment.py --stage controls --workers 4
```

This evaluates 472,000 conditions (`20 seeds × 100 ordered task pairs × 4 Gram methods × 59 ranks`) under task-code-matched calibration. Raw controls are written to `data/controls/`; final control figures are written to `figures/controls/`.

### 5. Final figures and bundle audit

```bash
python scripts/run_experiment.py --stage finalize --workers 4
```

This creates the ten-panel full-range and `K=0..50` figures in `figures/main/`, then audits file completeness, row counts, rank grids, seed counts, rank bounds, and numerical invariants.

## One-command reproduction

```bash
python scripts/run_experiment.py --stage all --workers 4
```

The workflow is fixed to four CPU workers, with one numerical thread per worker. Seed-level stages skip completed outputs and can be restarted after interruption. If only one file of an atomic seed-output pair exists, investigate rather than deleting or overwriting it silently.

## Expected final checks

- 20 unique teacher seeds.
- 10 tasks per seed.
- 59 analytic ranks and 12 learned ranks.
- 59,000 analytic reconstruction rows.
- 4,800 learned reconstruction rows.
- 472,000 transfer-control rows.
- Zero duplicate condition keys.
- All requested numerical rank bounds satisfied.
- Rank 128 recovers the full effective matrix and baseline accuracy.
- All seed-level and aggregate audits pass.
