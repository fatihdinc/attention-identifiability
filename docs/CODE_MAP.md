# Code map

All executable entry points are in `scripts/`. Library code contains no experiment-launch side effects.

| Script | Responsibility |
|---|---|
| `run_experiment.py` | Top-level four-worker orchestrator and resumable stage dispatcher. |
| `lock_protocol.py` | Hashes configs and source files into the three protocol records. |
| `train_models.py` | Trains and audits the 20 joint ten-task teacher models. |
| `run_gram_reconstructions.py` | Computes the four Grams, SVD basis, reconstructions, behavioral metrics, and seed audits. |
| `summarize_gram_reconstructions.py` | Aggregates analytic per-seed/task parts and correctness checks. |
| `train_low_rank.py` | Trains functional rank-factorized replacements against frozen teacher logits. |
| `summarize_low_rank.py` | Aggregates the 5k/10k benchmark and joins it to the analytic results. |
| `plot_main_results.py` | Produces the ten-task full-range and zoomed paper figures. |
| `run_task_transfer_controls.py` | Implements one-seed control computation, final aggregation, and control plots. |
| `run_task_transfer_controls_all.py` | Runs the transfer controls over 20 seeds on four workers. |
| `audit_bundle.py` | Checks final completeness, ranks, row counts, seed counts, invariants, and optional hashes. |

The core model and online generators are in `src/identifiability_llm/ten_task_attention.py`. Effective-score algebra and reconstruction live in `ten_task_effective_score.py`; learned functional replacement utilities live in `ten_task_distillation.py`.

