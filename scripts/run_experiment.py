from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("ATTENTION_IDENTIFIABILITY_DATA", str(ROOT / "data"))
).expanduser().resolve()
EXPERIMENT_CONFIG = ROOT / "configs/experiment.toml"
LOW_RANK_CONFIG = ROOT / "configs/low_rank.toml"
PROTOCOL_LOCK = ROOT / "protocols/teacher_and_gram_protocol.json"
LOW_RANK_LOCK = ROOT / "protocols/trained_low_rank_protocol.json"


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLCONFIGDIR": "/tmp/ten_rank_transformer_matplotlib_cache",
            "ATTENTION_IDENTIFIABILITY_DATA": str(DATA_ROOT),
        }
    )
    return env


def run_checked(command: list[str], *, log_path: Path | None = None) -> None:
    if log_path is None:
        subprocess.run(command, cwd=ROOT, env=environment(), check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_seed_stage(
    stage: str,
    base_command: list[str],
    seeds: list[int],
    workers: int,
) -> None:
    started = time.monotonic()
    durations: list[float] = []

    def one(seed: int) -> tuple[int, float]:
        seed_started = time.monotonic()
        run_checked(
            [*base_command, "--seed", str(seed)],
            log_path=DATA_ROOT / "logs" / f"{stage}_seed_{seed}.log",
        )
        return seed, time.monotonic() - seed_started

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, seed): seed for seed in seeds}
        for completed, future in enumerate(as_completed(futures), start=1):
            seed, duration = future.result()
            durations.append(duration)
            remaining = len(seeds) - completed
            eta_seconds = (sum(durations) / len(durations)) * remaining / workers
            print(
                json.dumps(
                    {
                        "event": "parallel_stage_progress",
                        "stage": stage,
                        "seed": seed,
                        "completed": completed,
                        "total": len(seeds),
                        "seed_seconds": duration,
                        "elapsed_seconds": time.monotonic() - started,
                        "eta_seconds": eta_seconds,
                        "workers": workers,
                    }
                ),
                flush=True,
            )


def validate() -> None:
    run_checked([sys.executable, "scripts/lock_protocol.py"])
    run_checked(
        [
            sys.executable,
            "scripts/train_models.py",
            "--config",
            str(EXPERIMENT_CONFIG),
            "--preregistration",
            str(PROTOCOL_LOCK),
            "--validate-only",
        ]
    )
    run_checked(
        [
            sys.executable,
            "scripts/run_gram_reconstructions.py",
            "--config",
            str(EXPERIMENT_CONFIG),
            "--preregistration",
            str(PROTOCOL_LOCK),
            "--validate-only",
        ]
    )
    run_checked(
        [
            sys.executable,
            "scripts/train_low_rank.py",
            "--config",
            str(LOW_RANK_CONFIG),
            "--protocol-lock",
            str(LOW_RANK_LOCK),
            "--validate-only",
        ]
    )
    run_checked(
        [
            sys.executable,
            "scripts/run_task_transfer_controls_all.py",
            "--workers",
            "4",
            "--validate-only",
        ]
    )
    run_checked(
        [
            sys.executable,
            "scripts/run_support_projected_svd.py",
            "--config",
            str(EXPERIMENT_CONFIG),
            "--validate-only",
        ]
    )
    run_checked(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_ten_task_attention",
            "tests.test_ten_task_analysis",
            "tests.test_ten_task_effective_score",
            "tests.test_ten_task_distillation",
            "tests.test_support_projected_svd",
        ]
    )
    print(json.dumps({"event": "self_contained_validation_complete"}))


def train(seeds: list[int], workers: int) -> None:
    run_seed_stage(
        "train",
        [
            sys.executable,
            "scripts/train_models.py",
            "--config",
            str(EXPERIMENT_CONFIG),
            "--preregistration",
            str(PROTOCOL_LOCK),
        ],
        seeds,
        workers,
    )


def reconstruct(seeds: list[int], workers: int) -> None:
    run_seed_stage(
        "reconstruct",
        [
            sys.executable,
            "scripts/run_gram_reconstructions.py",
            "--config",
            str(EXPERIMENT_CONFIG),
            "--preregistration",
            str(PROTOCOL_LOCK),
        ],
        seeds,
        workers,
    )
    table = (
        DATA_ROOT
        / "results/ten_task_effective_score_20seeds_v1/confirmatory/tables"
        / "reconstruction_results.csv"
    )
    if not table.exists():
        run_checked(
            [
                sys.executable,
                "scripts/summarize_gram_reconstructions.py",
                "--config",
                str(EXPERIMENT_CONFIG),
                "--result-root",
                str(
                    DATA_ROOT
                    / "results/ten_task_effective_score_20seeds_v1/confirmatory"
                ),
            ]
        )


def low_rank(seeds: list[int], workers: int) -> None:
    run_seed_stage(
        "low_rank",
        [
            sys.executable,
            "scripts/train_low_rank.py",
            "--config",
            str(LOW_RANK_CONFIG),
            "--protocol-lock",
            str(LOW_RANK_LOCK),
        ],
        seeds,
        workers,
    )
    table_root = (
        DATA_ROOT
        / "results/ten_task_effective_score_20seeds_v1/extensions"
        / "trained_low_rank_effective_score_20seeds_v1_steps5000_10000/tables"
    )
    if not (table_root / "trained_reconstruction_results.csv").exists():
        run_checked(
            [
                sys.executable,
                "scripts/summarize_low_rank.py",
                "--parts-root",
                str(table_root.parent / "parts"),
                "--audits-root",
                str(table_root.parent / "audits"),
                "--analytic-table",
                str(
                    DATA_ROOT
                    / "results/ten_task_effective_score_20seeds_v1/confirmatory/tables/reconstruction_results.csv"
                ),
                "--output",
                str(table_root),
            ]
        )


def controls(workers: int) -> None:
    run_checked(
        [
            sys.executable,
            "scripts/run_task_transfer_controls_all.py",
            "--workers",
            str(workers),
        ]
    )


def support_svd(seeds: list[int], workers: int) -> None:
    run_seed_stage(
        "support_svd",
        [
            sys.executable,
            "scripts/run_support_projected_svd.py",
            "--config",
            str(EXPERIMENT_CONFIG),
        ],
        seeds,
        workers,
    )
    run_checked(
        [
            sys.executable,
            "scripts/run_support_projected_svd.py",
            "--config",
            str(EXPERIMENT_CONFIG),
            "--summarize",
            "--replace-derived",
        ]
    )


def finalize() -> None:
    table = (
        DATA_ROOT
        / "results/ten_task_effective_score_20seeds_v1/extensions"
        / "support_projected_svd_v1/tables/figure_reconstruction_results.csv"
    )
    figure_root = ROOT / "figures/main"
    full_png = figure_root / "full_range_20seeds.png"
    run_checked(
        [
            sys.executable,
            "scripts/plot_support_projected_results.py",
            "--table",
            str(table),
            "--full-png",
            str(full_png),
            "--zoom-png",
            str(figure_root / "zoom_K0_50_20seeds.png"),
            "--full-page-pdf",
            str(figure_root / "full_range_20seeds.pdf"),
            "--zoom-page-pdf",
            str(figure_root / "zoom_K0_50_20seeds.pdf"),
            "--overwrite",
        ]
    )
    run_checked([sys.executable, "scripts/audit_bundle.py"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=[
            "validate",
            "train",
            "reconstruct",
            "low-rank",
            "support-svd",
            "controls",
            "finalize",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.workers != 4:
        raise ValueError("This experiment is locked to exactly four CPU workers")
    with EXPERIMENT_CONFIG.open("rb") as stream:
        config = tomllib.load(stream)
    seeds = [int(value) for value in config["experiment"]["confirmatory_seeds"]]

    if arguments.stage in {"validate", "all"}:
        validate()
    if arguments.stage in {"train", "all"}:
        train(seeds, arguments.workers)
    if arguments.stage in {"reconstruct", "all"}:
        reconstruct(seeds, arguments.workers)
    if arguments.stage in {"low-rank", "all"}:
        low_rank(seeds, arguments.workers)
    if arguments.stage in {"support-svd", "all"}:
        support_svd(seeds, arguments.workers)
    if arguments.stage in {"controls", "all"}:
        controls(arguments.workers)
    if arguments.stage in {"finalize", "all"}:
        finalize()


if __name__ == "__main__":
    main()
