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
CONTROL_ROOT = DATA_ROOT / "controls"
PARENT_ROOT = ROOT
PARENT_CONFIG = PARENT_ROOT / "configs/experiment.toml"
RUNNER = ROOT / "scripts/run_task_transfer_controls.py"


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(PARENT_ROOT / "src"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLCONFIGDIR": "/tmp/ten_rank_transfer_controls_matplotlib",
            "ATTENTION_IDENTIFIABILITY_DATA": str(DATA_ROOT),
        }
    )
    return env


def run_seed(seed: int) -> tuple[int, float]:
    started = time.monotonic()
    log_path = CONTROL_ROOT / "logs" / f"seed_{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        subprocess.run(
            [sys.executable, str(RUNNER), "--seed", str(seed)],
            cwd=CONTROL_ROOT,
            env=environment(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return seed, time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.workers != 4:
        raise ValueError("This control suite is locked to four CPU workers")
    subprocess.run(
        [sys.executable, str(RUNNER), "--validate-only"],
        cwd=CONTROL_ROOT,
        env=environment(),
        check=True,
    )
    if args.validate_only:
        return
    with PARENT_CONFIG.open("rb") as stream:
        config = tomllib.load(stream)
    seeds = [int(value) for value in config["experiment"]["confirmatory_seeds"]]
    suite_started = time.monotonic()
    durations: list[float] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_seed, seed): seed for seed in seeds}
        for completed, future in enumerate(as_completed(futures), start=1):
            seed, duration = future.result()
            durations.append(duration)
            remaining = len(seeds) - completed
            remaining_eta = (sum(durations) / len(durations)) * remaining / args.workers
            print(
                json.dumps(
                    {
                        "event": "task_transfer_control_progress",
                        "seed": seed,
                        "completed": completed,
                        "total": len(seeds),
                        "seed_seconds": duration,
                        "elapsed_seconds": time.monotonic() - suite_started,
                        "remaining_eta_seconds": remaining_eta,
                        "workers": args.workers,
                    }
                ),
                flush=True,
            )
    subprocess.run(
        [sys.executable, str(RUNNER), "--finalize"],
        cwd=CONTROL_ROOT,
        env=environment(),
        check=True,
    )


if __name__ == "__main__":
    main()
