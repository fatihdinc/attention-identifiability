from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import tomllib

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from identifiability_llm.paths import DATA_ROOT  # noqa: E402

RUN_ID = "ten_task_effective_score_20seeds_v1"
EXTENSION = "trained_low_rank_effective_score_20seeds_v1_steps5000_10000"
SUPPORT_EXTENSION = "support_projected_svd_v1"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    arguments = parser.parse_args()
    with (ROOT / "configs/experiment.toml").open("rb") as stream:
        experiment = tomllib.load(stream)
    seeds = [int(value) for value in experiment["experiment"]["confirmatory_seeds"]]
    tasks = [str(value) for value in experiment["experiment"]["task_names"]]
    ranks = [int(value) for value in experiment["sweep"]["ranks"]]
    methods = [str(value) for value in experiment["sweep"]["methods"]]
    base_artifacts = DATA_ROOT / "artifacts" / RUN_ID / "confirmatory"
    base_results = DATA_ROOT / "results" / RUN_ID / "confirmatory"
    extension_artifacts = DATA_ROOT / "artifacts" / RUN_ID / "extensions" / EXTENSION
    extension_results = DATA_ROOT / "results" / RUN_ID / "extensions" / EXTENSION
    support_results = (
        DATA_ROOT / "results" / RUN_ID / "extensions" / SUPPORT_EXTENSION
    )
    required = [
        ROOT / "protocols/teacher_and_gram_protocol.json",
        ROOT / "protocols/trained_low_rank_protocol.json",
        ROOT / "figures/main/full_range_20seeds.png",
        ROOT / "figures/main/zoom_K0_50_20seeds.png",
        base_results / "tables/reconstruction_results.csv",
        base_results / "tables/correctness_audit.json",
        extension_results / "tables/trained_reconstruction_results.csv",
        extension_results / "tables/analytic_and_trained_reconstruction_results.csv",
        extension_results / "tables/aggregate_audit.json",
        support_results / "tables/support_projected_svd_results.csv",
        support_results / "tables/figure_reconstruction_results.csv",
        support_results / "tables/aggregate_audit.json",
    ]
    required.extend(base_artifacts / "checkpoints" / f"seed_{seed}.pt" for seed in seeds)
    required.extend(base_results / "training" / f"seed_{seed}.json" for seed in seeds)
    required.extend(base_results / "audits" / f"seed_{seed}.json" for seed in seeds)
    required.extend(
        base_results / "parts" / f"seed_{seed}" / f"{task}.csv"
        for seed in seeds
        for task in tasks
    )
    required.extend(
        support_results / "parts" / f"seed_{seed}.csv" for seed in seeds
    )
    required.extend(
        support_results / "audits" / f"seed_{seed}.json" for seed in seeds
    )
    required.extend(
        extension_artifacts
        / "trained_bundles"
        / f"seed_{seed}"
        / f"task_{task_id + 1:02d}_{task}"
        / "effective_score.pt"
        for seed in seeds
        for task_id, task in enumerate(tasks)
    )
    missing = [display_path(path) for path in required if not path.exists()]

    analytic_path = base_results / "tables/reconstruction_results.csv"
    trained_path = extension_results / "tables/trained_reconstruction_results.csv"
    combined_path = support_results / "tables/figure_reconstruction_results.csv"
    support_path = support_results / "tables/support_projected_svd_results.csv"
    analytic = (
        pd.read_csv(analytic_path, low_memory=False)
        if analytic_path.exists()
        else pd.DataFrame()
    )
    trained = (
        pd.read_csv(trained_path, low_memory=False)
        if trained_path.exists()
        else pd.DataFrame()
    )
    combined = (
        pd.read_csv(combined_path, low_memory=False)
        if combined_path.exists()
        else pd.DataFrame()
    )
    support = (
        pd.read_csv(support_path, low_memory=False)
        if support_path.exists()
        else pd.DataFrame()
    )
    expected_analytic = len(seeds) * len(tasks) * len(methods) * len(ranks)
    expected_trained = len(seeds) * len(tasks) * 2 * 12
    expected_support = len(seeds) * len(tasks) * len(ranks)
    expected_combined = expected_analytic + expected_trained + expected_support
    checks = {
        "no_required_files_missing": not missing,
        "twenty_seeds": len(seeds) == 20 and len(set(seeds)) == 20,
        "analytic_row_count": len(analytic) == expected_analytic,
        "trained_row_count": len(trained) == expected_trained,
        "combined_row_count": len(combined) == expected_combined,
        "support_row_count": len(support) == expected_support,
        "analytic_seed_count": not analytic.empty and analytic["seed"].nunique() == 20,
        "trained_seed_count": not trained.empty and trained["seed"].nunique() == 20,
        "support_seed_count": not support.empty and support["seed"].nunique() == 20,
        "analytic_rank_grid": not analytic.empty
        and sorted(analytic["rank"].astype(int).unique().tolist()) == ranks,
        "trained_rank_grid": not trained.empty
        and sorted(trained["rank"].astype(int).unique().tolist())
        == [0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128],
        "all_analytic_rank_bounds": not analytic.empty
        and bool(analytic["rank_bound_satisfied"].all()),
        "all_trained_rank_bounds": not trained.empty
        and bool(trained["rank_bound_satisfied"].all()),
        "all_support_rank_bounds": not support.empty
        and bool(support["rank_bound_satisfied"].all()),
        "figure_uses_both_svd_curves": not combined.empty
        and int((combined["method"] == "support_projected_m_svd").sum())
        == expected_support
        and int((combined["method"] == "effective_m_svd").sum())
        == expected_support,
        "figure_has_no_duplicate_conditions": not combined.empty
        and not combined.duplicated(["seed", "task", "method", "rank"]).any(),
        "analytic_audit_pass": (base_results / "tables/correctness_audit.json").exists()
        and json.loads(
            (base_results / "tables/correctness_audit.json").read_text(encoding="utf-8")
        )["passed"],
        "trained_audit_pass": (extension_results / "tables/aggregate_audit.json").exists()
        and json.loads(
            (extension_results / "tables/aggregate_audit.json").read_text(encoding="utf-8")
        )["passed"],
        "support_audit_pass": (support_results / "tables/aggregate_audit.json").exists()
        and json.loads(
            (support_results / "tables/aggregate_audit.json").read_text(
                encoding="utf-8"
            )
        )["passed"],
        "no_symlinks": not any(path.is_symlink() for path in ROOT.rglob("*"))
        and not any(path.is_symlink() for path in DATA_ROOT.rglob("*")),
    }
    payload = {
        "passed": all(checks.values()),
        "checks": checks,
        "missing": missing,
        "counts": {
            "seeds": len(seeds),
            "tasks": len(tasks),
            "analytic_ranks": len(ranks),
            "analytic_rows": len(analytic),
            "expected_analytic_rows": expected_analytic,
            "trained_rows": len(trained),
            "expected_trained_rows": expected_trained,
            "combined_rows": len(combined),
            "expected_combined_rows": expected_combined,
            "support_rows": len(support),
        },
    }
    audit_path = DATA_ROOT / "reports/bundle_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if arguments.write_manifest:
        manifest = {
            str(path.relative_to(DATA_ROOT)): digest(path)
            for path in sorted(DATA_ROOT.rglob("*"))
            if path.is_file() and path.name != "MANIFEST.sha256.json"
        }
        (DATA_ROOT / "MANIFEST.sha256.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"event": "bundle_audit_complete", **payload}, indent=2))
    if not payload["passed"]:
        raise RuntimeError("Bundle audit failed")


if __name__ == "__main__":
    main()
