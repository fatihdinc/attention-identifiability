from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_once(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != payload:
            raise RuntimeError(f"Existing protocol lock differs: {path}")
        print({"event": "protocol_lock_already_valid", "path": str(path)})
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print({"event": "protocol_locked", "path": str(path)})


def main() -> None:
    experiment_config = ROOT / "configs/experiment.toml"
    low_rank_config = ROOT / "configs/low_rank.toml"
    with experiment_config.open("rb") as stream:
        experiment = tomllib.load(stream)
    with low_rank_config.open("rb") as stream:
        low_rank = tomllib.load(stream)
    if experiment["experiment"]["confirmatory_seeds"] != low_rank["sweep"]["seeds"]:
        raise AssertionError("Training and low-rank seed lists differ")

    shared_sources = [
        ROOT / "src/identifiability_llm/paths.py",
        ROOT / "src/identifiability_llm/ten_task_attention.py",
        ROOT / "src/identifiability_llm/ten_task_analysis.py",
        ROOT / "src/identifiability_llm/ten_task_effective_score.py",
        ROOT / "src/identifiability_llm/ten_task_distillation.py",
    ]
    experiment_sources = [
        *shared_sources,
        ROOT / "scripts/train_models.py",
        ROOT / "scripts/run_gram_reconstructions.py",
        ROOT / "scripts/summarize_gram_reconstructions.py",
    ]
    low_rank_sources = [
        *shared_sources,
        ROOT / "scripts/train_low_rank.py",
        ROOT / "scripts/summarize_low_rank.py",
        ROOT / "scripts/plot_main_results.py",
        ROOT / "tests/test_ten_task_effective_score.py",
        ROOT / "tests/test_ten_task_distillation.py",
    ]
    control_sources = [
        *shared_sources,
        ROOT / "scripts/run_task_transfer_controls.py",
        ROOT / "scripts/run_task_transfer_controls_all.py",
    ]
    missing = [
        str(path)
        for path in [
            experiment_config,
            low_rank_config,
            *experiment_sources,
            *low_rank_sources,
            *control_sources,
        ]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Cannot lock missing files: {missing}")

    experiment_payload = {
        "locked_before_confirmatory_data": True,
        "status": "fresh_20_seed_protocol",
        "config_path": "configs/experiment.toml",
        "config_sha256": digest(experiment_config),
        "source_files": {
            str(path.relative_to(ROOT)): digest(path) for path in experiment_sources
        },
        "seeds": experiment["experiment"]["confirmatory_seeds"],
        "ranks": experiment["sweep"]["ranks"],
        "workers": 4,
    }
    low_rank_payload = {
        "status": "fresh_20_seed_low_rank_protocol",
        "config_path": "configs/low_rank.toml",
        "config_sha256": digest(low_rank_config),
        "source_files": {
            str(path.relative_to(ROOT)): digest(path) for path in low_rank_sources
        },
        "teacher_checkpoints": {},
        "seeds": low_rank["sweep"]["seeds"],
        "ranks": low_rank["sweep"]["ranks"],
        "horizons": low_rank["sweep"]["horizons"],
        "workers": 4,
    }
    control_payload = {
        "status": "task_code_matched_transfer_control_protocol",
        "parent_config_path": "configs/experiment.toml",
        "parent_config_sha256": digest(experiment_config),
        "source_files": {
            str(path.relative_to(ROOT)): digest(path) for path in control_sources
        },
        "seeds": experiment["experiment"]["confirmatory_seeds"],
        "ranks": experiment["sweep"]["ranks"],
        "methods": experiment["sweep"]["methods"][:-1],
        "ordered_task_pairs_per_seed": 100,
        "workers": 4,
    }
    for payload in (experiment_payload, low_rank_payload, control_payload):
        payload["terminology_only_migration"] = True
        payload["completed_run_semantics_unchanged"] = True
    write_once(
        experiment_payload,
        ROOT / "protocols/teacher_and_gram_protocol.json",
    )
    write_once(
        low_rank_payload,
        ROOT / "protocols/trained_low_rank_protocol.json",
    )
    write_once(
        control_payload,
        ROOT / "protocols/task_transfer_control_protocol.json",
    )


if __name__ == "__main__":
    main()
