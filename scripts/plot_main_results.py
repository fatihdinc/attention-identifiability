from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from identifiability_llm.ten_task_attention import TASK_NAMES  # noqa: E402


TASK_LABELS = {
    "exact_a_lookup": "1. Exact A",
    "noisy_a_lookup": "2. Noisy A",
    "partial_a_lookup": "3. Partial A",
    "b_lookup": "4. B lookup",
    "two_key_lookup": "5. Two-key",
    "category_filtered_lookup": "6. Category filter",
    "highest_priority": "7. Highest priority",
    "lowest_priority": "8. Lowest priority",
    "category_majority_vote": "9. Category vote",
    "three_item_vote": "10. Three-item vote",
}

# Largest coincident dots are drawn first; smaller dots remain visible as
# nested colored centers. Trained horizons use dashed and solid lines so the
# comparison remains readable in grayscale.
METHOD_ORDER = [
    "trained_low_rank_5k",
    "trained_low_rank_10k",
    "query_input_gram",
    "query_output_gram",
    "key_input_gram",
    "key_output_gram",
    "effective_m_svd",
]
METHOD_LABELS = {
    "trained_low_rank_5k": "Trained low-rank (5k)",
    "trained_low_rank_10k": "Trained low-rank (10k)",
    "key_input_gram": "Key-input Gram",
    "key_output_gram": "Key-output Gram",
    "query_input_gram": "Query-input Gram",
    "query_output_gram": "Query-output Gram",
    "effective_m_svd": "Effective-M SVD",
}
METHOD_COLORS = {
    "trained_low_rank_5k": "#8C564B",
    "trained_low_rank_10k": "#6A3D9A",
    "key_input_gram": "#D55E00",
    "key_output_gram": "#009E73",
    "query_input_gram": "#CC79A7",
    "query_output_gram": "#56B4E9",
    "effective_m_svd": "#0072B2",
}
METHOD_MARKER_SIZES = {
    "trained_low_rank_5k": 10.2,
    "trained_low_rank_10k": 8.9,
    "key_input_gram": 7.8,
    "key_output_gram": 6.8,
    "query_input_gram": 5.8,
    "query_output_gram": 4.7,
    "effective_m_svd": 3.6,
}
METHOD_LINESTYLES = {
    "trained_low_rank_5k": "--",
    "trained_low_rank_10k": "-",
    "key_input_gram": "-",
    "key_output_gram": "-",
    "query_input_gram": "-",
    "query_output_gram": "-",
    "effective_m_svd": "-",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8.1,
            "figure.titlesize": 14,
            "savefig.dpi": 240,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def draw_figure(
    data: pd.DataFrame,
    *,
    x_max: int,
    x_ticks: list[int],
    title_suffix: str,
) -> plt.Figure:
    figure, axes = plt.subplots(2, 5, figsize=(18.5, 8.4), sharex=True, sharey=True)
    legend_handles = []
    legend_labels = []
    for panel_index, task in enumerate(TASK_NAMES):
        row_index, column_index = divmod(panel_index, 5)
        axis = axes[row_index, column_index]
        task_data = data[data["task"] == task]
        for method_index, method in enumerate(METHOD_ORDER):
            method_data = task_data[task_data["method"] == method]
            if method_data.empty:
                raise ValueError(f"Missing curve for {task}/{method}")
            summary = (
                method_data.groupby("rank")["accuracy"]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("rank")
            )
            x = summary["rank"].to_numpy(float)
            mean = summary["mean"].to_numpy(float)
            std = summary["std"].fillna(0).to_numpy(float)
            line = axis.plot(
                x,
                mean,
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                marker="o",
                markersize=METHOD_MARKER_SIZES[method],
                markeredgewidth=0,
                linewidth=1.55,
                label=METHOD_LABELS[method],
                zorder=3.0 + 0.2 * method_index,
            )[0]
            axis.fill_between(
                x,
                np.clip(mean - std, 0, 1.02),
                np.clip(mean + std, 0, 1.02),
                color=METHOD_COLORS[method],
                alpha=0.10,
                linewidth=0,
                zorder=1.5,
            )
            if panel_index == 0:
                legend_handles.append(line)
                legend_labels.append(METHOD_LABELS[method])

        baseline = float(task_data.groupby("seed")["full_model_accuracy"].first().mean())
        baseline_line = axis.axhline(
            baseline,
            color="#666666",
            linestyle="--",
            linewidth=1.15,
            label="Original model",
            zorder=1,
        )
        power_data = task_data[task_data["method"] == "effective_m_svd"]
        power = (
            power_data.groupby("rank")["svd_cumulative_power"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("rank")
        )
        power_axis = axis.twinx()
        power_line = power_axis.plot(
            power["rank"],
            power["mean"],
            color="black",
            linewidth=1.75,
            label="Cumulative SVD power",
            zorder=4,
        )[0]
        power_axis.fill_between(
            power["rank"],
            np.clip(power["mean"] - power["std"].fillna(0), 0, 1),
            np.clip(power["mean"] + power["std"].fillna(0), 0, 1),
            color="black",
            alpha=0.08,
            linewidth=0,
            zorder=1,
        )
        power_axis.set_ylim(0, 1.02)
        power_axis.set_yticks([0, 0.5, 1.0])
        show_right = column_index == 4
        power_axis.tick_params(
            axis="y",
            right=show_right,
            labelright=show_right,
            labelsize=8,
            length=2,
        )
        power_axis.spines["right"].set_visible(show_right)
        axis.set_title(TASK_LABELS[task], pad=7)
        axis.set_xlim(0, x_max)
        axis.set_ylim(0, 1.02)
        axis.set_xticks(x_ticks)
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axis.grid(alpha=0.20, linewidth=0.6)
        axis.tick_params(labelsize=8)
        if panel_index == 0:
            legend_handles.extend([power_line, baseline_line])
            legend_labels.extend(["Cumulative SVD power", "Original model"])

    figure.suptitle(
        "Joint ten-task model: trained effective-score rank reconstruction "
        f"({title_suffix})",
        y=0.985,
    )
    figure.supxlabel("Retained rank K", y=0.085)
    figure.text(
        0.012,
        0.50,
        "Test classification accuracy",
        rotation="vertical",
        va="center",
        ha="center",
        fontsize=10,
    )
    figure.text(
        0.992,
        0.50,
        "Cumulative SVD power",
        rotation=-90,
        va="center",
        ha="center",
        fontsize=10,
    )
    figure.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=5,
        frameon=False,
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.975,
        top=0.92,
        bottom=0.17,
        wspace=0.13,
        hspace=0.28,
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--full-png", type=Path, required=True)
    parser.add_argument("--zoom-png", type=Path, required=True)
    parser.add_argument("--full-page-pdf", type=Path, required=True)
    parser.add_argument("--zoom-page-pdf", type=Path, required=True)
    arguments = parser.parse_args()
    data = pd.read_csv(arguments.table.resolve(), low_memory=False)
    observed_methods = set(data["method"].unique())
    missing_methods = sorted(set(METHOD_ORDER) - observed_methods)
    if missing_methods:
        raise ValueError(f"Missing required methods: {missing_methods}")
    analytic = data[data["method"] == "effective_m_svd"]
    missing_dense = sorted(set(range(51)) - set(analytic["rank"].astype(int).unique()))
    required_tail = {60, 70, 80, 90, 100, 110, 120, 128}
    missing_tail = sorted(required_tail - set(analytic["rank"].astype(int).unique()))
    if missing_dense or missing_tail or int(analytic["rank"].max()) != 128:
        raise ValueError(
            f"Analytic dense curve is incomplete: missing={missing_dense}, "
            f"missing_tail={missing_tail}, maximum={int(analytic['rank'].max())}"
        )
    for method in ("trained_low_rank_5k", "trained_low_rank_10k"):
        observed = sorted(data[data["method"] == method]["rank"].astype(int).unique())
        if observed != [0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]:
            raise ValueError(f"Trained rank grid is incomplete for {method}: {observed}")

    style()
    full_figure = draw_figure(
        data,
        x_max=128,
        x_ticks=[0, 20, 40, 60, 80, 100, 120, 128],
        title_suffix="5k versus 10k updates; full rank range",
    )
    zoom_figure = draw_figure(
        data,
        x_max=50,
        x_ticks=[0, 10, 20, 30, 40, 50],
        title_suffix="5k versus 10k updates; ranks 0-50",
    )
    outputs = [
        arguments.full_png.resolve(),
        arguments.zoom_png.resolve(),
        arguments.full_page_pdf.resolve(),
        arguments.zoom_page_pdf.resolve(),
    ]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite final figure: {path}")
    full_figure.savefig(outputs[0], bbox_inches="tight")
    zoom_figure.savefig(outputs[1], bbox_inches="tight")
    full_figure.savefig(outputs[2], bbox_inches="tight")
    zoom_figure.savefig(outputs[3], bbox_inches="tight")
    plt.close(full_figure)
    plt.close(zoom_figure)
    print(
        {
            "event": "effective_score_distillation_figures_complete",
            "full_png": str(outputs[0]),
            "zoom_png": str(outputs[1]),
            "full_page_pdf": str(outputs[2]),
            "zoom_page_pdf": str(outputs[3]),
        }
    )


if __name__ == "__main__":
    main()
