from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import plot_main_results as base


SUPPORT_METHOD = "support_projected_m_svd"
INTERNAL_PLOT_KEY = "effective_m_svd"
ORIGINAL_SVD_PLOT_KEY = "original_effective_m_svd"


def remove_cumulative_power(figure: object) -> None:
    """Strip the parent's spectral curve, twin axes, and right-side label."""

    primary_axes = [axis for axis in figure.axes if axis.get_title()]
    secondary_axes = [axis for axis in figure.axes if not axis.get_title()]
    if len(primary_axes) != 10 or len(secondary_axes) != 10:
        raise AssertionError(
            f"Expected ten primary and ten spectral axes, got "
            f"{len(primary_axes)} and {len(secondary_axes)}"
        )
    for axis in secondary_axes:
        axis.remove()
    for text in figure.texts:
        if text.get_text() == "Cumulative SVD power":
            text.set_visible(False)
    for legend in list(figure.legends):
        legend.remove()

    for axis in primary_axes:
        for line in axis.lines:
            if line.get_label() == "Original M SVD":
                line.set_markerfacecolor("white")
                line.set_markeredgecolor("#264653")
                line.set_markeredgewidth(0.9)
                line.set_markersize(4.6)
                line.set_linewidth(1.45)

    handles, labels = primary_axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=5,
        frameon=False,
    )
    figure.subplots_adjust(right=0.985)


def save_atomic(figure: object, path: Path, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite final figure: {path}")
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite temporary figure: {temporary}")
    figure.savefig(temporary, bbox_inches="tight")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--full-png", type=Path, required=True)
    parser.add_argument("--zoom-png", type=Path, required=True)
    parser.add_argument("--full-page-pdf", type=Path, required=True)
    parser.add_argument("--zoom-page-pdf", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    data = pd.read_csv(arguments.table.resolve(), low_memory=False)
    observed = set(data["method"].unique())
    expected = set(base.METHOD_ORDER) | {SUPPORT_METHOD}
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"Missing required methods: {missing}")
    required_tail = {60, 70, 80, 90, 100, 110, 120, 128}
    for method, label in (
        (SUPPORT_METHOD, "Support-projected SVD"),
        (INTERNAL_PLOT_KEY, "Original SVD"),
    ):
        method_data = data[data["method"] == method]
        ranks = set(method_data["rank"].astype(int).unique())
        missing_dense = sorted(set(range(51)) - ranks)
        missing_tail = sorted(required_tail - ranks)
        if missing_dense or missing_tail or int(method_data["rank"].max()) != 128:
            raise ValueError(
                f"{label} curve is incomplete: missing={missing_dense}, "
                f"missing_tail={missing_tail}, maximum={int(method_data['rank'].max())}"
            )
    for method in ("trained_low_rank_5k", "trained_low_rank_10k"):
        observed_ranks = sorted(
            data[data["method"] == method]["rank"].astype(int).unique()
        )
        if observed_ranks != [0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]:
            raise ValueError(f"Trained rank grid is incomplete for {method}")

    plot_data = data.copy()
    plot_data.loc[
        plot_data["method"] == INTERNAL_PLOT_KEY, "method"
    ] = ORIGINAL_SVD_PLOT_KEY
    plot_data.loc[plot_data["method"] == SUPPORT_METHOD, "method"] = INTERNAL_PLOT_KEY
    base.METHOD_ORDER = [
        method for method in base.METHOD_ORDER if method != INTERNAL_PLOT_KEY
    ] + [ORIGINAL_SVD_PLOT_KEY, INTERNAL_PLOT_KEY]
    base.METHOD_LABELS[ORIGINAL_SVD_PLOT_KEY] = "Original M SVD"
    base.METHOD_LABELS[INTERNAL_PLOT_KEY] = "Support-projected M SVD"
    base.METHOD_COLORS[ORIGINAL_SVD_PLOT_KEY] = "#264653"
    base.METHOD_MARKER_SIZES[ORIGINAL_SVD_PLOT_KEY] = 3.0
    base.METHOD_LINESTYLES[ORIGINAL_SVD_PLOT_KEY] = "-."
    base.METHOD_MARKER_SIZES[INTERNAL_PLOT_KEY] = 3.8
    base.style()
    full_figure = base.draw_figure(
        plot_data,
        x_max=128,
        x_ticks=[0, 20, 40, 60, 80, 100, 128],
        title_suffix="5k versus 10k updates; full rank range",
    )
    zoom_figure = base.draw_figure(
        plot_data,
        x_max=50,
        x_ticks=[0, 10, 20, 30, 40, 50],
        title_suffix="5k versus 10k updates; ranks 0-50",
    )
    remove_cumulative_power(full_figure)
    remove_cumulative_power(zoom_figure)

    outputs = [
        (full_figure, arguments.full_png.resolve()),
        (zoom_figure, arguments.zoom_png.resolve()),
        (full_figure, arguments.full_page_pdf.resolve()),
        (zoom_figure, arguments.zoom_page_pdf.resolve()),
    ]
    for figure, path in outputs:
        save_atomic(figure, path, overwrite=arguments.overwrite)
    base.plt.close(full_figure)
    base.plt.close(zoom_figure)
    print(
        {
            "event": "dual_svd_figures_complete",
            "full_png": str(outputs[0][1]),
            "zoom_png": str(outputs[1][1]),
            "full_page_pdf": str(outputs[2][1]),
            "zoom_page_pdf": str(outputs[3][1]),
        }
    )


if __name__ == "__main__":
    main()
