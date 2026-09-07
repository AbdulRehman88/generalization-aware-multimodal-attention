#!/usr/bin/env python
"""
Generate the revised manuscript Figure 3 evidence from frozen deep-learning
conventional protocol summaries.

Scientific role:
  Panel (a): existing classical modality/classifier capacity figure, preserved.
  Panel (b): existing classical SHAP Top-k ablation figure, preserved.
  Panel (c): NEW matched deep-learning protocol-sensitivity comparison using
             the same model/path across all four internal evaluation protocols.

The script does not train models and does not recompute metrics.
It only reads frozen summary CSVs and existing manuscript figures.
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from PIL import Image
except Exception:
    Image = None


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = (
    DEFAULT_PROJECT_ROOT
    / "_research_audit"
    / "deep_learning_scientific_results_v1"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "paper_figures"

FALLBACK_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11.0,
            "axes.titlesize": 12.0,
            "axes.labelsize": 11.0,
            "xtick.labelsize": 10.2,
            "ytick.labelsize": 10.2,
            "legend.fontsize": 10.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color="#d9d9d9", linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def extract_palette(image_path: Path | None, n: int = 7) -> list[str] | None:
    if Image is None or image_path is None or not image_path.is_file():
        return None
    try:
        im = Image.open(image_path).convert("RGB")
        im.thumbnail((900, 900))
        q = im.quantize(colors=96, method=Image.Quantize.MEDIANCUT).convert("RGB")
        counts = q.getcolors(maxcolors=1_000_000) or []
        counts.sort(reverse=True)

        selected: list[tuple[int, int, int]] = []
        for _, rgb in counts:
            r, g, b = [x / 255.0 for x in rgb]
            _, s, v = colorsys.rgb_to_hsv(r, g, b)
            if s < 0.35 or v < 0.28 or v > 0.93:
                continue
            if max(rgb) - min(rgb) < 35:
                continue
            if any(np.linalg.norm(np.array(rgb) - np.array(x)) < 45 for x in selected):
                continue
            selected.append(rgb)
            if len(selected) >= n:
                break

        if len(selected) < n:
            return None
        return [_rgb_to_hex(rgb) for rgb in selected[:n]]
    except Exception as exc:
        warnings.warn(f"Could not extract palette from {image_path}: {exc}")
        return None


def find_first(project_root: Path, filename: str) -> Path:
    matches = list(project_root.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Could not locate required manuscript figure '{filename}' under {project_root}"
        )
    return matches[0]


def find_style_source(project_root: Path) -> Path | None:
    for name in [
        "Fig_8_ablation_accuracy.png",
        "Fig_3_Accuracy.png",
        "Fig_7_end_to_end_latency_by_model.png",
    ]:
        matches = list(project_root.rglob(name))
        if matches:
            return matches[0]
    return None


def require_csv(results_dir: Path) -> Path:
    path = results_dir / "01_internal_conventional_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Frozen result file not found: {path}")
    return path


def validate_summary(df: pd.DataFrame) -> None:
    required = {
        "protocol",
        "model",
        "path",
        "test_balanced_accuracy_mean",
        "test_balanced_accuracy_std",
        "test_macro_f1_mean",
        "test_macro_f1_std",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns from 01 summary: {missing}")

    protocols = {
        "legacy_stratified_window_5fold",
        "legacy_repeated_stratified_window_5x3",
        "legacy_shuffle_split_window",
        "primary_grouped_5fold",
    }
    observed = set(df["protocol"].astype(str))
    if not protocols.issubset(observed):
        raise ValueError(f"Missing required protocol(s): {sorted(protocols - observed)}")

    checks = [
        (
            "legacy_stratified_window_5fold",
            "MultibranchTCN",
            "ECG_EEG_Pupil",
            0.8474430094357539,
        ),
        (
            "primary_grouped_5fold",
            "ShallowConvNet",
            "EEG",
            0.5595696649029982,
        ),
        (
            "primary_grouped_5fold",
            "MultibranchTCN",
            "ECG_EEG_Pupil",
            0.422248,
        ),
    ]

    for protocol, model, path, expected in checks:
        row = df[
            (df["protocol"] == protocol)
            & (df["model"] == model)
            & (df["path"] == path)
        ]
        if len(row) != 1:
            raise AssertionError(
                f"Expected exactly one row for {protocol} / {model} / {path}, found {len(row)}"
            )
        value = float(row.iloc[0]["test_balanced_accuracy_mean"])
        if not np.isclose(value, expected, atol=5e-7):
            raise AssertionError(
                f"Frozen-result check failed for {protocol} / {model} / {path}: "
                f"{value} != {expected}"
            )


def protocol_rows(df: pd.DataFrame, model: str, path: str) -> pd.DataFrame:
    order = [
        "legacy_stratified_window_5fold",
        "legacy_repeated_stratified_window_5x3",
        "legacy_shuffle_split_window",
        "primary_grouped_5fold",
    ]
    labels = {
        "legacy_stratified_window_5fold": "Stratified\n5-fold",
        "legacy_repeated_stratified_window_5x3": "Repeated\n5 x 3",
        "legacy_shuffle_split_window": "Shuffle\nsplit",
        "primary_grouped_5fold": "Participant-\ngrouped 5-fold",
    }

    sub = df[(df["model"] == model) & (df["path"] == path)].copy()
    sub = sub[sub["protocol"].isin(order)]
    if len(sub) != 4:
        raise AssertionError(
            f"Expected four protocols for {model} / {path}, found {len(sub)}"
        )
    sub["protocol"] = pd.Categorical(sub["protocol"], categories=order, ordered=True)
    sub = sub.sort_values("protocol").reset_index(drop=True)
    sub["display_protocol"] = [labels[str(x)] for x in sub["protocol"]]
    return sub


def draw_protocol_sensitivity(
    ax: plt.Axes,
    summary: pd.DataFrame,
    colors: list[str],
) -> None:
    # Matched configurations only. This avoids comparing different "best" models
    # across protocols as though they were the same estimand.
    shallow = protocol_rows(summary, "ShallowConvNet", "EEG")
    trimodal = protocol_rows(summary, "MultibranchTCN", "ECG_EEG_Pupil")

    x = np.arange(4)

    ax.errorbar(
        x,
        shallow["test_balanced_accuracy_mean"].to_numpy(),
        yerr=shallow["test_balanced_accuracy_std"].to_numpy(),
        marker="s",
        markersize=6.2,
        linewidth=1.7,
        capsize=2.6,
        color=colors[0],
        markeredgecolor="white",
        markeredgewidth=0.6,
        label="ShallowConvNet, EEG",
        zorder=3,
    )

    ax.errorbar(
        x,
        trimodal["test_balanced_accuracy_mean"].to_numpy(),
        yerr=trimodal["test_balanced_accuracy_std"].to_numpy(),
        marker="o",
        markersize=6.2,
        linewidth=1.7,
        capsize=2.6,
        color=colors[1],
        markeredgecolor="white",
        markeredgewidth=0.6,
        label="TCN, ECG + EEG + Pupil",
        zorder=3,
    )

    ax.axhline(
        1.0 / 3.0,
        color="#6e6e6e",
        linestyle="--",
        linewidth=0.95,
        zorder=1,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(shallow["display_protocol"].tolist())
    ax.set_ylim(0.28, 0.92)
    ax.set_ylabel("Balanced accuracy")
    ax.set_xlabel("Evaluation protocol")
    ax.set_title("Matched deep-model protocol sensitivity")
    clean_axis(ax, "y")
    ax.legend(frameon=False, loc="lower left")

    # Lightly emphasize the participant-grouped setting without adding
    # interpretation text inside the data region.
    ax.axvspan(2.62, 3.38, color="#d9d9d9", alpha=0.16, zorder=0)


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")


def make_panel_c(
    summary: pd.DataFrame,
    outdir: Path,
    colors: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    draw_protocol_sensitivity(ax, summary, colors)
    fig.tight_layout()
    save_figure(fig, outdir / "Fig_3c_Deep_Protocol_Sensitivity")
    plt.close(fig)


def make_composite_preview(
    project_root: Path,
    summary: pd.DataFrame,
    outdir: Path,
    colors: list[str],
) -> None:
    # Panels a and b remain the original manuscript images.
    try:
        fig_a_path = find_first(project_root, "accuracy_comparison_grouped.png")
    except FileNotFoundError:
        fig_a_path = find_first(project_root, "Fig_3_Accuracy.png")
    fig_b_candidates = [
        Path(project_root) / "outputs" / "summary" / "ablation_study" / "figures" / "accuracy_vs_topk.png",
        Path(project_root) / "outputs" / "summary" / "ablation_study_quick" / "figures" / "accuracy_vs_topk_kfold.png",
    ]

    fig_b_path = next((p for p in fig_b_candidates if p.exists()), None)
    if fig_b_path is None:
        raise FileNotFoundError(
            "Could not locate the required ablation figure. "
            "Expected one of: "
            + ", ".join(str(p) for p in fig_b_candidates)
        )

    if Image is None:
        raise RuntimeError("Pillow is required to compose the Figure 3 preview.")

    img_a = np.asarray(Image.open(fig_a_path).convert("RGB"))
    img_b = np.asarray(Image.open(fig_b_path).convert("RGB"))

    fig = plt.figure(figsize=(13.0, 8.8), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.02],
        width_ratios=[1.0, 1.0],
        hspace=0.34,
        wspace=0.20,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    ax_a.imshow(img_a)
    ax_a.axis("off")
    ax_b.imshow(img_b)
    ax_b.axis("off")
    draw_protocol_sensitivity(ax_c, summary, colors)

    panel_label(ax_a, "a")
    panel_label(ax_b, "b")
    panel_label(ax_c, "c")

    fig.subplots_adjust(left=0.07, right=0.985, top=0.965, bottom=0.10)
    save_figure(fig, outdir / "Fig_3_Capacity_Protocol_Sensitivity_PREVIEW")
    plt.close(fig)


def write_manifest(
    outdir: Path,
    csv_path: Path,
    style_source: Path | None,
    colors: list[str],
) -> None:
    lines = [
        "Figure 3 generation manifest",
        "",
        f"Frozen summary: {csv_path}",
        f"Palette source: {style_source if style_source else 'fallback palette'}",
        "",
        "Panel a: existing Fig_3_Accuracy.png, preserved",
        "Panel b: existing Fig_8_ablation_accuracy.png, preserved",
        "Panel c: NEW matched deep-model protocol sensitivity",
        "  - ShallowConvNet / EEG",
        "  - MultibranchTCN / ECG_EEG_Pupil",
        "  - same model/path compared across four protocols",
        "  - mean balanced accuracy with descriptive SD",
        "  - participant-grouped 5-fold is participant-disjoint",
        "",
        f"Panel c Shallow color: {colors[0]}",
        f"Panel c TCN color: {colors[1]}",
        "",
        "The composite file is a preview because panels a and b originate as existing PNG files.",
        "For the manuscript, use the original a and b assets plus Fig_3c_Deep_Protocol_Sensitivity",
        "in LaTeX so each source retains its highest available quality.",
        "",
        "No training and no metric recomputation are performed.",
    ]
    (outdir / "Fig_3_generation_manifest.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    project_root = args.project_root
    results_dir = args.results_dir
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    configure_style()

    csv_path = require_csv(results_dir)
    summary = pd.read_csv(csv_path)
    validate_summary(summary)

    style_source = find_style_source(project_root)
    extracted = extract_palette(style_source, n=7) if style_source else None
    palette = extracted if extracted is not None else FALLBACK_COLORS

    # Use two colors from the same manuscript palette for the matched deep comparison.
    # Marker shape additionally distinguishes architecture.
    colors = [palette[0], palette[3] if len(palette) > 3 else palette[1]]

    make_panel_c(summary, outdir, colors)
    make_composite_preview(project_root, summary, outdir, colors)
    write_manifest(outdir, csv_path, style_source, colors)

    print("PASS: 01_ frozen conventional summary validated.")
    if style_source:
        print(f"Style source: {style_source}")
        print(
            "Palette: extracted from existing manuscript figure."
            if extracted
            else "Palette: fallback used because source extraction was not reliable."
        )
    else:
        print("Style source not found. Fallback palette used.")

    print(f"Saved Figure 3 outputs to: {outdir}")
    for name in [
        "Fig_3c_Deep_Protocol_Sensitivity.png",
        "Fig_3c_Deep_Protocol_Sensitivity.pdf",
        "Fig_3c_Deep_Protocol_Sensitivity.svg",
        "Fig_3_Capacity_Protocol_Sensitivity_PREVIEW.png",
        "Fig_3_generation_manifest.txt",
    ]:
        p = outdir / name
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
