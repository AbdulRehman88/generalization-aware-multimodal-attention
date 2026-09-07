#!/usr/bin/env python
"""
Generate publication-ready deep-learning figures for the revised attention manuscript.

Inputs are the frozen consolidated CSV files in:
  _research_audit/deep_learning_scientific_results_v1

Outputs:
  Fig_4_Internal_Deep_Generalization.{pdf,svg,png}
  Fig_5_BBBD_Deep_Generalization.{pdf,svg,png}
  plus individual panel files for flexible LaTeX composition.

The script never trains or recomputes models. It only reads frozen summary results.
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    from PIL import Image
except Exception:
    Image = None


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = DEFAULT_PROJECT_ROOT / "_research_audit" / "deep_learning_scientific_results_v1"

# Fallback is the standard restrained categorical family used by Matplotlib.
# If an existing manuscript source figure is found, its dominant saturated colors
# are extracted and used instead so the new figures inherit the established palette.
FALLBACK_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
]

PATH_ORDER = [
    "ECG",
    "Pupil",
    "EEG",
    "ECG_Pupil",
    "EEG_Pupil",
    "ECG_EEG",
    "ECG_EEG_Pupil",
]

DISPLAY_PATH = {
    "ECG": "ECG",
    "EEG": "EEG",
    "Pupil": "Pupil",
    "ECG_EEG": "ECG + EEG",
    "ECG_Pupil": "ECG + Pupil",
    "EEG_Pupil": "EEG + Pupil",
    "ECG_EEG_Pupil": "ECG + EEG + Pupil",
}

DURATION_COLORS_FALLBACK = {
    8: "#1f77b4",
    16: "#ff7f0e",
    24: "#2ca02c",
    32: "#d62728",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def require_file(root: Path, name: str) -> Path:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen result file not found: {path}")
    return path


def load_inputs(results_dir: Path) -> dict[str, pd.DataFrame]:
    files = {
        "nested": "02_internal_nested_loso_summary.csv",
        "duration": "02_internal_nested_loso_duration_counts.csv",
        "adaptive": "03_internal_subject_adaptive_summary.csv",
        "within": "04_bbbd_within_experiment_summary.csv",
        "cross": "05_bbbd_cross_experiment_summary.csv",
    }
    return {key: pd.read_csv(require_file(results_dir, name)) for key, name in files.items()}


def validate_inputs(d: dict[str, pd.DataFrame]) -> None:
    nested = d["nested"]
    duration = d["duration"]
    adaptive = d["adaptive"]
    within = d["within"]
    cross = d["cross"]

    assert len(nested) == 8, f"Expected 8 strict LOSO model/path rows, found {len(nested)}"
    assert set(nested["n_operations"].astype(int)) == {39}, "Strict LOSO must contain 39 fold-seed evaluations per model/path."
    assert set(nested["n_splits"].astype(int)) == {13}, "Strict LOSO must contain 13 outer participants."
    assert set(nested["n_seeds"].astype(int)) == {3}, "Strict LOSO must contain three fixed seeds."

    assert set(adaptive["calibration_budget_seconds_per_class"].astype(int)) == {30, 60, 120}
    assert len(adaptive) == 12, f"Expected 12 adaptive rows, found {len(adaptive)}"
    assert set(adaptive["n_operations"].astype(int)) == {39}

    eligible = {
        ("ShallowConvNet", "EEG"),
        ("MultibranchTCN", "EEG"),
        ("MultibranchTCN", "ECG_EEG"),
        ("MultibranchTCN", "ECG"),
    }
    for pair in eligible:
        sub = duration[(duration["model"] == pair[0]) & (duration["path"] == pair[1])]
        assert set(sub["selected_duration_seconds"].astype(int)) == {8, 16, 24, 32}
        assert int(sub["count"].sum()) == 39

    assert set(within["experiment"].astype(str)) == {"experiment2", "experiment3"}
    assert len(within) == 16, f"Expected 16 BBBD within rows, found {len(within)}"
    exp_counts = within.groupby("experiment")["n_splits"].first().astype(int).to_dict()
    assert exp_counts == {"experiment2": 20, "experiment3": 16}, exp_counts

    assert set(cross["split_id"].astype(str)) == {
        "experiment2_to_experiment3",
        "experiment3_to_experiment2",
    }
    assert len(cross) == 16, f"Expected 16 BBBD cross rows, found {len(cross)}"
    assert set(cross["n_operations"].astype(int)) == {3}, "Cross-experiment summaries must contain three seed runs per model/path."

    # Frozen topline spot checks. These catch accidental use of the wrong directory/version.
    best_nested = nested.sort_values(["test_balanced_accuracy_mean", "test_macro_f1_mean"], ascending=False).iloc[0]
    assert best_nested["model"] == "ShallowConvNet" and best_nested["path"] == "EEG"
    assert np.isclose(best_nested["test_balanced_accuracy_mean"], 0.5763557239801583, atol=1e-12)

    a120 = adaptive[adaptive["calibration_budget_seconds_per_class"] == 120].sort_values(
        ["test_balanced_accuracy_mean", "test_macro_f1_mean"], ascending=False
    ).iloc[0]
    assert a120["model"] == "ShallowConvNet" and a120["path"] == "EEG"
    assert np.isclose(a120["test_balanced_accuracy_mean"], 0.7153853203893281, atol=1e-12)


def find_style_source(project_root: Path) -> Path | None:
    candidates = [
        "Fig_8_ablation_accuracy.png",
        "Fig_3_Accuracy.png",
        "Fig_7_end_to_end_latency_by_model.png",
    ]
    if not project_root.exists():
        return None
    for name in candidates:
        found = list(project_root.rglob(name))
        if found:
            return found[0]
    return None


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def extract_palette(image_path: Path, n: int = 7) -> list[str] | None:
    if Image is None or image_path is None or not image_path.is_file():
        return None
    try:
        im = Image.open(image_path).convert("RGB")
        im.thumbnail((900, 900))
        q = im.quantize(colors=96, method=Image.Quantize.MEDIANCUT).convert("RGB")
        counts = q.getcolors(maxcolors=1_000_000) or []
        counts.sort(reverse=True)

        selected: list[tuple[int, int, int]] = []
        for count, rgb in counts:
            r, g, b = [x / 255.0 for x in rgb]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            if s < 0.35 or v < 0.28 or v > 0.93:
                continue
            if max(rgb) - min(rgb) < 35:
                continue
            # Avoid near-duplicate shades from anti-aliasing.
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


def clean_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
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


def model_path_label(model: str, path: str) -> str:
    prefix = "Shallow" if model == "ShallowConvNet" else "TCN"
    return f"{prefix}  {DISPLAY_PATH[path]}"


def model_path_key(model: str, path: str) -> tuple[int, int]:
    # Fixed order allows direct visual comparison across panels.
    model_rank = 1 if model == "ShallowConvNet" else 0
    return (PATH_ORDER.index(path), model_rank)


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")


def draw_internal_loso(ax: plt.Axes, nested: pd.DataFrame, path_colors: dict[str, str]) -> None:
    frame = nested.sort_values("test_balanced_accuracy_mean", ascending=True).reset_index(drop=True)
    y = np.arange(len(frame))
    for i, row in frame.iterrows():
        color = path_colors[row["path"]]
        marker = "s" if row["model"] == "ShallowConvNet" else "o"
        ax.errorbar(
            row["test_balanced_accuracy_mean"],
            i,
            xerr=row["test_balanced_accuracy_std"],
            fmt=marker,
            markersize=5.3,
            color=color,
            ecolor=color,
            elinewidth=1.05,
            capsize=2.2,
            markeredgecolor="white",
            markeredgewidth=0.55,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([model_path_label(r.model, r.path) for r in frame.itertuples()])
    ax.set_ylabel("Model / modality path")
    ax.axvline(1 / 3, color="#6e6e6e", linestyle="--", linewidth=0.9, zorder=1)
    ax.text(1 / 3 + 0.008, len(frame) - 0.35, "3-class chance", color="#666666", fontsize=7.4, rotation=90, va="top")
    ax.set_xlim(0.18, 0.78)
    ax.set_xlabel("Balanced accuracy")
    ax.set_title("Strict nested LOSO")
    clean_axis(ax, "x")


def draw_adaptation(ax: plt.Axes, nested: pd.DataFrame, adaptive: pd.DataFrame, path_colors: dict[str, str]) -> None:
    configs = [
        ("ShallowConvNet", "EEG"),
        ("MultibranchTCN", "ECG_EEG"),
        ("MultibranchTCN", "EEG"),
        ("MultibranchTCN", "ECG"),
    ]
    budgets = [0, 30, 60, 120]
    for model, path in configs:
        b0 = nested[(nested["model"] == model) & (nested["path"] == path)].iloc[0]
        means = [float(b0["test_balanced_accuracy_mean"])]
        stds = [float(b0["test_balanced_accuracy_std"])]
        for budget in budgets[1:]:
            row = adaptive[
                (adaptive["model"] == model)
                & (adaptive["path"] == path)
                & (adaptive["calibration_budget_seconds_per_class"] == budget)
            ].iloc[0]
            means.append(float(row["test_balanced_accuracy_mean"]))
            stds.append(float(row["test_balanced_accuracy_std"]))
        marker = "s" if model == "ShallowConvNet" else "o"
        ax.errorbar(
            budgets,
            means,
            yerr=stds,
            marker=marker,
            markersize=4.6,
            linewidth=1.35,
            capsize=2.1,
            color=path_colors[path],
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=model_path_label(model, path),
        )
    ax.axhline(1 / 3, color="#6e6e6e", linestyle="--", linewidth=0.9)
    ax.set_xticks(budgets, ["0", "30", "60", "120"])
    ax.set_ylim(0.20, 0.90)
    ax.set_xlabel("Calibration budget (s/class)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Subject-adaptive calibration")
    clean_axis(ax, "y")
    ax.legend(frameon=False, ncol=2, loc="upper left", handlelength=1.7, columnspacing=0.9)


def draw_duration(ax: plt.Axes, duration: pd.DataFrame, duration_colors: dict[int, str]) -> None:
    configs = [
        ("ShallowConvNet", "EEG"),
        ("MultibranchTCN", "EEG"),
        ("MultibranchTCN", "ECG_EEG"),
        ("MultibranchTCN", "ECG"),
    ]
    labels = [model_path_label(m, p) for m, p in configs]
    y = np.arange(len(configs))
    left = np.zeros(len(configs), dtype=float)
    for dur in [8, 16, 24, 32]:
        vals = []
        for model, path in configs:
            row = duration[
                (duration["model"] == model)
                & (duration["path"] == path)
                & (duration["selected_duration_seconds"] == dur)
            ]
            count = int(row.iloc[0]["count"])
            vals.append(100.0 * count / 39.0)
        vals = np.asarray(vals)
        ax.barh(y, vals, left=left, height=0.58, color=duration_colors[dur], edgecolor="white", linewidth=0.5, label=f"{dur} s")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylabel("Model / modality path")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Selected outer-fold/seed evaluations (%)")
    ax.set_title("Validation-selected input duration")
    clean_axis(ax, "x")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), handlelength=1.0, columnspacing=0.8)


def make_figure4(d: dict[str, pd.DataFrame], outdir: Path, path_colors: dict[str, str], duration_colors: dict[int, str]) -> None:
    nested = d["nested"]
    adaptive = d["adaptive"]
    duration = d["duration"]

    fig = plt.figure(figsize=(12.6, 7.9), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], width_ratios=[1.03, 0.97], hspace=0.42, wspace=0.42)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    draw_internal_loso(ax_a, nested, path_colors)
    draw_adaptation(ax_b, nested, adaptive, path_colors)
    draw_duration(ax_c, duration, duration_colors)
    panel_label(ax_a, "a")
    panel_label(ax_b, "b")
    panel_label(ax_c, "c")

    fig.subplots_adjust(left=0.15, right=0.985, top=0.955, bottom=0.12)
    save_figure(fig, outdir / "Fig_4_Internal_Deep_Generalization")
    plt.close(fig)

    # Separate panel files for flexible LaTeX composition.
    for tag, drawer, size in [
        ("4a_Strict_Nested_LOSO", lambda ax: draw_internal_loso(ax, nested, path_colors), (6.5, 4.5)),
        ("4b_Subject_Adaptation", lambda ax: draw_adaptation(ax, nested, adaptive, path_colors), (6.4, 4.2)),
        ("4c_Duration_Selection", lambda ax: draw_duration(ax, duration, duration_colors), (6.4, 3.6)),
    ]:
        f, ax = plt.subplots(figsize=size)
        drawer(ax)
        f.tight_layout()
        save_figure(f, outdir / f"Fig_{tag}")
        plt.close(f)


def draw_bbbd_forest(ax: plt.Axes, frame: pd.DataFrame, path_colors: dict[str, str], title: str, cross: bool) -> None:
    frame = frame.copy()
    frame["_order"] = [model_path_key(m, p) for m, p in zip(frame["model"], frame["path"])]
    frame = frame.sort_values("_order", ascending=True).reset_index(drop=True)
    y = np.arange(len(frame))

    for i, row in frame.iterrows():
        color = path_colors[row["path"]]
        marker = "s" if row["model"] == "ShallowConvNet" else "o"
        ax.errorbar(
            row["test_balanced_accuracy_mean"],
            i,
            xerr=row["test_balanced_accuracy_std"],
            fmt=marker,
            markersize=5.0,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.0,
            markeredgecolor="white",
            markeredgewidth=0.5,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([model_path_label(r.model, r.path) for r in frame.itertuples()])
    ax.set_ylabel("Model / modality path")
    ax.axvline(0.5, color="#6e6e6e", linestyle="--", linewidth=0.9)
    ax.set_xlim(0.30, 0.86)
    ax.set_xlabel("Balanced accuracy")
    ax.set_title(title)
    clean_axis(ax, "x")


def make_figure5(d: dict[str, pd.DataFrame], outdir: Path, path_colors: dict[str, str]) -> None:
    within = d["within"]
    cross = d["cross"]

    panels = [
        (within[within["experiment"] == "experiment2"], "Within Experiment 2", False),
        (within[within["experiment"] == "experiment3"], "Within Experiment 3", False),
        (cross[cross["split_id"] == "experiment2_to_experiment3"], "Experiment 2 to Experiment 3", True),
        (cross[cross["split_id"] == "experiment3_to_experiment2"], "Experiment 3 to Experiment 2", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.6), sharex=True)
    axes = axes.ravel()
    for idx, (ax, (frame, title, is_cross)) in enumerate(zip(axes, panels)):
        draw_bbbd_forest(ax, frame, path_colors, title, is_cross)
        panel_label(ax, chr(ord("a") + idx))
    fig.subplots_adjust(left=0.16, right=0.985, top=0.96, bottom=0.08, hspace=0.34, wspace=0.36)
    save_figure(fig, outdir / "Fig_5_BBBD_Deep_Generalization")
    plt.close(fig)

    separate_names = [
        "5a_BBBD_Within_Experiment2",
        "5b_BBBD_Within_Experiment3",
        "5c_BBBD_Cross_Experiment2_to_3",
        "5d_BBBD_Cross_Experiment3_to_2",
    ]
    for name, (frame, title, is_cross) in zip(separate_names, panels):
        f, ax = plt.subplots(figsize=(6.6, 4.5))
        draw_bbbd_forest(ax, frame, path_colors, title, is_cross)
        f.tight_layout()
        save_figure(f, outdir / f"Fig_{name}")
        plt.close(f)


def write_manifest(outdir: Path, results_dir: Path, style_source: Path | None, path_colors: dict[str, str]) -> None:
    lines = [
        "Publication figure generation manifest",
        "",
        f"Results directory: {results_dir}",
        f"Style source: {style_source if style_source else 'not found, fallback palette used'}",
        "",
        "Path palette:",
    ]
    for path in PATH_ORDER:
        lines.append(f"  {path}: {path_colors[path]}")
    lines += [
        "",
        "Figure 4:",
        "  a: strict nested LOSO deep balanced accuracy, mean and descriptive SD across 39 fold-seed evaluations",
        "  b: subject-adaptive deep balanced accuracy at 0, 30, 60, and 120 s/class",
        "  c: validation-selected duration distribution for selection-eligible pupil-free models",
        "",
        "Figure 5:",
        "  a-b: BBBD within-experiment deep balanced accuracy",
        "  c-d: BBBD bidirectional cross-experiment deep balanced accuracy",
        "  Cross-experiment SD is across three fixed seeds and is not an inferential confidence interval.",
        "",
        "No model training or metric recomputation is performed by this script.",
    ]
    (outdir / "figure_generation_manifest.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    project_root = args.project_root.resolve() if args.project_root.exists() else args.project_root
    outdir = (args.output_dir if args.output_dir is not None else results_dir / "paper_figures").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_inputs(results_dir)
    validate_inputs(data)

    configure_style()

    style_source = find_style_source(project_root)
    extracted = extract_palette(style_source, n=7) if style_source else None
    palette = extracted if extracted is not None else FALLBACK_COLORS
    path_colors = {path: palette[i] for i, path in enumerate(PATH_ORDER)}

    # Keep duration colors compact and distinct. If the source palette was extracted,
    # reuse its first four colors so the same visual family persists.
    duration_colors = {
        8: palette[0],
        16: palette[1],
        24: palette[2],
        32: palette[3],
    }

    make_figure4(data, outdir, path_colors, duration_colors)
    make_figure5(data, outdir, path_colors)
    write_manifest(outdir, results_dir, style_source, path_colors)

    print("PASS: frozen result contracts validated.")
    if style_source:
        print(f"Style source: {style_source}")
        print("Palette: extracted from existing manuscript figure." if extracted else "Palette: fallback categorical palette used because extraction was not reliable.")
    else:
        print("Style source not found. Fallback categorical palette used.")
    print(f"Saved publication figures to: {outdir}")
    for p in sorted(outdir.glob("Fig_*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
