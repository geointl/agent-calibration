"""Animate the calibration progress of each round as GIF and MP4.

Left: test data vs. the model response with the best parameters found so far,
one column per test. Right: parameter trajectories, each drawn over its search
bounds, with the true value dashed.

Usage (after `python reproduce.py --cases 1`, or any agent run):
    python src/calibration/post_process/animate_calibration.py logs/reproduce/case1
"""

import argparse
import pickle
import sys
from pathlib import Path

import yaml

import imageio_ffmpeg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.ticker import FuncFormatter, MaxNLocator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.utils import load_test_data

GT_COLOR = "#333333"
LATEX_LABELS = {
    "Gamma": r"$\Gamma$", "lambda": r"$\lambda$", "Mtc": r"$M_{tc}$",
    "N": r"$N$", "chi_tc": r"$\chi_{tc}$", "H0": r"$H_0$",
    "Hy": r"$H_y$", "G_at_p0": r"$G_0$",
}

# Full slide width (13.33 in at 144 dpi = 1920 px), so font sizes are true points on a 16:9 slide;
# 4.75 in tall leaves room for the slide title and footer.
FIGSIZE = (1920 / 144, 684 / 144)
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 15,
    "axes.linewidth": 1.0, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "figure.dpi": 144, "savefig.dpi": 144, "savefig.bbox": "standard",
    "animation.ffmpeg_path": imageio_ffmpeg.get_ffmpeg_exe(),
})

MODEL_COLOR = "#2166ac"
# Fitted channel shown under q, by drainage condition.
SECOND_CHANNEL = {"undrained": ("ue", "$u_e$ (kPa)"), "drained": ("epsV", r"$\varepsilon_v$ (%)")}

ANIMATE_CONFIG = {
    "fps": 6,
    "hold_start": 6,  # extra frames on the initial guess
    "hold_end": 18,   # extra frames on the final fit
}


def _as_list(v):
    return v if isinstance(v, list) else [v]


def animate_round(round_dir, cfg):
    with open(round_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    with open(round_dir / "results.pkl", "rb") as f:
        results = pickle.load(f)
    optimized = {p: spec for sec in config["OptimizationSettings"]["parameters"].values()
                 for p, spec in sec.items() if spec["optimize"]}
    ns = config["NorSandConfig"]
    drainage = _as_list(ns["simulation"]["drainage"])
    soil = ns["material_parameters"]["soil_state"]
    e0, sig0 = _as_list(soil["e0"]), _as_list(soil["sigM0"])
    dfs = [load_test_data(p) for p in _as_list(config["OptimizationSettings"]["data_path"])]

    res = [results[k] for k in sorted(results)]
    n = len(res)
    it = np.arange(n)
    loss = np.array([float(r["data_loss"]) for r in res])
    # pred[eval][test][channel]; multi-test predictions are (points, tests).
    pred = [[{c: (t[:, ti] if t.ndim == 2 else t).numpy() for c, t in r["predictions"].items()}
             for ti in range(len(dfs))] for r in res]
    names = list(optimized)
    theta = {p: np.array([float(r["material_params"][p]) for r in res]) for p in names}
    gt = {p: spec["ground_truth"] for p, spec in optimized.items() if spec.get("ground_truth") is not None}

    # L-BFGS-B logs every function evaluation, including rejected line-search trials;
    # follow the best estimate so far so the animated fit never jumps to a rejected trial.
    best = np.zeros(n, dtype=int)
    for k in range(1, n):
        best[k] = k if loss[k] < loss[best[k - 1]] else best[k - 1]
    assert np.all(np.diff(loss[best]) <= 0)

    fig = plt.figure(figsize=FIGSIZE)
    h = FIGSIZE[1]  # margins below are set in inches: 1.0 on top (legend, test titles), 0.6 at the bottom
    outer = fig.add_gridspec(1, 2, width_ratios=[1, 1.85], left=0.068, right=0.989,
                             top=1 - 1.0 / h, bottom=0.6 / h, wspace=0.16)
    resp = outer[0].subgridspec(2, len(dfs), hspace=0.05, wspace=0.75)

    # Fit panels: one column per test, q on top and the drainage-specific channel below.
    curves = []
    for ti, df in enumerate(dfs):
        c2, label2 = SECOND_CHANNEL[drainage[ti]]
        ax_top = None
        for row, (c, ylabel) in enumerate((("q", "$q$ (kPa)"), (c2, label2))):
            ax = fig.add_subplot(resp[row, ti], sharex=ax_top)
            ax_top = ax_top or ax
            ax.plot(df["eps1"], df[c], "o", ms=6, mfc="none", mec="k", mew=1.2, markevery=0.04,
                    label="Test data")
            ax.plot(pred[0][ti]["eps1"], pred[0][ti][c], ":", color="0.55", lw=2, label="Initial guess")
            curves.append((ax.plot([], [], color=MODEL_COLOR, lw=2.8, label="Model / parameters: best so far")[0],
                           ti, c))
            vals = np.concatenate([df[c].to_numpy()] + [pred[b][ti][c] for b in set(best)])
            ax.set(xlim=(0, df["eps1"].max()), ylim=(1.08 * min(0, vals.min()), 1.08 * vals.max()),
                   ylabel=ylabel)
        ax_top.tick_params(labelbottom=False)
        ax_top.set_title(f"Test {ti + 1}: {drainage[ti]}\n"
                         rf"$e_0$ = {e0[ti]:.2f}, $\sigma'_{{m0}}$ = {sig0[ti]:g} kPa", fontsize=13)
        ax.set_xlabel(r"Axial strain $\varepsilon_1$ (%)")

    # Parameter panels in 2 rows sharing one iteration axis; y-range = search bounds,
    # so the gap to the truth reads the same in every panel.
    ncol = -(-len(names) // 2)
    gr = outer[1].subgridspec(2, ncol, hspace=0.27, wspace=0.43)
    traj, dots, pax = [], [], []
    for i, p in enumerate(names):
        ax = fig.add_subplot(gr[i // ncol, i % ncol], sharex=pax[0] if pax else None)
        pax.append(ax)
        lo, hi = optimized[p]["bounds"]
        if p in gt:
            ax.axhline(gt[p], color=GT_COLOR, ls="--", lw=1.6, label="True value")
        traj.append(ax.plot([], [], color=MODEL_COLOR, lw=2.6)[0])
        dots.append(ax.plot([], [], "o", color=MODEL_COLOR, ms=7, clip_on=False)[0])
        ax.set(xlim=(0, n - 1), ylim=(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)))
        # Short tick labels fit the narrow column gaps: no 2.5 steps (0.025), and G0 as '100k'.
        ax.yaxis.set_major_locator(MaxNLocator("auto", steps=[1, 2, 5, 10]))
        if hi >= 1e4:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e3:g}k"))
        ax.set_title(LATEX_LABELS.get(p, p), loc="left", fontsize=17)
        if i < ncol:
            ax.tick_params(labelbottom=False)
    left, right = pax[-ncol].get_position(), pax[-1].get_position()
    fig.text((left.x0 + right.x1) / 2, left.y0 - 0.3 / h, "Iteration", ha="center", va="top", fontsize=16)

    handles = fig.axes[0].get_lines() + ([ax.get_lines()[0]] if gt else [])
    fig.legend(handles, [ln.get_label() for ln in handles], loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, frameon=False, handlelength=2.4, columnspacing=1.6)

    def update(k):
        b = best[k]
        for line, ti, c in curves:
            line.set_data(pred[b][ti]["eps1"], pred[b][ti][c])
        for p, tr, dot in zip(names, traj, dots):
            v = theta[p][best[:k + 1]]
            tr.set_data(it[:k + 1], v)
            dot.set_data([k], [v[-1]])

    frames = [0] * cfg["hold_start"] + list(range(n)) + [n - 1] * cfg["hold_end"]
    anim = FuncAnimation(fig, update, frames=frames)
    out = round_dir / "figures" / "calibration_progress"
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out.with_suffix(".gif"), writer=PillowWriter(fps=cfg["fps"]))
    # Default x264 quality (crf 23) leaves faint ghosts beside the axis lines; crf 15 keeps line art clean.
    anim.save(out.with_suffix(".mp4"),
              writer=FFMpegWriter(fps=cfg["fps"], extra_args=["-crf", "15", "-tune", "animation"]))
    plt.close(fig)
    print(f"Animation ({len(frames)} frames) -> {out}.gif / .mp4")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case_dir", help="directory holding round_1/, round_2/, ... of one run")
    args = parser.parse_args()
    for round_dir in sorted(Path(args.case_dir).glob("round_*")):  # outputs -> <round_dir>/figures/
        animate_round(round_dir, ANIMATE_CONFIG)


if __name__ == "__main__":
    main()
