#!/usr/bin/env python3
# Copyright 2025-present Niobium Microsystems, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Plaintext-vs-FPGA comparison plots for the NID (Network Monitor) workload.

The report's headline comparison is the **cleartext-CPU ground truth**
(`plaintext_batch*.csv`) against the **FPGA (FHE)** replay (`scores_batch*.csv`),
both written into the run artifact by the harness. A CPU (FHE) reference is an
OPTIONAL third series (`--cpu <csv>`) — it is not required for the comparison.

Labels and the CFAR threshold are sourced independently of any CPU CSV
(dataset label file + tuned `cfar_threshold`), so the comparison works from a
bare FPGA run artifact.

Produces (in <output_dir>):
  score_timelines.png     stacked RMSE score panels (Plaintext / [CPU] / FPGA)
  anomaly_timelines.png   stacked CFAR anomaly-decision panels
  scatter_agreement.png   pairwise agreement vs FPGA (diagonal = perfect)

Usage:
  python3 plot_comparison.py <run_dir> <output_dir> [--profile P]
                             [--cpu cpu_scores.csv] [--cfar X] [--nid-root DIR]
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Reuse the FPGA-report helpers (same scripts/ dir) so labels + CFAR are
# sourced identically across the two report tools.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_fpga_results import load_labels, compute_cfar, tuned_cfar, backend_name  # noqa: E402

# Consistent color palette
COLOR_PLAIN = "#9C27B0"   # purple
COLOR_CPU   = "#00BCD4"   # teal/cyan
COLOR_FPGA  = "#FF9800"   # orange
FHE_SERIES  = "FPGA (FHE)"   # set per-run from run.json target


def load_cpu_scores(cpu_path):
    """Load an optional CPU (FHE) scores CSV (CFAR header + Packet,Score (RMSE),Anomaly,Label)."""
    rmse, labels, cfar = [], [], None
    with open(cpu_path) as f:
        first_line = f.readline().strip()
        if first_line.startswith("CFAR"):
            cfar = float(first_line.split(",")[1])
        for row in csv.DictReader(f):
            rmse.append(float(row["Score (RMSE)"]))
            labels.append(int(row["Label"]))
    return np.array(rmse), cfar, np.array(labels)


def _load_batched(run_dir, prefix):
    """Load per-batch RMSE from <prefix>batch*.csv (Packet,MSE,RMSE)."""
    rmse, batch_idx = [], 0
    while True:
        csv_path = run_dir / f"{prefix}{batch_idx}.csv"
        if not csv_path.exists():
            break
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                rmse.append(float(row["RMSE"]))
        batch_idx += 1
    return np.array(rmse), batch_idx


def plot_score_timelines(series, fpga_rmse, labels, n, output_path,
                         figsize=(14, 8), dpi=150):
    """Stacked score-timeline panels (one per series), shared axes."""
    num_panels = len(series)
    h = figsize[1] * (1.0 + 0.4 * (num_panels - 2))
    fig, axes = plt.subplots(num_panels, 1, figsize=(figsize[0], h),
                             sharex=True, sharey=True)
    if num_panels == 1:
        axes = [axes]

    x = np.arange(n)
    stride = max(1, n // 50000)
    xs = x[::stride]
    has_labels = labels is not None and len(labels) >= n and labels.sum() > 0

    def scatter_panel(ax, scores, color_default, threshold):
        if has_labels:
            lab = labels[:n:stride]
            ax.scatter(xs[lab == 0], scores[:n:stride][lab == 0],
                       s=0.3, alpha=0.4, c="#2196F3", label="Normal", rasterized=True)
            ax.scatter(xs[lab == 1], scores[:n:stride][lab == 1],
                       s=0.3, alpha=0.4, c="#F44336", label="Malicious", rasterized=True)
        else:
            ax.scatter(xs, scores[:n:stride], s=0.3, alpha=0.4, c=color_default,
                       rasterized=True)
        if threshold is not None:
            ax.axhline(y=threshold, color="#4CAF50", linewidth=1.5, linestyle="--",
                       label=f"CFAR = {threshold:.4f}")

    for ax, (label, scores, color, threshold) in zip(axes, series):
        scatter_panel(ax, scores, color, threshold)
        ax.set_title(f"{label} — Anomaly Scores (RMSE)", fontsize=13, fontweight="bold")

    axes[-1].set_xlabel("Packet Index", fontsize=11)
    for ax in axes:
        ax.set_ylabel("RMSE Score", fontsize=11)
        ax.legend(loc="upper right", markerscale=10, fontsize=9)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved score timelines: {output_path}")


def plot_anomaly_timelines(series, fpga_rmse, n, output_path,
                           figsize=(14, 7), dpi=150):
    """Stacked anomaly-decision panels: green=normal, red=anomaly (per-series CFAR)."""
    num_panels = len(series)
    x = np.arange(n)
    stride = max(1, n // 50000)
    xs = x[::stride]
    c_normal, c_anomaly = "#4CAF50", "#F44336"

    h = figsize[1] * (1.0 + 0.4 * (num_panels - 2))
    fig, axes = plt.subplots(num_panels, 1, figsize=(figsize[0], h),
                             sharex=True, sharey=True)
    if num_panels == 1:
        axes = [axes]

    def anom_panel(ax, scores, title_prefix, threshold):
        s = scores[:n:stride]
        anom = s > threshold
        ax.scatter(xs[~anom], s[~anom], s=0.3, alpha=0.4,
                   c=c_normal, label="Normal", rasterized=True)
        ax.scatter(xs[anom], s[anom], s=0.3, alpha=0.4,
                   c=c_anomaly, label="Anomaly", rasterized=True)
        ax.axhline(y=threshold, color="black", linewidth=1, linestyle="--", alpha=0.5)
        n_anom = int(np.sum(scores[:n] > threshold))
        ax.set_title(f"{title_prefix} — Anomaly Decisions ({n_anom:,} / {n:,} anomalous)",
                     fontsize=13, fontweight="bold")

    for ax, (label, scores, _color, threshold) in zip(axes, series):
        anom_panel(ax, scores, label, threshold)

    axes[-1].set_xlabel("Packet Index", fontsize=11)
    for ax in axes:
        ax.set_ylabel("RMSE Score", fontsize=11)
        ax.legend(loc="upper right", markerscale=10, fontsize=9)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved anomaly timelines: {output_path}")


def plot_scatter(ref_series, fpga_rmse, fpga_cfar, n, output_path,
                 figsize=(8, 8), dpi=150):
    """Scatter: each reference series (Plaintext, [CPU]) vs FPGA; diagonal = agreement."""
    num_scatter = len(ref_series)
    if num_scatter == 1:
        fig, ax_list = plt.subplots(figsize=figsize)
        ax_list = [ax_list]
    else:
        fig, ax_list = plt.subplots(1, num_scatter,
                                    figsize=(figsize[0] * num_scatter, figsize[1]))

    stride = max(1, n // 50000)

    def scatter_agreement(ax, x_scores, x_label, x_thresh, title):
        x_s, y_s = x_scores[:n:stride], fpga_rmse[:n:stride]
        xt, yt = x_thresh, fpga_cfar
        both_anom = (x_s > xt) & (y_s > yt)
        both_norm = (x_s <= xt) & (y_s <= yt)
        disagree = ~both_anom & ~both_norm
        ax.scatter(x_s[both_norm], y_s[both_norm], s=0.5, alpha=0.3,
                   c="#4CAF50", label="Both normal", rasterized=True)
        ax.scatter(x_s[both_anom], y_s[both_anom], s=0.5, alpha=0.3,
                   c="#F44336", label="Both anomaly", rasterized=True)
        if disagree.sum() > 0:
            ax.scatter(x_s[disagree], y_s[disagree], s=1.5, alpha=0.6,
                       c="#FF9800", label=f"Disagree ({disagree.sum():,})", rasterized=True)
        lim = max(x_s.max(), y_s.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
        ax.axvline(x=xt, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
        ax.axhline(y=yt, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(f"{FHE_SERIES} RMSE", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", markerscale=8, fontsize=10)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")

    for ax, (label, scores, _color, thr) in zip(ax_list, ref_series):
        scatter_agreement(ax, scores, f"{label} RMSE", thr,
                          f"{label} vs FPGA Score Agreement")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved scatter plot: {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Plaintext-vs-FPGA comparison plots (CPU-FHE optional).")
    ap.add_argument("run_dir", type=Path, help="run artifact dir (scores_batch*/plaintext_batch* CSVs)")
    ap.add_argument("output_dir", type=Path, help="where to write the PNGs")
    ap.add_argument("--profile", default=None, help="profile name for labels (mirai, unit, ...)")
    ap.add_argument("--cpu", type=Path, default=None, help="optional CPU (FHE) scores CSV (third series)")
    ap.add_argument("--cfar", type=float, default=None, help="override CFAR threshold")
    ap.add_argument("--nid-root", type=Path, default=Path(__file__).resolve().parent.parent,
                    help="NID repo root (for label files + profiles.json)")
    args = ap.parse_args()

    run_dir, output_dir = args.run_dir, args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    global FHE_SERIES
    _bn, _bhw, _bnote = backend_name(run_dir)
    FHE_SERIES = f"{_bn} (FHE)"
    if _bnote:
        print(f"\u26a0 NOTE: {_bnote}")
    print(f"Loading {FHE_SERIES} scores...")
    fpga_rmse, num_batches = _load_batched(run_dir, "scores_batch")
    if len(fpga_rmse) == 0:
        sys.exit(f"error: no scores_batch*.csv in {run_dir}")
    print(f"  {len(fpga_rmse):,} packets from {num_batches} batches")

    # Prefer the exact-activation reference (true model, scripts/plaintext_reference.py);
    # fall back to the harness's Chebyshev plaintext_batch (matches the FHE circuit).
    print("Loading plaintext (cleartext) scores...")
    plain_rmse, plain_batches = _load_batched(run_dir, "reference_batch")
    plain_label = "Plaintext (exact)"
    if len(plain_rmse) == 0:
        plain_rmse, plain_batches = _load_batched(run_dir, "plaintext_batch")
        plain_label = "Plaintext"
    if len(plain_rmse) > 0:
        print(f"  {plain_label}: {len(plain_rmse):,} packets from {plain_batches} batches")
    else:
        print("  Not available — run scripts/plaintext_reference.py or the harness")

    cpu_rmse, cpu_cfar, cpu_labels = None, None, None
    if args.cpu:
        print("Loading optional CPU (FHE) scores...")
        cpu_rmse, cpu_cfar, cpu_labels = load_cpu_scores(args.cpu)
        print(f"  {len(cpu_rmse):,} packets, CFAR={cpu_cfar}")

    if len(plain_rmse) == 0 and cpu_rmse is None:
        sys.exit("error: nothing to compare against FPGA — need plaintext_batch*.csv or --cpu")

    # Reference length: shortest available series (packet-aligned).
    lens = [len(fpga_rmse)]
    if len(plain_rmse):
        lens.append(len(plain_rmse))
    if cpu_rmse is not None:
        lens.append(len(cpu_rmse))
    n = min(lens)

    # Labels: dataset label file (independent of any CPU CSV), else CPU CSV's.
    labels = None
    if args.profile:
        labels = load_labels(args.nid_root, args.profile, n, num_batches)
    if (labels is None or len(labels) == 0) and cpu_labels is not None and len(cpu_labels):
        labels = cpu_labels[:n]

    # CFAR: explicit > tuned (run.json / profiles.json) > percentile of NORMAL.
    cfar = args.cfar
    src = "explicit"
    if cfar is None:
        cfar = tuned_cfar(run_dir, args.profile, args.nid_root)
        src = "profile-tuned (cfar_threshold)"
    if cfar is None:
        ref = plain_rmse if len(plain_rmse) else fpga_rmse
        cfar = compute_cfar(ref[:n], 99.9, labels)
        src = "99.9th percentile of NORMAL"
    print(f"CFAR threshold = {cfar:.6f}  [{src}]")

    # Build the series list (plaintext first, CPU optional, FPGA last).
    series = []
    if len(plain_rmse):
        series.append((plain_label, plain_rmse, COLOR_PLAIN, cfar))
    if cpu_rmse is not None:
        series.append(("CPU (FHE)", cpu_rmse, COLOR_CPU, cpu_cfar or cfar))
    series.append((FHE_SERIES, fpga_rmse, COLOR_FPGA, cfar))
    ref_series = [s for s in series if s[0] != FHE_SERIES]

    plot_score_timelines(series, fpga_rmse, labels, n, output_dir / "score_timelines.png")
    plot_anomaly_timelines(series, fpga_rmse, n, output_dir / "anomaly_timelines.png")
    plot_scatter(ref_series, fpga_rmse, cfar, n, output_dir / "scatter_agreement.png")
    print(f"\n✓ Comparison plots -> {output_dir}/")


if __name__ == "__main__":
    main()