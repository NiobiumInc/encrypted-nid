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
"""
Niobium FHE Demo Report
=======================
Reads a completed run archive and produces:
  1. Terminal summary (benchmarking + ML stats)
  2. Three PNG plots saved to <run_dir>/plots/

Usage:
    python3 scripts/demo_report.py runs/latest/ --profile mirai
    python3 scripts/demo_report.py runs/20260403_152830_mirai/ --profile mirai
"""

import argparse
import csv
import json
import socket
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


BACKEND = "FPGA"   # series label; set per-run from run.json's target (honest labeling)

def backend_name(run_dir):
    """(short_name, is_hardware, note) from run.json's target so a simulator run is
    never labeled 'FPGA'. FOG / FPGA_* -> hardware; local -> Simulator; *SIM* -> Software."""
    import json
    from pathlib import Path as _P
    target = ""
    rj = _P(run_dir) / "run.json"
    if rj.exists():
        try:
            target = str(json.loads(rj.read_text()).get("target", "")).strip()
        except Exception:
            target = ""
    t = target.upper()
    if t == "FOG" or t.startswith("FPGA"):
        return "FPGA", True, None
    if t == "LOCAL":
        return "Simulator", False, "simulator run (target=local) — in-process, NOT FPGA hardware"
    if "SIM" in t:
        return "Software", False, f"software-sim run (target={target}) — NOT FPGA hardware"
    if not target:
        return "FHE", False, "backend not recorded in run.json"
    return target, False, f"non-hardware run (target={target})"


BATCH_SIZE = 32768

LABEL_FILES = {
    "mirai":      "Datasets/Mirai/Mirai_labels.csv",
    "unit":       "Datasets/Unit/Unit_labels.csv",
}

WARMUP_SKIP = {
    "mirai":      60810,
    "unit":       25000,
}


# ==============================================================================
# Data loading
# ==============================================================================

def load_score_csv(path: Path):
    """Load Packet,MSE,RMSE CSV. Returns (rmse_array, mse_array)."""
    mse, rmse = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mse.append(float(row["MSE"]))
            rmse.append(float(row["RMSE"]))
    return np.array(rmse), np.array(mse)


def load_all_batches(run_dir: Path, prefix: str):
    """Load all scores_batch*.csv or plaintext_batch*.csv from run_dir."""
    all_rmse, all_mse = [], []
    batch_idx = 0
    while True:
        path = run_dir / f"{prefix}_batch{batch_idx}.csv"
        if not path.exists():
            break
        rmse, mse = load_score_csv(path)
        all_rmse.append(rmse)
        all_mse.append(mse)
        batch_idx += 1
    return all_rmse, all_mse


def load_labels(nid_root: Path, profile: str, num_batches: int):
    """Load ground-truth labels aligned to batch boundaries."""
    label_file = LABEL_FILES.get(profile)
    if not label_file:
        return None
    label_path = nid_root / label_file
    if not label_path.exists():
        return None

    warmup = WARMUP_SKIP.get(profile, 0)
    all_labels = []
    with open(label_path) as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 2:
                all_labels.append(int(row[1]))

    start = warmup
    end = warmup + num_batches * BATCH_SIZE
    labels = all_labels[start:end]
    if len(labels) < num_batches * BATCH_SIZE:
        labels.extend([0] * (num_batches * BATCH_SIZE - len(labels)))
    return np.array(labels[:num_batches * BATCH_SIZE])


def load_metadata(run_dir: Path):
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def find_timing_summary(nid_root: Path):
    """Find timing_summary.json in the most recent workload directory."""
    candidates = sorted(
        nid_root.glob("NID_Mirai_*workload*/timing_summary.json"),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    if candidates:
        with open(candidates[0]) as f:
            return json.load(f)
    return {}


def load_cfar_from_summary(run_dir: Path):
    """Read CFAR threshold from anomaly_detection_summary.txt if present."""
    summary_path = run_dir / "anomaly_detection_summary.txt"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        for line in f:
            if "CFAR threshold:" in line:
                try:
                    return float(line.split(":")[-1].strip())
                except ValueError:
                    pass
    return None


# ==============================================================================
# Stats
# ==============================================================================

def compute_detection_stats(fpga_rmse_all, labels, cfar_threshold):
    """Compute TP/FP/TN/FN and derived metrics."""
    decisions = (fpga_rmse_all >= cfar_threshold).astype(int)
    if labels is None or len(labels) != len(decisions):
        return None
    tp = int(np.sum((decisions == 1) & (labels == 1)))
    fp = int(np.sum((decisions == 1) & (labels == 0)))
    tn = int(np.sum((decisions == 0) & (labels == 0)))
    fn = int(np.sum((decisions == 0) & (labels == 1)))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tpr": tpr, "fpr": fpr, "precision": precision, "f1": f1}


def compute_per_batch_error(fpga_rmse_batches, plain_rmse_batches):
    """Compute mean relative error between FPGA and plaintext RMSE per batch."""
    errors = []
    for fpga, plain in zip(fpga_rmse_batches, plain_rmse_batches):
        mask = plain > 0
        if mask.sum() == 0:
            errors.append(0.0)
        else:
            rel_err = np.abs(fpga[mask] - plain[mask]) / np.abs(plain[mask]) * 100.0
            errors.append(float(np.mean(rel_err)))
    return errors


# ==============================================================================
# Terminal report
# ==============================================================================

def print_report(run_dir, profile, fpga_rmse_batches, plain_rmse_batches,
                 labels, cfar_threshold, meta, per_path_cfar=False):
    n_batches = len(fpga_rmse_batches)
    n_packets = n_batches * BATCH_SIZE
    fpga_all = np.concatenate(fpga_rmse_batches)
    plain_all = np.concatenate(plain_rmse_batches) if plain_rmse_batches else None

    # CFAR threshold is pre-calibrated from training data (profiles.json),
    # applied uniformly to both FPGA and plaintext — matching real deployment
    # where the threshold is fixed at calibration time, not recomputed per batch.
    # CI nightly validates with independent per-path thresholds across 688K
    # packets: CPU/FPGA all compute 0.064370, plaintext 0.065384.
    per_batch_anomalies = [int(np.sum(b >= cfar_threshold)) for b in fpga_rmse_batches]
    per_batch_errors = compute_per_batch_error(fpga_rmse_batches, plain_rmse_batches) if plain_rmse_batches else []

    sep = "=" * 68

    print()
    print(sep)
    print("  Niobium FHE Demo — Run Report")
    print(f"  Profile: {profile.upper()}   Batches: {n_batches}   "
          f"Packets: {n_packets:,}")
    print(f"  Run: {run_dir.name}")
    print(sep)

    # --- Benchmarking ---
    exec_info = meta.get("execution", {})
    ts = meta.get("_timing_summary", {})
    instr = exec_info.get("total_instructions", "N/A")
    total = exec_info.get("total_pipeline_sec", "N/A")

    if not ts:
        # No structured timing — show harness-level wall clocks
        print()
        print("  BENCHMARKING")
        print("  " + "-" * 40)
        rec = exec_info.get("recording_time_sec", "N/A")
        rep = exec_info.get("replay_time_sec", "N/A")
        if isinstance(rec, (int, float)):
            print(f"  Recording time:    {rec}s")
        if isinstance(rep, (int, float)):
            print(f"  Replay time:       {rep}s/batch")
        if isinstance(total, (int, float)):
            mins = int(total) // 60
            secs = int(total) % 60
            print(f"  Total pipeline:    {total}s (~{mins}m {secs}s)")

    # Rich timing from timing_summary.json (already fetched above)
    if ts:
        print()
        print("  BENCHMARKING")
        print("  " + "-" * 40)
        setup_ms    = ts.get("setup_ms")
        record_ms   = ts.get("record_ms")
        replay_ms   = ts.get("replay_ms")
        verify_ms   = ts.get("verify_ms")
        phases      = ts.get("replay_phases")
        fpga_ms     = ts.get("fpga_ms")
        dma_w       = ts.get("dma_write_ms")
        cfg_exec    = ts.get("configure_and_execute_ms")
        dma_r       = ts.get("dma_read_ms")
        compile_ms  = ts.get("compile_ms")
        serial_ms   = ts.get("serialize_ms")
        load_ms     = ts.get("file_loading_ms")
        bin_write   = ts.get("binary_write_ms")
        retrieve_ms = ts.get("retrieval_ms")

        def _fmt_ms(ms):
            """Format milliseconds as human-readable time (always in seconds)."""
            if ms >= 60000:
                return f"{ms/1000:.1f}s ({ms/60000:.1f}m)"
            else:
                return f"{ms/1000:.1f}s"

        if isinstance(total, (int, float)):
            mins = int(total) // 60
            secs = int(total) % 60
            print(f"  Pipeline wall:           {_fmt_ms(total * 1000):>18}")
        print()
        if setup_ms:
            print(f"  Setup (CC + keys):       {_fmt_ms(setup_ms):>18}")
        if record_ms:
            print(f"  Record (trace):          {_fmt_ms(record_ms):>18}")
        if compile_ms:
            print(f"  Compile (optimization):  {_fmt_ms(compile_ms):>18}")
        if serial_ms:
            print(f"  Serialize (binary):      {_fmt_ms(serial_ms):>18}")
        if replay_ms:
            if phases:
                phase_note = f"  ({phases} batch{'es' if phases != 1 else ''})"
            else:
                phase_note = ""
            print(f"  Replay:{phase_note:17s} {_fmt_ms(replay_ms):>18}")
        if load_ms:
            print(f"    File loading:          {_fmt_ms(load_ms):>18}")
        if dma_w:
            print(f"    DMA write (inputs):    {_fmt_ms(dma_w):>18}")
        if cfg_exec:
            print(f"    Configure + execute:   {_fmt_ms(cfg_exec):>18}")
        if fpga_ms:
            print(f"      FHE (firmware):      {_fmt_ms(fpga_ms):>18}")
        if dma_r:
            print(f"    DMA read (outputs):    {_fmt_ms(dma_r):>18}")
        if bin_write:
            print(f"    Binary write:          {_fmt_ms(bin_write):>18}")
        if retrieve_ms:
            print(f"    Result retrieval:      {_fmt_ms(retrieve_ms):>18}")
        if verify_ms:
            print(f"  Verify:                  {_fmt_ms(verify_ms):>18}")

        # Visual separator before pipeline overhead
        if record_ms or compile_ms or serial_ms:
            print("  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄")

        # Show harness-measured pipeline overhead (keygen, encrypt, decrypt).
        # Read from metadata.json if available, otherwise fall back to subtraction.
        keygen_ms  = meta.get("execution", {}).get("keygen_ms")
        encrypt_ms = meta.get("execution", {}).get("encrypt_ms")
        decrypt_ms = meta.get("execution", {}).get("decrypt_ms")

        if any(v and v > 0 for v in [keygen_ms, encrypt_ms, decrypt_ms]):
            print(f"  Pipeline overhead:")
            if keygen_ms and keygen_ms > 0:
                print(f"    Key generation:        {_fmt_ms(keygen_ms):>18}")
            if encrypt_ms and encrypt_ms > 0:
                print(f"    Encryption:            {_fmt_ms(encrypt_ms):>18}")
            if decrypt_ms and decrypt_ms > 0:
                print(f"    Decryption + verify:   {_fmt_ms(decrypt_ms):>18}")
            # Show remaining unaccounted time
            if isinstance(total, (int, float)):
                measured_ms = sum(v or 0 for v in [
                    setup_ms, record_ms, compile_ms, serial_ms, replay_ms, verify_ms,
                    keygen_ms, encrypt_ms, decrypt_ms])
                remainder_ms = total * 1000 - measured_ms
                if remainder_ms > 500:
                    print(f"    Other (host overhead):  {_fmt_ms(remainder_ms):>18}")
        elif isinstance(total, (int, float)):
            # Fallback: no per-stage breakdown, show lumped overhead
            measured_ms = sum(v or 0 for v in [
                setup_ms, record_ms, compile_ms, serial_ms, replay_ms, verify_ms])
            overhead_ms = total * 1000 - measured_ms
            if overhead_ms > 500:
                print(f"  Other (keygen/enc/dec):  {_fmt_ms(overhead_ms):>18}")

        print()
        print("  Memory Usage")
        print("  " + "-" * 40)
        def _fmt_mb(mb):
            if mb >= 1024:
                return f"{mb/1024:.1f} GB"
            return f"{mb} MB"

        rec_mb    = ts.get("record_peak_mb")
        setup_mb  = ts.get("setup_peak_mb")
        sys_mb    = ts.get("system_peak_mb")
        child_mb  = ts.get("replay_child_peak_mb")
        load_mb   = ts.get("replay_child_load_mb")
        if setup_mb is not None and setup_mb > 0:
            print(f"  Setup peak RSS:          {_fmt_mb(setup_mb):>12}")
        if rec_mb is not None and rec_mb > 0:
            print(f"  Recording peak RSS:      {_fmt_mb(rec_mb):>12}")
        if load_mb is not None and load_mb > 0:
            print(f"  Replay child load RSS:   {_fmt_mb(load_mb):>12}")
        if child_mb is not None and child_mb > 0:
            print(f"  Replay child peak RSS:   {_fmt_mb(child_mb):>12}")
        if sys_mb is not None and sys_mb > 0:
            print(f"  System peak RSS:         {_fmt_mb(sys_mb):>12}")

    # --- FHE accuracy ---
    print()
    print("  FHE ACCURACY  (FPGA vs plaintext baseline)")
    print("  " + "-" * 40)
    if per_batch_errors:
        mean_err = np.mean(per_batch_errors)
        max_err = np.max(per_batch_errors)
        print(f"  Mean relative error (all packets): {mean_err:.4f}%")
        print(f"  Max  relative error (worst batch): {max_err:.4f}%")
        print()
        print(f"  {'Batch':<8} {BACKEND + ' mean RMSE':>16} {'Plain mean RMSE':>16} {'Error':>8}  {'Anomalies':>12}")
        print("  " + "-" * 66)
        for i, (fb, pb, err) in enumerate(zip(fpga_rmse_batches,
                                               plain_rmse_batches,
                                               per_batch_errors)):
            print(f"  {i:<8} {np.mean(fb):>16.6f} {np.mean(pb):>16.6f} "
                  f"{err:>7.4f}%  "
                  f"{per_batch_anomalies[i]:>6,}/{BATCH_SIZE:,} "
                  f"({per_batch_anomalies[i]/BATCH_SIZE*100:.1f}%)")
    else:
        print("  No plaintext scores found — skipping error analysis")
        print()
        print(f"  {'Batch':<8} {BACKEND + ' mean RMSE':>16} {'Anomalies':>20}")
        print("  " + "-" * 48)
        for i, fb in enumerate(fpga_rmse_batches):
            print(f"  {i:<8} {np.mean(fb):>16.6f}  "
                  f"{per_batch_anomalies[i]:>6,}/{BATCH_SIZE:,} "
                  f"({per_batch_anomalies[i]/BATCH_SIZE*100:.1f}%)")

    # --- ML detection ---
    print()
    print("  ML DETECTION QUALITY")
    print("  " + "-" * 40)

    # Determine thresholds: default uses pre-calibrated (deployment-correct).
    # --per-path-cfar: each path uses its own 99.9th pct of normal scores.
    fpga_thresh = cfar_threshold
    plain_thresh = cfar_threshold
    if per_path_cfar and labels is not None and len(labels) == len(fpga_all):
        fpga_normal = fpga_all[labels == 0]
        if len(fpga_normal) > 0:
            fpga_thresh = float(np.percentile(fpga_normal, 99.9))
        if plain_all is not None and len(plain_all) == len(labels):
            plain_normal = plain_all[labels == 0]
            if len(plain_normal) > 0:
                plain_thresh = float(np.percentile(plain_normal, 99.9))

    if per_path_cfar:
        print(f"  CFAR threshold (independent per-path, 99.9th pct of normal traffic):")
        print(f"    {BACKEND} FHE:        {fpga_thresh:.6f}")
        if plain_all is not None:
            print(f"    Plaintext:       {plain_thresh:.6f}")
        print(f"    Reference:       {cfar_threshold:.6f}  (profiles.json)")
    else:
        print(f"  CFAR threshold:    {cfar_threshold:.6f}")
    print(f"  Total packets:     {n_packets:,}")

    anomalies = int(np.sum(fpga_all >= fpga_thresh))
    print()
    print(f"  {'':28} {BACKEND + " FHE":>12} {'Plaintext':>12}")
    print("  " + "-" * 55)
    plain_anomalies = int(np.sum(plain_all >= plain_thresh)) if plain_all is not None else None
    print(f"  {'Anomalies flagged:':<28} {anomalies:>12,} {plain_anomalies:>12,}" if plain_anomalies is not None
          else f"  {'Anomalies flagged:':<28} {anomalies:>12,} {'N/A':>12}")

    det = compute_detection_stats(fpga_all, labels, fpga_thresh)
    det_plain = compute_detection_stats(plain_all, labels, plain_thresh) if plain_all is not None and labels is not None else None

    if det and det_plain:
        print(f"  {'True Positives:':<28} {det['tp']:>12,} {det_plain['tp']:>12,}")
        print(f"  {'False Positives:':<28} {det['fp']:>12,} {det_plain['fp']:>12,}")
        print(f"  {'True Negatives:':<28} {det['tn']:>12,} {det_plain['tn']:>12,}")
        print(f"  {'False Negatives:':<28} {det['fn']:>12,} {det_plain['fn']:>12,}")
        print()
        print(f"  {'TPR (Recall):':<28} {det['tpr']:>11.4f}  {det_plain['tpr']:>11.4f}")
        print(f"  {'Precision:':<28} {det['precision']:>11.4f}  {det_plain['precision']:>11.4f}")
        print(f"  {'FPR:':<28} {det['fpr']:>11.4f}  {det_plain['fpr']:>11.4f}")
        print(f"  {'F1 Score:':<28} {det['f1']:>11.4f}  {det_plain['f1']:>11.4f}")
    elif det:
        print(f"  {'True Positives:':<28} {det['tp']:>12,}")
        print(f"  {'False Positives:':<28} {det['fp']:>12,}")
        print(f"  {'True Negatives:':<28} {det['tn']:>12,}")
        print(f"  {'False Negatives:':<28} {det['fn']:>12,}")
        print()
        print(f"  {'TPR (Recall):':<28} {det['tpr']:>11.4f}")
        print(f"  {'Precision:':<28} {det['precision']:>11.4f}")
        print(f"  {'FPR:':<28} {det['fpr']:>11.4f}")
        print(f"  {'F1 Score:':<28} {det['f1']:>11.4f}")
    else:
        print("  (No ground-truth labels available — skipping TPR/FPR)")

    # Decision agreement — each path uses its own threshold
    if plain_all is not None:
        fpga_decisions = fpga_all >= fpga_thresh
        plain_decisions = plain_all >= plain_thresh
        agree = int(np.sum(fpga_decisions == plain_decisions))
        disagree = n_packets - agree
        print()
        print(f"  {BACKEND} vs Plaintext Agreement")
        if per_path_cfar:
            print(f"  (each path uses its own independently-computed threshold)")
        print("  " + "-" * 40)
        print(f"  Matching decisions:  {agree:,}/{n_packets:,}  ({agree/n_packets*100:.2f}%)")
        print(f"  Disagreements:       {disagree:,}  ({disagree/n_packets*100:.4f}%)")

    print()
    print(sep)


# ==============================================================================
# Plots
# ==============================================================================

def make_plots(run_dir, profile, fpga_rmse_batches, plain_rmse_batches,
               labels, cfar_threshold, fpga_thresh=None, plain_thresh=None):
    if not HAS_MATPLOTLIB:
        print("  matplotlib not available — skipping plots")
        return

    # Default: both paths use the same pre-calibrated threshold
    if fpga_thresh is None:
        fpga_thresh = cfar_threshold
    if plain_thresh is None:
        plain_thresh = cfar_threshold

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    n_batches = len(fpga_rmse_batches)
    fpga_all = np.concatenate(fpga_rmse_batches)
    plain_all = np.concatenate(plain_rmse_batches) if plain_rmse_batches else None
    n_total = len(fpga_all)

    # Downsample to ~50K points for dense scatter (same as plot_comparison.py)
    stride = max(1, n_total // 50000)
    xs = np.arange(n_total)[::stride]
    fpga_ds = fpga_all[::stride]
    plain_ds = plain_all[::stride] if plain_all is not None else None
    labels_ds = labels[::stride] if labels is not None and len(labels) == n_total else None

    # Colors matching repo style
    C_NORMAL  = "#2196F3"   # blue
    C_MAL     = "#F44336"   # red
    C_CFAR    = "#4CAF50"   # green
    C_PLAIN   = "#FF9800"   # orange

    # ------------------------------------------------------------------
    # Fig 1: FPGA scores colored by label (normal=blue, malicious=red)
    # ------------------------------------------------------------------
    n_panels = 2 if plain_ds is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    def scatter_by_label(ax, x, scores, lbl, title, thresh):
        if lbl is not None:
            nm = lbl == 0
            ml = lbl == 1
            ax.scatter(x[nm], scores[nm], s=0.3, alpha=0.4, c=C_NORMAL,
                       label="Normal", rasterized=True)
            ax.scatter(x[ml], scores[ml], s=0.3, alpha=0.4, c=C_MAL,
                       label="Malicious", rasterized=True)
        else:
            ax.scatter(x, scores, s=0.3, alpha=0.4, c="steelblue", rasterized=True)
        ax.axhline(thresh, color=C_CFAR, linewidth=1.5, linestyle="--",
                   label=f"CFAR = {thresh:.4f}")
        ax.set_yscale("log")
        ax.set_ylabel("RMSE Score (log)")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8, markerscale=5)

    if plain_ds is not None:
        scatter_by_label(axes[0], xs, plain_ds, labels_ds,
                         f"Plaintext Baseline — {profile.upper()} ({n_batches} batches)",
                         plain_thresh)
        scatter_by_label(axes[1], xs, fpga_ds, labels_ds,
                         f"{BACKEND} FHE (encrypted computation)", fpga_thresh)
    else:
        scatter_by_label(axes[0], xs, fpga_ds, labels_ds,
                         f"{BACKEND} FHE Anomaly Scores — {profile.upper()} ({n_batches} batches)",
                         fpga_thresh)

    axes[-1].set_xlabel("Packet index")
    plt.tight_layout()
    out = plots_dir / "fig1_score_timeline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    # ------------------------------------------------------------------
    # Fig 2: Anomaly detection decisions scatter (colored by TP/FP/TN/FN)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))

    decisions_ds = (fpga_ds >= fpga_thresh)

    if labels_ds is not None:
        tp_mask = decisions_ds & (labels_ds == 1)
        fp_mask = decisions_ds & (labels_ds == 0)
        fn_mask = ~decisions_ds & (labels_ds == 1)
        tn_mask = ~decisions_ds & (labels_ds == 0)
        ax.scatter(xs[tp_mask], fpga_ds[tp_mask], s=0.3, alpha=0.5,
                   c=C_MAL, label="True Positive", rasterized=True)
        ax.scatter(xs[fp_mask], fpga_ds[fp_mask], s=1.0, alpha=0.8,
                   c="purple", label="False Positive", rasterized=True)
        ax.scatter(xs[fn_mask], fpga_ds[fn_mask], s=0.3, alpha=0.5,
                   c=C_PLAIN, label="False Negative", rasterized=True)
        ax.scatter(xs[tn_mask], fpga_ds[tn_mask], s=0.3, alpha=0.3,
                   c=C_NORMAL, label="True Negative", rasterized=True)
    else:
        ax.scatter(xs[decisions_ds], fpga_ds[decisions_ds], s=0.3, alpha=0.5,
                   c=C_MAL, label="Anomaly", rasterized=True)
        ax.scatter(xs[~decisions_ds], fpga_ds[~decisions_ds], s=0.3, alpha=0.3,
                   c=C_NORMAL, label="Normal", rasterized=True)

    ax.axhline(fpga_thresh, color=C_CFAR, linewidth=1.5, linestyle="--",
               label=f"CFAR = {fpga_thresh:.4f}")
    ax.set_yscale("log")
    ax.set_ylabel("RMSE Score (log)")
    ax.set_xlabel("Packet index")
    ax.set_title(f"Anomaly Detection Decisions — {profile.upper()}")
    ax.legend(loc="upper left", fontsize=8, markerscale=5)

    plt.tight_layout()
    out = plots_dir / "fig2_detection_timeline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    # ------------------------------------------------------------------
    # Fig 3: Scatter — Plaintext vs FPGA scores (agreement plot)
    # ------------------------------------------------------------------
    if plain_ds is not None:
        fig, ax = plt.subplots(figsize=(8, 8))

        lim = max(fpga_ds.max(), plain_ds.max()) * 1.05

        if labels_ds is not None:
            # Color by agreement: TP=red, TN=blue, FP=purple, FN=orange
            d = fpga_ds >= fpga_thresh
            p = plain_ds >= plain_thresh
            tp = d & (labels_ds == 1)
            tn = ~d & (labels_ds == 0)
            fp = d & (labels_ds == 0)
            fn = ~d & (labels_ds == 1)
            ax.scatter(plain_ds[tn], fpga_ds[tn], s=0.5, alpha=0.3,
                       c=C_NORMAL, label="True Negative", rasterized=True)
            ax.scatter(plain_ds[tp], fpga_ds[tp], s=0.5, alpha=0.3,
                       c=C_MAL, label="True Positive", rasterized=True)
            ax.scatter(plain_ds[fp], fpga_ds[fp], s=2.0, alpha=0.8,
                       c="purple", label="False Positive", rasterized=True)
            ax.scatter(plain_ds[fn], fpga_ds[fn], s=2.0, alpha=0.8,
                       c=C_PLAIN, label="False Negative", rasterized=True)
        else:
            ax.scatter(plain_ds, fpga_ds, s=0.5, alpha=0.3,
                       c="steelblue", rasterized=True)

        ax.plot([1e-4, lim], [1e-4, lim], "k--", linewidth=1,
                label="y = x (perfect match)")
        ax.axhline(fpga_thresh, color=C_CFAR, linewidth=1.0,
                   linestyle=":", alpha=0.7, label=f"CFAR({BACKEND}) = {fpga_thresh:.4f}")
        ax.axvline(plain_thresh, color=C_PLAIN, linewidth=1.0,
                   linestyle=":", alpha=0.7, label=f"CFAR(Plain) = {plain_thresh:.4f}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Plaintext RMSE")
        ax.set_ylabel(f"{BACKEND} FHE RMSE")
        ax.set_title(f"Plaintext vs FPGA FHE Scores — {profile.upper()}")
        ax.legend(fontsize=8, markerscale=5)

        corr = float(np.corrcoef(np.log1p(plain_all), np.log1p(fpga_all))[0, 1])
        ax.text(0.05, 0.95, f"Pearson r = {corr:.6f} (log scale)",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.tight_layout()
        out = plots_dir / "fig3_plaintext_vs_fpga_scatter.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")

    # ------------------------------------------------------------------
    # Fig 4: ROC curve — FPGA FHE vs Plaintext (if labels available)
    # ------------------------------------------------------------------
    if labels is not None and len(labels) == n_total and plain_all is not None:
        fig, ax = plt.subplots(figsize=(7, 7))

        def compute_roc(scores, labels):
            thresholds = np.percentile(scores, np.linspace(0, 100, 500))
            thresholds = np.unique(thresholds)[::-1]
            tprs, fprs = [], []
            pos = np.sum(labels == 1)
            neg = np.sum(labels == 0)
            for t in thresholds:
                d = scores >= t
                tprs.append(np.sum(d & (labels == 1)) / pos if pos > 0 else 0)
                fprs.append(np.sum(d & (labels == 0)) / neg if neg > 0 else 0)
            return np.array(fprs), np.array(tprs)

        fpga_fpr, fpga_tpr = compute_roc(fpga_all, labels)
        plain_fpr, plain_tpr = compute_roc(plain_all, labels)

        # AUC via trapezoidal integration
        fpga_auc = float(np.trapezoid(fpga_tpr, fpga_fpr) if hasattr(np, 'trapezoid') else np.trapz(fpga_tpr, fpga_fpr))
        plain_auc = float(np.trapezoid(plain_tpr, plain_fpr) if hasattr(np, 'trapezoid') else np.trapz(plain_tpr, plain_fpr))

        ax.plot(fpga_fpr, fpga_tpr, color=C_MAL, linewidth=2,
                label=f"FPGA FHE  (AUC = {abs(fpga_auc):.3f})")
        ax.plot(plain_fpr, plain_tpr, color=C_NORMAL, linewidth=2,
                linestyle="--", label=f"Plaintext  (AUC = {abs(plain_auc):.3f})")

        # Kitsune paper reference
        ax.axhline(0.978, color="gray", linewidth=1, linestyle=":",
                   alpha=0.7, label="Kitsune paper AUC ≈ 0.978")

        # Current operating point
        decisions = fpga_all >= fpga_thresh
        n_pos = np.sum(labels == 1)
        op_tpr = np.sum(decisions & (labels == 1)) / n_pos if n_pos > 0 else 0.0
        op_fpr = np.sum(decisions & (labels == 0)) / np.sum(labels == 0)
        ax.scatter([op_fpr], [op_tpr], color=C_MAL, s=100, zorder=5,
                   label=f"Operating point (CFAR={fpga_thresh:.4f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5,
                label="Random classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {profile.upper()} ({n_batches} batches)")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = plots_dir / "fig4_roc_curve.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Niobium FHE Demo Report")
    parser.add_argument("run_dir", help="Path to run archive (e.g. runs/latest/)")
    parser.add_argument("--profile", default="mirai",
                        choices=list(LABEL_FILES.keys()),
                        help="Workload profile (default: mirai)")
    parser.add_argument("--nid-root", default=".",
                        help="Path to fhe-NetworkMonitor root (default: .)")
    parser.add_argument("--cfar", type=float, default=None,
                        help="Override CFAR threshold (default: read from metadata or compute)")
    parser.add_argument("--per-path-cfar", action="store_true",
                        help="Show independent per-path CFAR thresholds (FPGA vs plaintext)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    nid_root = Path(args.nid_root).resolve()
    global BACKEND
    _bn, _bhw, _bnote = backend_name(run_dir)
    BACKEND = _bn
    if _bnote:
        print(f"\u26a0 NOTE: {_bnote}", file=sys.stderr)

    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    # Load data
    meta = load_metadata(run_dir)
    ts = find_timing_summary(nid_root)
    if ts:
        meta["_timing_summary"] = ts
    fpga_rmse_batches, _ = load_all_batches(run_dir, "scores")
    # Prefer the exact-activation reference (true model); fall back to the
    # harness's Chebyshev plaintext_batch (matches the FHE circuit).
    plain_rmse_batches, _ = load_all_batches(run_dir, "reference")
    plain_source = "exact-activation reference (true model)"
    if not plain_rmse_batches:
        plain_rmse_batches, _ = load_all_batches(run_dir, "plaintext")
        plain_source = "Chebyshev plaintext (FHE-matched)"
    print(f"  Plaintext baseline: {plain_source}", file=sys.stderr)

    if not fpga_rmse_batches:
        print(f"ERROR: No scores_batch*.csv found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    n_batches = len(fpga_rmse_batches)
    fpga_all = np.concatenate(fpga_rmse_batches)

    # CFAR threshold — priority: CLI arg > summary file > metadata > error
    if args.cfar is not None:
        cfar_threshold = args.cfar
    else:
        cfar_threshold = load_cfar_from_summary(run_dir)
        if cfar_threshold is None:
            cfar_threshold = meta.get("cfar_threshold", None)
        if cfar_threshold is None:              # profile-tuned value (profiles.json), same as plot_comparison
            try:
                from plot_fpga_results import tuned_cfar
                cfar_threshold = tuned_cfar(run_dir, args.profile, nid_root)
            except Exception:
                cfar_threshold = None
        if cfar_threshold is None:
            print("ERROR: CFAR threshold not found. Pass --cfar <value> explicitly.",
                  file=sys.stderr)
            sys.exit(1)

    # Labels
    labels = load_labels(nid_root, args.profile, n_batches)

    # Print terminal report
    print_report(run_dir, args.profile, fpga_rmse_batches,
                 plain_rmse_batches, labels, cfar_threshold, meta,
                 per_path_cfar=args.per_path_cfar)

    # Generate plots — pass per-path thresholds when flag is set
    fpga_thresh = None
    plain_thresh = None
    if args.per_path_cfar and labels is not None:
        fpga_all = np.concatenate(fpga_rmse_batches)
        plain_all = np.concatenate(plain_rmse_batches) if plain_rmse_batches else None
        fpga_normal = fpga_all[labels == 0] if len(labels) == len(fpga_all) else np.array([])
        if len(fpga_normal) > 0:
            fpga_thresh = float(np.percentile(fpga_normal, 99.9))
        if plain_all is not None and len(plain_all) == len(labels):
            plain_normal = plain_all[labels == 0]
            if len(plain_normal) > 0:
                plain_thresh = float(np.percentile(plain_normal, 99.9))

    if HAS_MATPLOTLIB:
        print("  PLOTS")
        print("  " + "-" * 40)
        make_plots(run_dir, args.profile, fpga_rmse_batches,
                   plain_rmse_batches, labels, cfar_threshold,
                   fpga_thresh=fpga_thresh, plain_thresh=plain_thresh)
    else:
        print("  Install matplotlib to generate plots: pip install matplotlib")

    print()
    print("  To copy plots to your local machine, run from your laptop:")
    hostname = meta.get("environment", {}).get("hostname") or socket.gethostname()
    print(f"    scp -r $(whoami)@{hostname}:{run_dir.resolve()}/plots/ ./niobium-report/")
    print()


if __name__ == "__main__":
    main()
