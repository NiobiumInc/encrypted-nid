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
Plot FPGA anomaly detection results — same visualizations as the CPU path.

Reads per-batch MSE score CSVs saved by decrypt_probe (--save-scores) and
produces the FPGA anomaly-score figures.

Usage:
    # After an FPGA run with score saving:
    python scripts/plot_fpga_results.py runs/<run_id>/

    # With explicit label file and profile:
    python scripts/plot_fpga_results.py runs/<run_id>/ --profile mirai --save-figs

    # Show specific batches only:
    python scripts/plot_fpga_results.py runs/<run_id>/ --batches 0-5
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


BATCH_SIZE = 32768

# Label file locations per profile (relative to NID root)
LABEL_FILES = {
    "mirai": "Datasets/Mirai/Mirai_labels.csv",
    "unit": "Datasets/Unit/Unit_labels.csv",
}

# Training grace period: first malicious packet index / 2 (matches networkmonitor.py)
# For profiles where all data is normal, grace period = total / 2
TRAINING_GRACE = {
    "mirai": 52863,     # firstmal=105726, grace=52863
    "unit": 30000,      # firstmal=60001, grace=30000
}


def load_scores(run_dir: Path, batch_range=None):
    """Load all scores_batch*.csv from a run directory."""
    all_mse = []
    all_rmse = []
    batch_idx = 0

    while True:
        csv_path = run_dir / f"scores_batch{batch_idx}.csv"
        if not csv_path.exists():
            break

        if batch_range and batch_idx not in batch_range:
            batch_idx += 1
            continue

        mse_batch = []
        rmse_batch = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                mse_batch.append(float(row["MSE"]))
                rmse_batch.append(float(row["RMSE"]))

        all_mse.extend(mse_batch)
        all_rmse.extend(rmse_batch)
        print(f"  Loaded batch {batch_idx}: {len(mse_batch)} packets", file=sys.stderr)
        batch_idx += 1

    if not all_rmse:
        print(f"ERROR: No scores_batch*.csv found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    return np.array(all_mse), np.array(all_rmse), batch_idx


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


def load_labels(nid_root: Path, profile: str, num_packets: int, num_batches: int):
    """Load ground-truth labels for the given profile, aligned to batch boundaries."""
    if profile not in LABEL_FILES:
        print(f"  No label file for profile '{profile}', skipping labels", file=sys.stderr)
        return None

    label_path = nid_root / LABEL_FILES[profile]
    if not label_path.exists():
        print(f"  Label file not found: {label_path}", file=sys.stderr)
        return None

    # Read all labels (1-indexed CSV: index,label)
    all_labels = []
    with open(label_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            all_labels.append(int(row[1]))

    # Training grace period — FPGA batches start after training
    grace = TRAINING_GRACE.get(profile, 0)
    start_idx = grace
    end_idx = start_idx + num_packets

    if end_idx > len(all_labels):
        print(f"  WARNING: Need labels up to {end_idx} but file has {len(all_labels)}", file=sys.stderr)
        end_idx = len(all_labels)

    labels = all_labels[start_idx:end_idx]

    # Pad if needed
    if len(labels) < num_packets:
        labels.extend([0] * (num_packets - len(labels)))

    print(f"  Loaded {len(labels)} labels (packets {start_idx}-{end_idx}), "
          f"{sum(labels)} malicious", file=sys.stderr)
    return np.array(labels[:num_packets])


def compute_cfar(scores, percentile=99.9, labels=None):
    """CFAR threshold = the given percentile of NORMAL-traffic scores. The
    constant false-alarm rate is defined on benign traffic, so computing it over
    ALL scores (incl. malicious) wrongly inflates the threshold above the attack
    band. Falls back to all scores only when labels are unavailable."""
    ref = scores
    if labels is not None and len(labels) == len(scores) and np.any(labels == 0):
        ref = scores[labels == 0]
    return float(np.percentile(ref, percentile))


def tuned_cfar(run_dir, profile, nid_root):
    """The profile's tuned CFAR (cfar_threshold) from the run manifest
    (run.json / metadata.json) or profiles.json; None if not defined."""
    import json
    for name in ("run.json", "metadata.json"):
        p = run_dir / name
        if p.exists():
            v = json.load(open(p)).get("cfar_threshold")
            if v:
                return float(v)
    pj = nid_root / "profiles.json"
    if pj.exists():
        v = json.load(open(pj)).get("profiles", {}).get(profile, {}).get("cfar_threshold")
        if v:
            return float(v)
    return None


def plot_results(rmse, labels, cfar, profile, run_dir,
                 save_figs=False, hide_figs=False):
    """Generate the anomaly-score visualization figures."""
    num_pkts = len(rmse)

    if save_figs:
        out_dir = run_dir / "graphs"
        out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Packet anomaly scores (colored by batch)
    batch_ids = np.array([i // BATCH_SIZE for i in range(num_pkts)])

    fig1 = plt.figure(1, figsize=(10, 5))
    plt.scatter(range(num_pkts), rmse, s=0.1, c=batch_ids, cmap='viridis')
    plt.plot([0, num_pkts], [cfar, cfar], 'k--', label=f'CFAR={cfar:.4f}')
    plt.yscale("log")
    plt.title(f"{BACKEND} Packet Anomaly Scores — {profile.upper()}")
    plt.ylabel("RMSE (log scaled)")
    plt.xlabel("Packet Number")
    plt.legend()
    figbar = plt.colorbar()
    figbar.ax.set_ylabel('Batch ID\n ', rotation=270)
    if save_figs:
        fig1.savefig(out_dir / f"{profile}_fig1_packet_anomaly_scores.png", dpi=300)

    # Figure 2: Anomaly scores with ground truth labels
    if labels is not None:
        fig2 = plt.figure(2, figsize=(10, 5))
        plt.scatter(range(num_pkts), rmse, s=0.1, c=labels, cmap='RdYlGn_r')
        plt.plot([0, num_pkts], [cfar, cfar], 'k--', label=f'CFAR={cfar:.4f}')
        plt.yscale("log")
        plt.title(f"{BACKEND} Anomaly Scores with Labels — {profile.upper()}")
        plt.ylabel("RMSE (log scaled)")
        plt.xlabel("Packet Number")
        plt.legend()
        if save_figs:
            fig2.savefig(out_dir / f"{profile}_fig2_anomaly_scores_labeled.png", dpi=300)

    # Figure 3: Per-batch average anomaly score
    num_batches = int(np.ceil(num_pkts / BATCH_SIZE))
    batch_avgs = []
    for b in range(num_batches):
        start = b * BATCH_SIZE
        end = min((b + 1) * BATCH_SIZE, num_pkts)
        batch_avgs.append(np.mean(rmse[start:end]))

    fig3 = plt.figure(3, figsize=(8, 4))
    plt.bar(range(num_batches), batch_avgs, color='steelblue')
    plt.axhline(y=cfar, color='k', linestyle='--', label=f'CFAR={cfar:.4f}')
    plt.yscale("log")
    plt.title(f"Average Anomaly Score per Batch — {profile.upper()}")
    plt.ylabel("RMSE (log scaled)")
    plt.xlabel("Batch Index")
    plt.legend()
    if save_figs:
        fig3.savefig(out_dir / f"{profile}_fig3_avg_score_per_batch.png", dpi=300)

    # Figure 4: Score distribution histogram
    fig4 = plt.figure(4, figsize=(10, 5))
    plt.hist(rmse, bins=200, color='steelblue', alpha=0.7, density=True)
    plt.axvline(x=cfar, color='red', linestyle='--', linewidth=2, label=f'CFAR={cfar:.4f}')
    plt.yscale("log")
    plt.title(f"{BACKEND} Score Distribution — {profile.upper()}")
    plt.xlabel("RMSE")
    plt.ylabel("Density (log scaled)")
    plt.legend()
    if save_figs:
        fig4.savefig(out_dir / f"{profile}_fig4_score_distribution.png", dpi=300)

    # Figure 5: Detection summary (if labels available)
    if labels is not None:
        detected = rmse > cfar
        tp = np.sum(detected & (labels == 1))
        fp = np.sum(detected & (labels == 0))
        fn = np.sum(~detected & (labels == 1))
        tn = np.sum(~detected & (labels == 0))
        total_mal = np.sum(labels == 1)
        total_norm = np.sum(labels == 0)

        fig5 = plt.figure(5, figsize=(8, 5))
        categories = ['True Pos', 'False Pos', 'False Neg', 'True Neg']
        values = [tp, fp, fn, tn]
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
        bars = plt.bar(categories, values, color=colors)
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                     str(val), ha='center', va='bottom', fontweight='bold')
        plt.title(f"{BACKEND} Detection Results — {profile.upper()}\n"
                  f"Detection rate: {tp}/{total_mal} ({100*tp/max(1,total_mal):.1f}%), "
                  f"False alarm: {fp}/{total_norm} ({100*fp/max(1,total_norm):.2f}%)")
        plt.ylabel("Packet Count")
        if save_figs:
            fig5.savefig(out_dir / f"{profile}_fig5_detection_summary.png", dpi=300)

    # Figure 6: Cumulative anomaly detection over time (if labels)
    if labels is not None:
        fig6 = plt.figure(6, figsize=(10, 5))
        cum_detected = np.cumsum(detected.astype(int))
        cum_actual = np.cumsum((labels == 1).astype(int))
        plt.plot(range(num_pkts), cum_actual, 'r-', linewidth=1.5, label='Actual malicious')
        plt.plot(range(num_pkts), cum_detected, 'b--', linewidth=1.5, label=f'{BACKEND} detected')
        plt.title(f"Cumulative Anomaly Detection — {profile.upper()}")
        plt.xlabel("Packet Number")
        plt.ylabel("Cumulative Count")
        plt.legend()
        if save_figs:
            fig6.savefig(out_dir / f"{profile}_fig6_cumulative_detection.png", dpi=300)

    if save_figs:
        print(f"\nFigures saved to {out_dir}/", file=sys.stderr)

    if not hide_figs:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot FPGA anomaly detection results (same visualizations as CPU path)")
    parser.add_argument("run_dir", type=Path,
                        help="Path to run archive directory containing scores_batch*.csv")
    parser.add_argument("--profile", type=str, default=None,
                        help="Profile name for labels (mirai, unit, etc.). "
                             "Auto-detected from metadata.json if present.")
    parser.add_argument("--batches", type=str, default=None,
                        help="Batch range to plot, e.g. '0-5' or '0,1,3'")
    parser.add_argument("--cfar", type=float, default=None,
                        help="Explicit CFAR threshold (overrides the profile-tuned value)")
    parser.add_argument("--cfar-percentile", type=float, default=99.9,
                        help="Percentile of NORMAL traffic for CFAR when no tuned value exists (default: 99.9)")
    parser.add_argument("--save-figs", action="store_true",
                        help="Save figures as PNGs to run_dir/graphs/")
    parser.add_argument("--hide-figs", action="store_true",
                        help="Don't display figures (use with --save-figs)")
    parser.add_argument("--save-scores-csv", type=str, default=None,
                        help="Save combined scores+labels to CSV")
    args = parser.parse_args()

    nid_root = Path(__file__).resolve().parent.parent
    run_dir = args.run_dir.resolve()

    # Auto-detect profile from the run manifest (run.json / metadata.json)
    profile = args.profile
    if profile is None:
        import json
        for name in ("run.json", "metadata.json"):
            p = run_dir / name
            if p.exists():
                profile = json.load(open(p)).get("profile")
                if profile:
                    print(f"Auto-detected profile: {profile}", file=sys.stderr)
                    break
        if profile is None:
            profile = "mirai"
            print(f"No run manifest found, defaulting to profile: {profile}", file=sys.stderr)

    # Parse batch range
    batch_range = None
    if args.batches:
        batch_range = set()
        for part in args.batches.split(","):
            if "-" in part:
                a, b = part.split("-")
                batch_range.update(range(int(a), int(b) + 1))
            else:
                batch_range.add(int(part))

    global BACKEND
    _bn, _bhw, _bnote = backend_name(run_dir)
    BACKEND = _bn
    if _bnote:
        print(f"\u26a0 NOTE: {_bnote}", file=sys.stderr)
    print(f"Loading {BACKEND} scores from {run_dir}...", file=sys.stderr)
    mse, rmse, num_batches = load_scores(run_dir, batch_range)
    num_pkts = len(rmse)
    print(f"Total: {num_pkts} packets from {num_batches} batches", file=sys.stderr)

    # Load labels
    print(f"Loading labels for profile '{profile}'...", file=sys.stderr)
    labels = load_labels(nid_root, profile, num_pkts, num_batches)

    # CFAR threshold: explicit --cfar > profile-tuned (run.json/profiles.json,
    # pre-calibrated on training data, as demo_report.py uses) > 99.9th pct of
    # NORMAL traffic. NEVER a percentile of all scores (that inflates the
    # threshold above the attack band and zeroes out detection).
    if args.cfar is not None:
        cfar, src = args.cfar, "override"
    else:
        cfar = tuned_cfar(run_dir, profile, nid_root)
        if cfar is not None:
            src = "profile-tuned (cfar_threshold)"
        else:
            cfar = compute_cfar(rmse, args.cfar_percentile, labels)
            src = f"{args.cfar_percentile}th pct of normal"
    detected = np.sum(rmse > cfar)
    print(f"\nCFAR threshold: {cfar:.6f}  [{src}]", file=sys.stderr)
    print(f"Detected anomalies: {detected}/{num_pkts} ({100*detected/num_pkts:.2f}%)", file=sys.stderr)

    if labels is not None:
        actual = np.sum(labels == 1)
        tp = np.sum((rmse > cfar) & (labels == 1))
        print(f"Actual malicious: {actual}/{num_pkts} ({100*actual/num_pkts:.2f}%)", file=sys.stderr)
        print(f"True positive rate: {tp}/{actual} ({100*tp/max(1,actual):.1f}%)", file=sys.stderr)

    # Save combined CSV if requested
    if args.save_scores_csv:
        out_path = Path(args.save_scores_csv)
        idx = np.arange(num_pkts)
        anomalies = (rmse > cfar).astype(int)
        lbl = labels if labels is not None else np.zeros(num_pkts, dtype=int)
        with open(out_path, "w") as f:
            f.write(f"CFAR,{cfar}\n")
            f.write("Packet,Score (RMSE),Anomaly,Label\n")
            for i in range(num_pkts):
                f.write(f"{i},{rmse[i]:.6f},{anomalies[i]},{lbl[i]}\n")
        print(f"Saved scores CSV to {out_path}", file=sys.stderr)

    # Plot
    plot_results(rmse, labels, cfar, profile, run_dir,
                 save_figs=args.save_figs, hide_figs=args.hide_figs)


if __name__ == "__main__":
    main()
