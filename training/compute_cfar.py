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
"""Compute CFAR thresholds for all datasets and store in profiles.json

CFAR (Constant False Alarm Rate) = 99.9th percentile of RMSE scores on
normal test traffic.  This is a one-time, plaintext computation that
depends only on the trained model and the PCAP — not on any FHE library.

The computed thresholds are written to profiles.json under each profile's
"cfar_threshold" field so both the CPU and FPGA paths can use them.

Usage:
    python3 compute_cfar.py                 # Compute all datasets
    python3 compute_cfar.py mirai unit      # Compute specific datasets
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add parent directory to path for KitNET / Features modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import Features.FeatureExtractor as fe
import Features.processlabel as pl
import KitNET.KitNET as ad
import KitNET.NormStat as fn


def compute_cfar(dataset_name, project_root, percentile=99.9):
    """Compute CFAR threshold for a dataset.

    Replicates the CPU path (networkmonitor.py):
      1. Warm up AfterImage for totaltrain packets
      2. Score normal test traffic (totaltrain .. firstmalpkt)
      3. CFAR = percentile of those scores
    """
    # Load profiles.json for dataset config
    profiles_path = project_root / "profiles.json"
    with open(profiles_path) as f:
        profiles = json.load(f)["profiles"]

    profile = profiles[dataset_name]
    data_source = profile["data_source"]

    # Paths
    dataset_dir = project_root / "Datasets" / data_source
    pcap_path = dataset_dir / f"{data_source}_pcap.pcap"
    label_path = dataset_dir / f"{data_source}_labels.csv"
    model_path = dataset_dir / f"{data_source}_model.bin"
    norm_path = dataset_dir / f"{data_source}_norm.bin"

    print(f"\n{'='*60}")
    print(f"Computing CFAR for: {dataset_name} ({data_source})")
    print(f"{'='*60}")

    # Validate files
    for p, name in [(pcap_path, "PCAP"), (label_path, "Labels"),
                     (model_path, "Model"), (norm_path, "Norm")]:
        if not p.exists():
            print(f"ERROR: {name} not found: {p}")
            return None
        print(f"  {name}: {p}")

    # Read labels to find firstmalpkt
    LF = pl.labelParse(label_path)
    totalpkts = LF.read_labels()
    firstmalpkt = LF.get_firstmal()
    totaltrain = int(np.floor(0.5 * firstmalpkt))
    num_test_normal = firstmalpkt - totaltrain

    print(f"  Total packets: {totalpkts:,}")
    print(f"  First malicious: {firstmalpkt:,}")
    print(f"  Training (warm-up): {totaltrain:,}")
    print(f"  Normal test packets: {num_test_normal:,}")

    # Load model and normalization
    detector = ad.KitNET(model_file=model_path)
    num_feat = detector.get_num_features()
    FN = fn.normStat(num_feat, norm_path)
    print(f"  Features: {num_feat}")

    # Create feature extractor
    extractor = fe.featExt(pcap_path)

    # Phase 1: warm up AfterImage (training grace period)
    print(f"\n  Warm-up: processing {totaltrain:,} training packets...")
    for ii in range(totaltrain):
        feature_vector, conn_id, timestamp = extractor.get_next_vector()
        if len(feature_vector) == 0:
            print(f"  ERROR: PCAP ended at packet {ii:,} during warm-up")
            return None
        if ii % 50000 == 0 and ii > 0:
            print(f"    {ii:,}/{totaltrain:,}")

    # Phase 2: score normal test traffic (totaltrain .. firstmalpkt)
    print(f"  Scoring: {num_test_normal:,} normal test packets...")
    scores = np.zeros(num_test_normal)
    for ii in range(num_test_normal):
        feature_vector, conn_id, timestamp = extractor.get_next_vector()
        if len(feature_vector) == 0:
            print(f"  ERROR: PCAP ended at packet {ii:,} during scoring")
            scores = scores[:ii]
            break

        # Normalize and run through model (same as networkmonitor.py)
        normalized = FN.get_logisticnorm(np.array(feature_vector))
        mse = detector.execute(normalized)
        scores[ii] = np.sqrt(mse)  # RMSE

        if ii % 50000 == 0 and ii > 0:
            print(f"    {ii:,}/{num_test_normal:,}")

    # Compute CFAR
    cfar = float(np.percentile(scores, percentile))
    print(f"\n  CFAR ({percentile}th percentile): {cfar:.6f}")
    print(f"  Score range: [{scores.min():.6f}, {scores.max():.6f}]")
    print(f"  Mean RMSE: {scores.mean():.6f}")
    print(f"  Median RMSE: {np.median(scores):.6f}")

    return cfar


def main():
    project_root = Path(__file__).resolve().parent.parent
    profiles_path = project_root / "profiles.json"

    with open(profiles_path) as f:
        data = json.load(f)

    # Datasets that support CFAR (have labels + full feature set)
    cfar_datasets = ["unit", "mirai"]

    # Allow filtering from CLI
    if len(sys.argv) > 1:
        requested = [d.lower() for d in sys.argv[1:]]
        for d in requested:
            if d not in cfar_datasets:
                print(f"ERROR: '{d}' not in {cfar_datasets}")
                sys.exit(1)
        cfar_datasets = requested

    results = {}
    for dataset in cfar_datasets:
        cfar = compute_cfar(dataset, project_root)
        if cfar is not None:
            results[dataset] = cfar
            data["profiles"][dataset]["cfar_threshold"] = round(cfar, 6)

    # Write updated profiles.json
    with open(profiles_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for dataset, cfar in results.items():
        print(f"  {dataset:<15} CFAR = {cfar:.6f}")
    print(f"\nUpdated {profiles_path}")


if __name__ == "__main__":
    main()
