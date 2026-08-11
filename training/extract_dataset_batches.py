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
"""Extract multiple batches from any dataset PCAP file

This is a NEW script for multi-batch support. The existing generate_mirai_batch.py
is kept unchanged for backward compatibility with CI/nightly tests.

NOTE: Generated batch files are NOT committed to the repository (.gitignored).
They are auto-generated at runtime and cached locally for subsequent runs.

Usage:
    python3 extract_dataset_batches.py unit 2      # Extract 2 batches from Unit
    python3 extract_dataset_batches.py mirai 22    # Extract 22 batches from Mirai
    python3 extract_dataset_batches.py mirai 21  # Extract 21 batches

Output: assets/datasets/{Dataset}_batch_{N}.bin (13 MB each, cached locally)
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path to access Features and KitNET modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import Features.FeatureExtractor as fe
import KitNET.KitNET as ad
import KitNET.NormStat as fn

def extract_batches(dataset_name, num_batches, batch_size=32768, skip=0):
    """Extract multiple batches from dataset PCAP

    Args:
        dataset_name: Name of the dataset (mirai, unit, etc.)
        num_batches: Number of batches to extract
        batch_size: Packets per batch (default 32768)
        skip: Number of initial packets to skip (warm-up period).
              When >0, the first `skip` packets are read through the
              feature extractor (to warm up AfterImage statistics) but
              not included in any batch. This matches the CPU (networkmonitor.py) path
              (networkmonitor.py totaltrain).
    """

    project_root = Path(__file__).resolve().parent.parent

    # Dataset name to directory name mapping
    dataset_dir_map = {
        'mirai': 'Mirai',
        'unit': 'Unit',
    }

    # Get directory name (with proper capitalization)
    dir_name = dataset_dir_map.get(dataset_name.lower(), dataset_name.capitalize())

    # Dataset-specific paths
    dataset_dir = project_root / f"Datasets/{dir_name}"
    pcap_path = dataset_dir / f"{dir_name}_pcap.pcap"
    model_path = dataset_dir / f"{dir_name}_model.bin"
    norm_path = dataset_dir / f"{dir_name}_norm.bin"

    print(f"=" * 60)
    print(f"{dataset_name.upper()} Dataset - Multi-Batch Extraction")
    print(f"=" * 60)
    print(f"PCAP file: {pcap_path}")
    print(f"Model file: {model_path}")
    print(f"Norm file: {norm_path}")
    print(f"Batches: {num_batches}")
    print(f"Batch size: {batch_size:,} packets")
    if skip > 0:
        print(f"Warm-up skip: {skip:,} packets (matching the CPU path)")
    print(f"Total packets: {num_batches * batch_size:,}")
    print()

    # Validate files exist
    if not pcap_path.exists():
        print(f"ERROR: PCAP not found: {pcap_path}")
        print(f"Expected: {pcap_path}")
        sys.exit(1)

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        print(f"Expected: {model_path}")
        sys.exit(1)

    if not norm_path.exists():
        print(f"ERROR: Normalization file not found: {norm_path}")
        print(f"Expected: {norm_path}")
        sys.exit(1)

    # Load model
    print("Loading model...")
    # KitNET expects Path object with .exists() method (not string)
    detector = ad.KitNET(model_file=model_path)
    num_feat = detector.get_num_features()
    print(f"✓ Model loaded: {num_feat} features")

    if num_feat != 50:
        print(f"WARNING: Expected 50 features, got {num_feat}")

    # Load normalization parameters (logistic sigmoid: matches training pipeline)
    print("Loading normalization parameters...")
    FN = fn.normStat(num_feat, norm_path)
    print(f"✓ Normalization loaded from: {norm_path}")

    # Create feature extractor
    # Pass Path object directly (processpcap.py expects Path with .exists() method)
    extractor = fe.featExt(pcap_path)

    # Warm-up: skip initial packets to match the CPU path (networkmonitor.py totaltrain).
    # These packets are read through AfterImage so connection statistics build up,
    # but they are NOT included in any output batch.
    if skip > 0:
        print(f"\nWarm-up: reading {skip:,} packets through AfterImage (not saved)...")
        for ii in range(skip):
            feature_vector, conn_id, timestamp = extractor.get_next_vector()
            if len(feature_vector) == 0:
                print(f"ERROR: PCAP ended during warm-up at packet {ii:,} (need {skip:,})")
                sys.exit(1)
            if ii % 10000 == 0 and ii > 0:
                print(f"  Warm-up progress: {ii:,}/{skip:,}")
        print(f"✓ Warm-up complete: {skip:,} packets consumed")

    print(f"\nExtracting {num_batches} batches from PCAP...")

    # Extract each batch
    batches_extracted = 0
    for batch_idx in range(num_batches):
        print(f"\n--- Batch {batch_idx}/{num_batches} ---")

        # Create batch buffer
        batch_buf = np.zeros((num_feat, batch_size))

        # Extract features for this batch
        packets_in_batch = 0
        for pkt_idx in range(batch_size):
            feature_vector, conn_id, timestamp = extractor.get_next_vector()

            if len(feature_vector) == 0:
                # Reached end of PCAP
                total_extracted = batch_idx * batch_size + pkt_idx
                print(f"\nWARNING: Only {total_extracted:,} packets available in PCAP!")
                print(f"Expected at least {num_batches * batch_size:,} packets")

                if pkt_idx > 0:
                    # Partial batch - save what we have
                    print(f"Saving partial batch {batch_idx} with {pkt_idx:,} packets")
                    batch_buf = batch_buf[:, :pkt_idx]  # Trim to actual size
                    packets_in_batch = pkt_idx
                else:
                    print(f"Batch {batch_idx} is empty, stopping")
                    return batches_extracted
                break

            batch_buf[:, pkt_idx] = FN.get_logisticnorm(np.array(feature_vector))
            packets_in_batch = pkt_idx + 1

            if pkt_idx % 5000 == 0 and pkt_idx > 0:
                print(f"  Extracted {pkt_idx:,}/{batch_size:,} packets...")

        # Determine output path
        output_dir = project_root / "assets/datasets"
        output_dir.mkdir(exist_ok=True)

        # Always use batch suffix for multi-batch datasets (use dir_name for consistency)
        output_path = output_dir / f"{dir_name}_batch_{batch_idx}.bin"

        # Save batch in packet-major layout (matches encrypt_mirai.cpp: all_data[pkt * num_features + feat])
        print(f"Saving to: {output_path}")
        np.ascontiguousarray(batch_buf.T).astype(np.float64).tofile(str(output_path))

        file_size = output_path.stat().st_size
        expected_size = num_feat * packets_in_batch * 8

        print(f"✓ Batch {batch_idx} saved")
        print(f"  Size: {file_size:,} bytes")
        print(f"  Format: {num_feat} features × {packets_in_batch:,} packets × 8 bytes")

        if file_size != expected_size:
            print(f"WARNING: Size mismatch! Expected {expected_size:,}, got {file_size:,}")

        batches_extracted += 1

        # If this was a partial batch, stop
        if packets_in_batch < batch_size:
            break

    print(f"\n{'='*60}")
    print(f"✓ SUCCESS: {batches_extracted} batches extracted")
    print(f"{'='*60}")
    return batches_extracted


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 extract_dataset_batches.py <dataset> <num_batches> [--skip=N]")
        print()
        print("Examples:")
        print("  python3 extract_dataset_batches.py unit 2       # Unit: 2 batches (65,536 packets)")
        print("  python3 extract_dataset_batches.py mirai 21     # Mirai: 21 batches, auto-skip from profiles.json")
        print("  python3 extract_dataset_batches.py mirai 21 --skip=60810  # Explicit warm-up skip")
        print()
        print("The --skip parameter controls how many initial packets to read through")
        print("AfterImage (warm-up) before saving batches. If omitted, the value is")
        print("read from profiles.json 'warmup_skip' field (matching the CPU path).")
        print()
        print("Valid datasets: mirai, unit")
        sys.exit(1)

    dataset = sys.argv[1].lower()
    num_batches = int(sys.argv[2])

    valid_datasets = ['mirai', 'unit']
    if dataset not in valid_datasets:
        print(f"ERROR: Invalid dataset '{dataset}'")
        print(f"Valid datasets: {', '.join(valid_datasets)}")
        sys.exit(1)

    if num_batches < 1 or num_batches > 100:
        print(f"ERROR: Invalid num_batches {num_batches}. Must be 1-100")
        sys.exit(1)

    # Parse --skip from CLI or profiles.json
    skip = None
    for arg in sys.argv[3:]:
        if arg.startswith("--skip="):
            skip = int(arg.split("=", 1)[1])

    if skip is None:
        # Read from profiles.json
        import json
        profiles_path = Path(__file__).resolve().parent.parent / "profiles.json"
        if profiles_path.exists():
            with open(profiles_path) as f:
                profiles = json.load(f)
            profile = profiles.get("profiles", {}).get(dataset, {})
            skip = profile.get("warmup_skip", 0)
            if skip > 0:
                print(f"Using warmup_skip={skip:,} from profiles.json")
        else:
            skip = 0

    extract_batches(dataset, num_batches, skip=skip)
