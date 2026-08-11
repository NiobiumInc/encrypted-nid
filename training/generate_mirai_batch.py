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

"""Generate Mirai_first_batch_32K.bin from PCAP dataset

This script extracts the first 32,768 packets from the Mirai PCAP file,
computes 50-dimensional feature vectors for each packet, and saves them
as a binary file for encryption by encrypt_mirai.cpp.

Output format: 50 features × 32,768 packets (float64, column-major order)
Total size: 50 × 32,768 × 8 bytes = 13,107,200 bytes
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import Features.FeatureExtractor as fe
import KitNET.KitNET as ad

project_root = Path(__file__).resolve().parent
pcap_path = project_root / "Datasets/Mirai/Mirai_pcap.pcap"
model_path = project_root / "Mirai_model.bin"
output_path = project_root / "Mirai_first_batch_32K.bin"

batch_size = 2**15  # 32,768

print("=" * 60)
print("Mirai Batch Feature Extraction")
print("=" * 60)

# Validate input files exist
if not pcap_path.exists():
    print(f"ERROR: PCAP file not found: {pcap_path}")
    print("Expected location: Datasets/Mirai/Mirai_pcap.pcap")
    sys.exit(1)

if not model_path.exists():
    print(f"ERROR: Model file not found: {model_path}")
    print("Please train the model first (or use Mirai_model.bin)")
    sys.exit(1)

print(f"✓ PCAP file: {pcap_path}")
print(f"✓ Model file: {model_path}")

print(f"\nLoading model...")
detector = ad.KitNET(model_file=str(model_path))
num_feat = detector.get_num_features()

print(f"✓ Model loaded: {num_feat} features")

if num_feat != 50:
    print(f"WARNING: Expected 50 features, got {num_feat}")

print(f"\nExtracting first {batch_size:,} packets from PCAP...")
print("This may take several minutes...")

extractor = fe.featExt(str(pcap_path))

# Create batch buffer
buf = np.zeros((num_feat, batch_size))

# Extract first 32,768 packets
for pkt_idx in range(batch_size):
    feature_vector, conn_id, timestamp = extractor.get_next_vector()
    if len(feature_vector) == 0:
        print(f"\nERROR: Only {pkt_idx} packets available in PCAP!")
        print(f"Need at least {batch_size} packets")
        sys.exit(1)

    buf[:, pkt_idx] = feature_vector

    if pkt_idx % 5000 == 0:
        print(f"  Extracted {pkt_idx:,}/{batch_size:,} packets...")

print(f"✓ Extracted {batch_size:,} packets")

# Save as binary file (column-major: feature 0, then feature 1, ...)
print(f"\nSaving to: {output_path}")
buf.astype(np.float64).tofile(str(output_path))

file_size = output_path.stat().st_size
expected_size = num_feat * batch_size * 8

print(f"✓ Binary file created")
print(f"  Size: {file_size:,} bytes (expected: {expected_size:,})")
print(f"  Format: {num_feat} features × {batch_size:,} packets × 8 bytes")

if file_size != expected_size:
    print(f"\nWARNING: File size mismatch!")
    print(f"  Expected: {expected_size:,} bytes")
    print(f"  Actual:   {file_size:,} bytes")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ SUCCESS: Feature batch ready for encryption")
print("=" * 60)
print(f"\nNext step: Run encrypt_mirai to encrypt these features")
print(f"  cd Server/build")
print(f"  ./encrypt_mirai")
