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

"""Extract first 32,768 packets and compute plaintext anomaly score"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import Features.FeatureExtractor as fe
import KitNET.KitNET as ad

project_root = Path(__file__).resolve().parent
pcap_path = project_root / "Datasets/Mirai/Mirai_pcap.pcap"
model_path = project_root / "Mirai_model.bin"

batch_size = 2**15  # 32,768

print(f"Loading model: {model_path}")
detector = ad.KitNET(model_file=model_path)
num_feat = detector.get_num_features()

print(f"Extracting first {batch_size} packets from PCAP...")
extractor = fe.featExt(pcap_path)

# Create batch buffer (no openfhe needed)
buf = np.zeros((num_feat, batch_size))

# Extract first 32,768 packets
for pkt_idx in range(batch_size):
    feature_vector, conn_id, timestamp = extractor.get_next_vector()
    if len(feature_vector) == 0:
        print(f"ERROR: Only {pkt_idx} packets available!")
        sys.exit(1)
    buf[:, pkt_idx] = feature_vector
    if pkt_idx % 5000 == 0:
        print(f"  Extracted {pkt_idx}/{batch_size}")

print(f"✓ Extracted {batch_size} packets")

# Compute plaintext anomaly scores
scores = np.zeros(batch_size)

print("Computing anomaly scores...")
for pkt_idx in range(batch_size):
    x = buf[:, pkt_idx]
    scores[pkt_idx] = detector.execute(x)
    if pkt_idx % 5000 == 0:
        print(f"  Computed {pkt_idx}/{batch_size}")

# Compare
expected = 0.0000377519182625
final_score = scores[-1]

print(f"\n{'='*60}")
print(f"PLAINTEXT BATCH VERIFICATION")
print(f"{'='*60}")
print(f"Expected FHE result: {expected:.16e}")
print(f"Final packet score:  {final_score:.16e}")
print(f"Mean score:          {np.mean(scores):.16e}")
print(f"Max score:           {np.max(scores):.16e}")
print(f"Min score:           {np.min(scores):.16e}")

error = abs(final_score - expected)
rel_error = (error / abs(expected)) * 100.0

print(f"\nError: {error:.16e}")
print(f"Relative: {rel_error:.4f}%")

if rel_error < 1.0:
    print("\n✅ PLAINTEXT MATCHES FHE RESULT!")
else:
    print("\n⚠️  Difference detected")