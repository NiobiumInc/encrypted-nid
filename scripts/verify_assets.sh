#!/bin/bash
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

# Verify fhe-NetworkMonitor assets for CI
set -e

PROFILE="${1:-full}"  # Default: full
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set NID_ROOT for scripts
export NID_ROOT="$REPO_ROOT"

case "$PROFILE" in
    full)
        MODEL_FILE="assets/models/Mirai_model_FULL.bin"
        NORM_FILE="Mirai_Workload_Inputs/Mirai_norm.bin"
        EXPECTED_MODEL_FEATURES=50
        EXPECTED_NORM_FEATURES=50
        ;;
    mirai|unit)
        # New independent datasets (all 50 features)
        # Read data_source from profiles.json for correct casing
        DATASET_NAME="$(python3 -c "
import json
with open('$REPO_ROOT/profiles.json') as f:
    print(json.load(f)['profiles']['$PROFILE']['data_source'])
")"
        MODEL_FILE="Datasets/${DATASET_NAME}/${DATASET_NAME}_model.bin"
        NORM_FILE="Datasets/${DATASET_NAME}/${DATASET_NAME}_norm.bin"
        EXPECTED_MODEL_FEATURES=50
        EXPECTED_NORM_FEATURES=50
        ;;
    *)
        echo "ERROR: Invalid dataset '$PROFILE'"
        echo "Available datasets: full, mirai, unit"
        exit 1
        ;;
esac

case "$PROFILE" in
    full)
        echo "=== Asset Verification for $PROFILE Profile ==="
        ;;
    *)
        echo "=== Asset Verification for $PROFILE Dataset ==="
        ;;
esac
echo

# For FULL profile, materialize assets if missing (NO GIT LFS)
if [ "$PROFILE" = "full" ]; then
    if [ ! -f "$MODEL_FILE" ] || [ ! -f "$NORM_FILE" ]; then
        echo "📦 FULL profile assets missing - materializing from embedded data..."
        python3 "$SCRIPT_DIR/materialize_assets.py"
        if [ $? -ne 0 ]; then
            echo "❌ Failed to materialize assets"
            exit 1
        fi
        echo
    fi
fi

# Check model file
if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Model file missing: $MODEL_FILE"
    if [ "$PROFILE" = "full" ]; then
        echo "   Run: python3 scripts/materialize_assets.py"
    fi
    exit 1
fi
echo "✓ Model file exists: $MODEL_FILE ($(ls -lh $MODEL_FILE | awk '{print $5}'))"

# Parse model header (7 uint16 = 14 bytes)
MODEL_FEATURES=$(od -An -t u2 -N 4 "$MODEL_FILE" | awk '{print $2}')
if [ "$MODEL_FEATURES" != "$EXPECTED_MODEL_FEATURES" ]; then
    echo "❌ Model feature count mismatch: $MODEL_FEATURES (expected $EXPECTED_MODEL_FEATURES)"
    exit 1
fi
echo "✓ Model feature count: $MODEL_FEATURES"

# Check normalization file
if [ ! -f "$NORM_FILE" ]; then
    echo "❌ Normalization file missing: $NORM_FILE"
    if [ "$PROFILE" = "full" ]; then
        echo "   Run: python3 scripts/materialize_assets.py"
    fi
    exit 1
fi
echo "✓ Normalization file exists: $NORM_FILE ($(ls -lh $NORM_FILE | awk '{print $5}'))"

# Parse norm header (1 uint32 = 4 bytes)
NORM_FEATURES=$(od -An -t u4 -N 4 "$NORM_FILE" | awk '{print $1}')
if [ "$NORM_FEATURES" != "$EXPECTED_NORM_FEATURES" ]; then
    echo "❌ Norm feature count mismatch: $NORM_FEATURES (expected $EXPECTED_NORM_FEATURES)"
    exit 1
fi
echo "✓ Norm feature count: $NORM_FEATURES"

# Check dataset file
case "$PROFILE" in
    mirai|unit)
        # Multi-batch profiles use the full dataset PCAP (batches extracted at runtime)
        DATASET_PCAP="Datasets/${DATASET_NAME}/${DATASET_NAME}_pcap.pcap"
        if [ ! -f "$DATASET_PCAP" ]; then
            echo "❌ Dataset PCAP missing: $DATASET_PCAP"
            echo "   Run: git lfs pull"
            exit 1
        fi
        echo "✓ Dataset file exists: $DATASET_PCAP ($(ls -lh "$DATASET_PCAP" | awk '{print $5}'))"
        ;;
    *)
        # full uses the pre-extracted first batch
        DATASET_FILE="assets/datasets/Mirai_first_batch_32K.bin"
        PACKET_COUNT=32768
        EXPECTED_SIZE=13107200  # 32768 * 50 * 8
        if [ ! -f "$DATASET_FILE" ]; then
            echo "❌ Dataset file missing: $DATASET_FILE"
            echo "   Generate with: python3 extract_first_batch.py"
            exit 1
        fi
        DATASET_SIZE=$(stat -f%z "$DATASET_FILE" 2>/dev/null || stat -c%s "$DATASET_FILE" 2>/dev/null)
        if [ "$DATASET_SIZE" != "$EXPECTED_SIZE" ]; then
            echo "❌ Dataset size mismatch: $DATASET_SIZE bytes (expected $EXPECTED_SIZE)"
            exit 1
        fi
        echo "✓ Dataset file exists: $DATASET_FILE (13 MB)"
        echo "✓ Dataset size validated: $PACKET_COUNT packets × 50 features"
        ;;
esac

echo
case "$PROFILE" in
    full)
        echo "=== All assets verified for $PROFILE profile ==="
        ;;
    *)
        echo "=== All assets verified for $PROFILE dataset ==="
        ;;
esac
exit 0
