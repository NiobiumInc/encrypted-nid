#!/usr/bin/env bash
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
#
# build_task.sh — self-contained build (FBS-submission style): the niobium-client
# submodule builds its OWN bundled OpenFHE + libnbfhetch + the FHETCH transport,
# and the NID stages build against that. Run from the repo root. No compiler
# checkout required. See NIOBIUM_INTEGRATION.md.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== [1/3] sync niobium-client (its OpenFHE + niobium-fhetch + cpp-httplib) ==="
git submodule update --init niobium-client
git -C niobium-client submodule update --init --recursive

echo "=== [2/3] build the client's bundled OpenFHE + libnbfhetch + transport (make release) ==="
make -C niobium-client release        # installs OpenFHE to niobium-client/vendor/lib/openfhe

OPENFHE_PREFIX="$ROOT/niobium-client/vendor/lib/openfhe"
[[ -d "$OPENFHE_PREFIX" ]] || { echo "error: client OpenFHE not at $OPENFHE_PREFIX after 'make release'" >&2; exit 1; }

echo "=== [3/3] build the NID stages + SDK server against the client's OpenFHE ==="
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DNIOBIUM_SDK_BUILD=ON \
  -DCMAKE_PREFIX_PATH="$OPENFHE_PREFIX"
cmake --build build -j \
  --target keygen encrypt_mirai decrypt_probe server_standalone_sdk

echo "=== done: binaries in $ROOT/build ==="
ls -la build/keygen build/encrypt_mirai build/decrypt_probe build/server_standalone_sdk