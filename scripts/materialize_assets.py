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
materialize_assets.py - Extract embedded FULL profile assets without Git LFS

Reads embedded model/normalization data from Python constants and writes
them to Mirai_Workload_Inputs/ directory. This lets CI run the FULL
profile without requiring Git LFS.

Usage: ./scripts/materialize_assets.py [--verify-only] [--force]
"""
import sys
import os
import hashlib
from pathlib import Path

# Expected checksums (hardcoded from source files)
FULL_MODEL_SHA256 = "5a812eb364d28f732b739c94395283fe3e241c226edb44b1baeda3cf7f2badb3"
FULL_NORM_SHA256 = "a3a11f626d355d29435cd934d9af19b463a84f54f07af8294dd8b0c77cdceec8"

def compute_sha256(data: bytes) -> str:
    """Compute SHA256 checksum."""
    return hashlib.sha256(data).hexdigest()

def load_embedded_data(filename: str) -> bytes:
    """Load embedded asset from assets/ directory."""
    repo_root = Path(__file__).parent.parent

    # Map filenames to new assets/ structure
    if filename == "Mirai_model.bin":
        source_path = repo_root / "assets" / "models" / "Mirai_model_FULL.bin"
    elif filename == "Mirai_norm.bin":
        source_path = repo_root / "assets" / "normalization" / "Mirai_norm.bin"
    else:
        print(f"ERROR: Unknown asset file: {filename}", file=sys.stderr)
        sys.exit(1)

    if source_path.exists():
        return source_path.read_bytes()

    # If not found, this is an error
    print(f"ERROR: Source file not found: {source_path}", file=sys.stderr)
    print(f"  Expected in assets/ directory structure", file=sys.stderr)
    sys.exit(1)

def verify_file(path: Path, expected_sha256: str) -> bool:
    """Check if file exists and matches checksum."""
    if not path.exists():
        return False

    data = path.read_bytes()
    actual_sha256 = compute_sha256(data)

    if actual_sha256 != expected_sha256:
        print(f"  ⚠️  SHA256 mismatch", file=sys.stderr)
        print(f"    Expected: {expected_sha256}", file=sys.stderr)
        print(f"    Actual:   {actual_sha256}", file=sys.stderr)
        return False

    return True

def write_asset(path: Path, data: bytes, expected_sha256: str) -> bool:
    """Write data to file and verify."""
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write file
    path.write_bytes(data)

    # Verify what we wrote
    actual_sha256 = compute_sha256(data)
    if actual_sha256 != expected_sha256:
        print(f"  ❌ Verification failed after write", file=sys.stderr)
        print(f"    Expected: {expected_sha256}", file=sys.stderr)
        print(f"    Actual:   {actual_sha256}", file=sys.stderr)
        return False

    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Materialize FULL profile assets')
    parser.add_argument('--verify-only', action='store_true',
                       help='Check if files exist and match (don\'t write)')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing files even if checksums match')
    args = parser.parse_args()

    print("=== Niobium FHE NetworkMonitor Asset Materialization ===\n")

    # Determine NID_ROOT
    nid_root = os.environ.get('NID_ROOT')
    if not nid_root:
        script_dir = Path(__file__).parent
        nid_root = script_dir.parent
        # Verify we're in the right place
        if not (nid_root / "assets" / "models" / "Mirai_model_FULL.bin").exists():
            print("❌ ERROR: Cannot determine NID_ROOT directory", file=sys.stderr)
            print(f"  Current directory: {Path.cwd()}", file=sys.stderr)
            print(f"  Script directory: {nid_root}", file=sys.stderr)
            print("  Expected to find: assets/models/Mirai_model_FULL.bin", file=sys.stderr)
            print("  Please run from the repo root or set NID_ROOT", file=sys.stderr)
            sys.exit(1)

    nid_root = Path(nid_root).resolve()
    print(f"NID_ROOT: {nid_root}\n")

    workload_dir = nid_root / "Mirai_Workload_Inputs"
    model_path = nid_root / "assets" / "models" / "Mirai_model_FULL.bin"
    norm_path = workload_dir / "Mirai_norm.bin"

    # Load embedded data
    model_data = load_embedded_data("Mirai_model.bin")
    norm_data = load_embedded_data("Mirai_norm.bin")

    # Verify loaded data matches expected checksums
    if compute_sha256(model_data) != FULL_MODEL_SHA256:
        print(f"❌ ERROR: Source model checksum mismatch", file=sys.stderr)
        sys.exit(1)
    if compute_sha256(norm_data) != FULL_NORM_SHA256:
        print(f"❌ ERROR: Source norm checksum mismatch", file=sys.stderr)
        sys.exit(1)

    # Check existing files
    print(f"Checking: {model_path.name}")
    model_valid = verify_file(model_path, FULL_MODEL_SHA256)
    if model_valid:
        print(f"  ✓ Exists and valid ({len(model_data)} bytes)")
        print(f"  ✓ SHA256: {FULL_MODEL_SHA256}")
    else:
        print(f"  ⚠️  Missing or invalid")

    print(f"\nChecking: {norm_path.name}")
    norm_valid = verify_file(norm_path, FULL_NORM_SHA256)
    if norm_valid:
        print(f"  ✓ Exists and valid ({len(norm_data)} bytes)")
        print(f"  ✓ SHA256: {FULL_NORM_SHA256}")
    else:
        print(f"  ⚠️  Missing or invalid")

    all_valid = model_valid and norm_valid

    if args.verify_only:
        print()
        if all_valid:
            print("✓ All assets verified")
            return 0
        else:
            print("❌ Asset verification failed")
            return 1

    # Materialize if needed
    if all_valid and not args.force:
        print("\n✓ All assets already materialized and valid")
        print("  Use --force to overwrite")
        return 0

    print("\n=== Materializing Assets ===\n")

    # Write model
    if not model_valid or args.force:
        print(f"Writing: {model_path}")
        if not write_asset(model_path, model_data, FULL_MODEL_SHA256):
            return 1
        print(f"  ✓ Written ({len(model_data)} bytes)")
        print(f"  ✓ SHA256: {FULL_MODEL_SHA256}")

    # Write normalization
    if not norm_valid or args.force:
        print(f"\nWriting: {norm_path}")
        if not write_asset(norm_path, norm_data, FULL_NORM_SHA256):
            return 1
        print(f"  ✓ Written ({len(norm_data)} bytes)")
        print(f"  ✓ SHA256: {FULL_NORM_SHA256}")

    print("\n✓ Asset materialization complete")
    print(f"\nTotal materialized: {len(model_data) + len(norm_data)} bytes")

    return 0

if __name__ == "__main__":
    sys.exit(main())
