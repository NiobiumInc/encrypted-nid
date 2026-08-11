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
"""Cleartext KitNET **reference** scorer — the TRUE model, EXACT activations.

This is a REFERENCE, not the FHE production path. It runs the Python KitNET
(`KitNET/`) with the Chebyshev approximation disabled (`params.chebypoly = None`),
so the autoencoders use exact `sigmoid`/`tanh` — i.e. the model as trained, with
none of the FHE-related polynomial approximation. It does NOT touch the C++
server/decrypt stages.

For a run artifact it writes `reference_batch{N}.csv` (Packet,MSE,RMSE) next to the
harness's `scores_batch{N}.csv` (FPGA FHE), so the report can compare the encrypted
FPGA result against the true cleartext model.

Usage:
    python3 scripts/plaintext_reference.py runs/<id> [--profile mirai] [--nid-root .]
"""
import argparse
import contextlib
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np


def _score_batch(nid_root: Path, model_path: Path, data_bin: Path, num_feat: int):
    sys.path.insert(0, str(nid_root))
    import KitNET.KitNET as ad
    with contextlib.redirect_stdout(io.StringIO()):        # hush load_model's header print
        K = ad.KitNET(model_file=model_path)
        for ae in K.ensembleLayer:                          # drop Chebyshev -> exact sigmoid
            ae.params.chebypoly = None
        K.outputLayer.params.chebypoly = None               # -> exact tanh
    data = np.fromfile(str(data_bin), dtype=np.float64).reshape(-1, num_feat)
    return np.array([K.execute(data[i]) for i in range(data.shape[0])])   # RMSE per packet


def _batch_data_files(nid_root: Path, cfg: dict, n_batches: int):
    """Mirror harness batch_files(): full = single committed batch; multi-batch =
    assets/datasets/<src>_batch_<b>.bin."""
    if n_batches <= 1:
        df = cfg.get("data_file", "assets/datasets/Mirai_first_batch_32K.bin")
        return [nid_root / df]
    src = cfg.get("data_source", "Mirai")
    return [nid_root / f"assets/datasets/{src}_batch_{b}.bin" for b in range(n_batches)]


def main():
    ap = argparse.ArgumentParser(description="Cleartext (exact-activation) KitNET reference scorer.")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--profile", default=None, help="defaults to run.json's profile")
    ap.add_argument("--nid-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = ap.parse_args()
    nid_root, run = args.nid_root.resolve(), args.run_dir

    profile = args.profile
    if profile is None:
        profile = json.loads((run / "run.json").read_text())["profile"]
    cfg = json.loads((nid_root / "profiles.json").read_text())["profiles"][profile]

    # one reference per FPGA scores_batch present in the artifact
    n_batches = len(list(run.glob("scores_batch*.csv"))) or cfg.get("batch_limit", 1)
    model = cfg.get("model") or f"Datasets/{cfg.get('data_source','Mirai')}/{cfg.get('data_source','Mirai')}_model.bin"
    model_path = nid_root / model
    num_feat = int(cfg.get("features", 50))

    data_files = _batch_data_files(nid_root, cfg, n_batches)
    print(f"[reference] {profile}: exact-activation cleartext scores for {n_batches} batch(es)")
    for b in range(n_batches):
        rmse = _score_batch(nid_root, model_path, data_files[b], num_feat)
        out = run / f"reference_batch{b}.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["Packet", "MSE", "RMSE"])
            for i, r in enumerate(rmse):
                w.writerow([i, r * r, r])
        print(f"  batch {b}: {len(rmse):,} packets -> {out.name}")


if __name__ == "__main__":
    main()