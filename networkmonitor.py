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

# Network Intrusion Detection — model training (in the clear).
#   Trains the KitNET model (an ensemble of autoencoders) from a PCAP capture and
#   saves the trained model + normalization parameters into Datasets/, for later
#   use by the FHE server. This is the offline, cleartext training step.
#
#   To RUN the workload under FHE (record, then replay on local or the Fog) use
#   harness/run_submission.py — not this script.
#
#   Usage: python3 networkmonitor.py <DataDir> -train [-pcap]

import sys
import json
from pathlib import Path
import argparse
import time
import numpy as np

import Features.FeatureExtractor as fe
import KitNET.NormStat as fn
import KitNET.KitNET as ad
import Features.processlabel as pl

def fixed_map():
    #Fixed mapping of features to autoencoders for FHE version of KitNET
    #only needed for training. Map is saved in the model parameter file for use during execution.
    fix_map = []
    tmpmap = [0,7,14,21,28,35,38,41,44,47] # packet counts
    fix_map.append(tmpmap)
    tmpmap = [1,4,8,11,15,18,22,25,29,32] #packet length mean and magnitude
    fix_map.append(tmpmap)
    tmpmap = [2,3,9,10,16,17,23,24,30,31]  #packet length variance and radius
    fix_map.append(tmpmap)
    tmpmap = [36,37,39,40,42,43,45,46,48,49]  #jitter mean and variance
    fix_map.append(tmpmap)
    tmpmap = [5,6,12,13,19,20,26,27,33,34]   #packet length covariance and pcc
    fix_map.append(tmpmap)
    return fix_map

def resolve_profile(data_path, sd):
    """Resolve profile from profiles.json or fall back to Datasets/{name}/ convention."""
    profiles_path = sd / "profiles.json"
    name_lower = str(data_path).lower()
    profile = None

    if profiles_path.exists():
        with open(profiles_path) as f:
            profiles = json.load(f).get("profiles", {})
        profile = profiles.get(name_lower)

    if profile is not None:
        data_source = profile["data_source"]
        base_path = sd / "Datasets" / data_source
        return {
            "features": profile["features"],
            "model_path": sd / profile["model"] if "model" in profile else base_path / f"{data_source}_model.bin",
            "norm_path": sd / profile["norm"] if "norm" in profile else base_path / f"{data_source}_norm.bin",
            "base_path": base_path,
            "batch_limit": profile.get("batch_limit"),
            "ring_dim": profile.get("ring_dim", 65536),
            "batch_size": profile.get("batch_size", 32768),
            "profile_name": name_lower,
        }
    else:
        # Convention fallback (original behavior)
        base_path = sd / "Datasets" / data_path
        name = str(data_path)
        return {
            "features": None,
            "model_path": base_path / f"{name}_model.bin",
            "norm_path": base_path / f"{name}_norm.bin",
            "base_path": base_path,
            "batch_limit": None,
            "ring_dim": 65536,
            "batch_size": 32768,
            "profile_name": None,
        }

def main(data_path: Path, mode: str, new_tsv: bool,
         hide_figs: bool = False, save_figs: bool = False, save_scores: bool = False):

    # Resolve profile configuration and create paths
    sourcedir = Path(__file__).resolve()    #raw data and NN model files are in the source directory
    sd = sourcedir.parents[0]
    prof = resolve_profile(data_path, sd)
    base_path = prof["base_path"]
    model_path = prof["model_path"]
    norm_path = prof["norm_path"]
    batch_limit = prof["batch_limit"]
    data_source = base_path.name  # preserves case for file naming (e.g., "Mirai")

    # Training guard for profiles with pre-built models
    if mode == 'train' and prof["profile_name"] in ("full",):
        raise RuntimeError(f"Profile '{prof['profile_name']}' uses a pre-built model. Training not supported.")

    # Labels and pcap use data_source (e.g., "Mirai" when profile is "full")
    label_name = data_source + "_labels.csv"
    label_path = base_path / label_name

    if new_tsv:
        file_name = data_source + "_pcap.pcap"
        file_path = base_path / file_name
    else:
        file_name = data_source + "_pcap.tsv"
        file_path = base_path / file_name
        if not file_path.exists():  # tsv file does not exist, so try pcap file
            print(f"File: {file_name} does not exist. Generating from pcap")
            new_tsv = True
            file_name = data_source + "_pcap.pcap"
            file_path = base_path / file_name

    packet_limit = np.inf #the number of packets to process

    # Read label file and calculate training grace periods
    LF = pl.labelParse(label_path)
    totalpkts = LF.read_labels()
    firstmalpkt = LF.get_firstmal()
    totaltrain = int(np.floor(0.5*firstmalpkt))
    totaltest = totalpkts - totaltrain
    ADgrace = int(np.floor(0.9*totaltrain))
    FNgrace = totaltrain - ADgrace
    print("Pkts for FN = " + str(FNgrace) + "  Pkts for AE = " + str(ADgrace) + " first malicious = " + str(firstmalpkt) + "  total test = " + str(totaltest), flush=True, file=sys.stderr)

    # model_path and norm_path already resolved by profile

    # Create feature extraction, feature normalization, and anomaly detector (KitNET) objects
    FE = fe.featExt(file_path, packet_limit)
    num_feat = FE.get_num_features()
    # Use profile feature count for encryption, or all features from data
    encrypt_features = prof["features"] if prof["features"] is not None else num_feat
    if (mode == 'train'):
        #create empty models
        FN = fn.normStat(num_feat)
        AD = ad.KitNET(num_feat, feature_map=fixed_map())

        ii = 0
        start = time.time()
       
        print("Start training Anomaly Detector from "+str(file_name), end='')
        while (ii < totaltrain):
            ii += 1
            if ii % 1000 == 0:
                print('.', end = '', flush=True)
                
            [x,_,_] = FE.get_next_vector()

            if (len(x)==0):
                #no packets left
                break

            if (ii < FNgrace):
                # train the normalization parameters
                FN.update(x)
                rmse = 0.0
            else:
                # train the NN model
                xhat = FN.get_logisticnorm(x)
                rmse = AD.train(xhat)

            if rmse == -1:
                break

        FE.close_tsv()
        print('Training complete on ' + str(ii) + ' samples. \nWriting normalization and model parameters') 
        FN.save_parm(norm_path)
        AD.save_model(model_path)

        stop = time.time()
        print("Time elapsed: "+ str(stop - start))

# Main script activation
# accept operator inputs and start the main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_data", help="Name of data directory", type=Path)
    parser.add_argument("-train", help="train a new model", action = 'store_true')
    parser.add_argument("-pcap", help="generate tsv from pcap. only valid with the -train option", action='store_true')

    args = parser.parse_args()
    if not args.train:
        sys.exit("networkmonitor.py trains the model (-train). To run the workload "
                 "under FHE (record/replay on local or the Fog), use "
                 "harness/run_submission.py.")
    main(args.path_to_data, 'train', args.pcap)
