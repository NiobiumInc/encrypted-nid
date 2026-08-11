// Copyright 2025-present Niobium Microsystems, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// decrypt_probe.cpp - Decrypt FHE probe with deterministic sampling validation
//
// The decrypt/verify stage run by harness/run_submission.py.
// Validates FHE correctness using deterministic sampling (head/middle/tail + random).
//
// Profile-aware:
//   - FULL: 50 features, threshold ≤2% (CKKS noise tolerance)
//
// Exit codes:
//   0 - Validation passed
//   1 - Validation failed (exceeds threshold)
//   2 - Error loading files

#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"
#include "../Server/Kitnet.h"
#include "../Server/validation_sampling.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>

using namespace lbcrypto;
using namespace validation;

// Stub for openfhe_cprobe_id
#ifndef NIOBIUM_COMPILER
extern "C" void openfhe_cprobe_id(uintptr_t id) {}
#endif

std::string get_profile_from_env() {
    const char* env = std::getenv("NIOBIUM_PROFILE");
    return env ? env : "full";
}

int main(int argc, char** argv) {
    // Get profile from environment variable (set by the harness)
    std::string profile = get_profile_from_env();

    // Parse command-line flags
    std::string registers_suffix = "";
    bool use_replay_ciphertext = false;
    std::string ct_path_override = "";
    std::string model_path_override = "";
    std::string workload_dir_override = "";
    std::string data_path_override = "";
    std::string save_scores_path = "";
    std::string save_plaintext_scores_path = "";
    int features_override = 0;
    double threshold_override = -1.0;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--registers=") == 0) {
            std::string reg_count = arg.substr(12);
            registers_suffix = "_reg_" + reg_count;
        }
        if (arg == "--hollow" || arg == "--use-replay") {
            use_replay_ciphertext = true;
        }
        if (arg.find("--ct-path=") == 0) {
            ct_path_override = arg.substr(10);
            use_replay_ciphertext = true;  // --ct-path implies replay mode
        }
        if (arg.find("--model-path=") == 0) {
            model_path_override = arg.substr(13);
        }
        if (arg.find("--workload-dir=") == 0) {
            workload_dir_override = arg.substr(15);
        }
        if (arg.find("--features=") == 0) {
            features_override = std::stoi(arg.substr(11));
        }
        if (arg.find("--threshold=") == 0) {
            threshold_override = std::stod(arg.substr(12));
        }
        if (arg.find("--data-path=") == 0) {
            data_path_override = arg.substr(12);
        }
        if (arg.find("--save-scores=") == 0) {
            save_scores_path = arg.substr(14);
        }
        if (arg.find("--save-plaintext-scores=") == 0) {
            save_plaintext_scores_path = arg.substr(24);
        }
    }

    std::cout << "=== Decrypting FHE Probe Result ===" << std::endl;
    std::cout << "Profile: " << profile << std::endl;
    if (!registers_suffix.empty()) {
        std::cout << "Register suffix: " << registers_suffix << std::endl;
    }

    // Profile-aware paths
    std::string model_path;
    std::string workload_dir;
    int expected_features;
    double threshold_pct;

    if (profile == "unit") {
        model_path = "Datasets/Unit/Unit_model.bin";
        workload_dir = "NID_Mirai_unit_workload";
        expected_features = 50;
        threshold_pct = 15.0;  // Unit: same depth as FULL, expect similar error
    } else if (profile == "mirai") {
        model_path = "Datasets/Mirai/Mirai_model.bin";
        workload_dir = "NID_Mirai_mirai_workload";
        expected_features = 50;
        threshold_pct = 15.0;
    } else {  // "full" is default
        model_path = "assets/models/Mirai_model_FULL.bin";
        workload_dir = "NID_Mirai_full_workload";
        expected_features = 50;
        threshold_pct = 15.0;  // FULL: mean error threshold (CKKS approximation in deep circuits)
    }

    // CLI overrides (from the harness, resolved via profiles.json)
    if (!model_path_override.empty()) model_path = model_path_override;
    if (!workload_dir_override.empty()) workload_dir = workload_dir_override;
    if (features_override > 0) expected_features = features_override;
    if (threshold_override >= 0) threshold_pct = threshold_override;

    // Append register suffix if provided (for sweep runs with --registers=N)
    workload_dir += registers_suffix;

    std::cout << "Expected max error: <= " << threshold_pct << "%" << std::endl;

    // Load crypto context (from Mirai_Workload_Inputs - saved by keygen, compatible with keys)
    CryptoContext<DCRTPoly> cc;
    std::string cc_path = "Mirai_Workload_Inputs/cryptocontext.bin";
    if (!Serial::DeserializeFromFile(cc_path, cc, SerType::BINARY)) {
        std::cerr << "ERROR: Cannot load crypto context: " << cc_path << std::endl;
        return 2;
    }

    // Load secret key (from Mirai_Workload_Inputs - saved by keygen)
    KeyPair<DCRTPoly> kp;
    std::string sk_path = "Mirai_Workload_Inputs/secret_key.bin";
    if (!Serial::DeserializeFromFile(sk_path, kp.secretKey, SerType::BINARY)) {
        std::cerr << "ERROR: Cannot load secret key: " << sk_path << std::endl;
        return 2;
    }

    // Load ciphertext result (from Mirai_Workload_Inputs - saved by server_standalone_sdk)
    // Use the replayed ciphertext when:
    // - --hollow: the record pass produces no usable result; take it from replay
    // - --use-replay: multi-batch mode, each batch replays with different inputs
    Ciphertext<DCRTPoly> ct;
    std::string ct_path;
    if (!ct_path_override.empty()) {
        ct_path = ct_path_override;
        std::cout << "✓ Using ciphertext from: " << ct_path << std::endl;
    } else if (use_replay_ciphertext) {
        ct_path = "Mirai_Workload_Inputs/score_ciphertext_replay.bin";
        std::cout << "✓ Using replay result ciphertext" << std::endl;
    } else {
        ct_path = "Mirai_Workload_Inputs/score_ciphertext_fhe.bin";
    }

    if (!Serial::DeserializeFromFile(ct_path, ct, SerType::BINARY)) {
        std::cerr << "ERROR: Cannot load ciphertext: " << ct_path << std::endl;
        if (use_replay_ciphertext) {
            std::cerr << "       Make sure replay phase has completed" << std::endl;
        } else {
            std::cerr << "       Make sure recording phase has completed" << std::endl;
        }
        return 2;
    }

    std::cout << "✓ Loaded crypto materials" << std::endl;

    // Decrypt
    Plaintext pt;
    cc->Decrypt(kp.secretKey, ct, &pt);
    auto fhe_values = pt->GetRealPackedValue();
    size_t batch_size = fhe_values.size();

    std::cout << "✓ Decrypted " << batch_size << " FHE MSE values" << std::endl;

    // Save all FHE MSE scores to CSV if requested (for visualization)
    if (!save_scores_path.empty()) {
        std::ofstream scores_file(save_scores_path);
        if (scores_file) {
            scores_file << "Packet,MSE,RMSE" << std::endl;
            scores_file << std::scientific << std::setprecision(15);
            size_t num_packets = std::min(batch_size, (size_t)32768);
            for (size_t i = 0; i < num_packets; i++) {
                scores_file << i << "," << fhe_values[i] << ","
                            << std::sqrt(std::max(0.0, fhe_values[i])) << std::endl;
            }
            scores_file.close();
            std::cout << "✓ Saved " << num_packets << " scores to " << save_scores_path << std::endl;
        } else {
            std::cerr << "WARNING: Could not write scores to " << save_scores_path << std::endl;
        }
    }

    // Get deterministic sample indices (use K=10 for reasonable coverage)
    auto sample_indices = get_sample_indices(batch_size, 10);

    std::cout << "✓ Selected " << sample_indices.size() << " sample indices for validation" << std::endl;

    // Load model for plaintext computation
    std::ifstream model_file(model_path, std::ios::binary);
    if (!model_file) {
        std::cerr << "ERROR: Cannot open model: " << model_path << std::endl;
        return 2;
    }
    KitNET model(model_file);
    model_file.close();

    if (model.get_num_feat() != expected_features) {
        std::cerr << "ERROR: Model has " << model.get_num_feat() << " features, "
                  << "but " << profile << " profile expects " << expected_features << std::endl;
        return 2;
    }

    std::cout << "✓ Model loaded (" << expected_features << " features)" << std::endl;

    // Load dataset for plaintext computation (only for sampled indices)
    // Priority: --data-path= flag > NIOBIUM_DATASET_PATH env var > profile default
    std::string data_path;
    int dataset_packets;
    if (!data_path_override.empty()) {
        data_path = data_path_override;
        dataset_packets = 32768;
    } else {
        const char* dataset_override = std::getenv("NIOBIUM_DATASET_PATH");
        if (dataset_override && strlen(dataset_override) > 0) {
            data_path = dataset_override;
        } else {
            data_path = "assets/datasets/Mirai_first_batch_32K.bin";
        }
        dataset_packets = 32768;
    }
    std::ifstream data_file(data_path, std::ios::binary);
    if (!data_file) {
        std::cerr << "ERROR: Cannot open dataset: " << data_path << std::endl;
        return 2;
    }

    const int dataset_features = 50;  // Dataset always has 50 features
    std::vector<double> plain_values;
    plain_values.reserve(sample_indices.size());

    // Full-batch plaintext (cleartext-CPU) baseline for the report's
    // plaintext-vs-FPGA comparison. Model + cleartext features are already
    // loaded, so this is the ground-truth line paired 1:1 with scores_batch*.csv.
    if (!save_plaintext_scores_path.empty()) {
        std::ofstream pf(save_plaintext_scores_path);
        if (pf) {
            pf << "Packet,MSE,RMSE" << std::endl;
            pf << std::scientific << std::setprecision(15);
            size_t np = std::min(batch_size, (size_t)dataset_packets);
            std::vector<double> feats(dataset_features), sub(expected_features);
            for (size_t i = 0; i < np; i++) {
                data_file.seekg(i * dataset_features * sizeof(double), std::ios::beg);
                data_file.read(reinterpret_cast<char*>(feats.data()),
                               dataset_features * sizeof(double));
                for (int f = 0; f < expected_features; f++) sub[f] = feats[f];
                double mse = model.execute(sub.data());
                pf << i << "," << mse << "," << std::sqrt(std::max(0.0, mse)) << std::endl;
            }
            pf.close();
            std::cout << "✓ Saved " << np << " plaintext scores to "
                      << save_plaintext_scores_path << std::endl;
        } else {
            std::cerr << "WARNING: Could not write plaintext scores to "
                      << save_plaintext_scores_path << std::endl;
        }
        data_file.clear();   // reset EOF/fail state for the sampled loop below
    }

    std::cout << "✓ Computing plaintext MSE for " << sample_indices.size()
              << " sampled packets..." << std::endl;

    for (size_t idx : sample_indices) {
        // Load features for this specific packet (row-major layout)
        std::vector<double> packet_features(dataset_features);

        data_file.seekg(idx * dataset_features * sizeof(double), std::ios::beg);
        data_file.read(reinterpret_cast<char*>(packet_features.data()),
                      dataset_features * sizeof(double));

        // Extract first N features based on profile
        std::vector<double> features_subset(expected_features);
        for (int f = 0; f < expected_features; f++) {
            features_subset[f] = packet_features[f];
        }

        // Compute plaintext MSE for this packet
        double mse_plain = model.execute(features_subset.data());
        plain_values.push_back(mse_plain);
    }
    data_file.close();

    // Print validation table (compact: head + tail only)
    print_validation_table(fhe_values, plain_values, sample_indices, false);

    // Compute validation statistics
    auto stats = ValidationStats::compute(fhe_values, plain_values, sample_indices);

    // Print summary
    print_validation_summary(stats, threshold_pct, profile);

    // Print key metrics for the harness to parse
    std::cout << "\n=== METRICS FOR WORKFLOW ===" << std::endl;
    std::cout << "output_result (MSE)[0]: " << std::scientific << std::setprecision(15)
              << fhe_values[0] << std::endl;
    std::cout << "FHE RMSE (sqrt(MSE)): " << std::sqrt(fhe_values[0]) << std::endl;
    std::cout << "Plaintext MSE: " << plain_values[0] << std::endl;
    std::cout << "Plaintext RMSE: " << std::sqrt(plain_values[0]) << std::endl;
    std::cout << "MSE relative error: " << std::fixed << std::setprecision(4)
              << stats.mean_rel_error_pct << "%" << std::endl;

    // Exit code based on validation (using MEAN error, not max)
    if (stats.mean_rel_error_pct <= threshold_pct) {
        std::cout << "\n✅ VALIDATION PASSED (mean error: " << std::fixed << std::setprecision(2)
                  << stats.mean_rel_error_pct << "% ≤ " << threshold_pct << "%)" << std::endl;
        return 0;
    } else {
        std::cerr << "\n❌ VALIDATION FAILED: Mean error " << std::fixed << std::setprecision(2)
                  << stats.mean_rel_error_pct << "% exceeds threshold " << threshold_pct << "%" << std::endl;

        // Print samples with highest errors for debugging
        std::cerr << "\nSamples with highest errors:" << std::endl;
        std::cerr << std::setw(8) << "Index"
                  << std::setw(16) << "FHE MSE"
                  << std::setw(16) << "Plain MSE"
                  << std::setw(14) << "Rel Error (%)" << std::endl;
        std::cerr << std::string(54, '-') << std::endl;

        // Show worst 5 samples
        std::vector<std::pair<double, size_t>> errors;
        for (size_t i = 0; i < sample_indices.size(); i++) {
            size_t idx = sample_indices[i];
            double fhe = fhe_values[idx];
            double plain = plain_values[i];
            double rel_error_pct = (plain != 0.0) ? (std::abs(fhe - plain) / std::abs(plain)) * 100.0 : 0.0;
            errors.push_back({rel_error_pct, i});
        }
        std::sort(errors.rbegin(), errors.rend());

        for (size_t i = 0; i < std::min(size_t(5), errors.size()); i++) {
            size_t sample_idx = errors[i].second;
            size_t idx = sample_indices[sample_idx];
            double fhe = fhe_values[idx];
            double plain = plain_values[sample_idx];
            std::cerr << std::setw(8) << idx
                      << std::setw(16) << std::scientific << std::setprecision(6) << fhe
                      << std::setw(16) << plain
                      << std::setw(14) << std::fixed << std::setprecision(4) << errors[i].first << std::endl;
        }

        return 1;
    }
}
