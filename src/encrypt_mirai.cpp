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

// Encrypt Mirai features - encrypts the FULL 50-feature workload
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <cstdlib>
#include "openfhe.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"

using namespace lbcrypto;
namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    int features_to_encrypt = 50;  // FULL: 50 features
    std::string keys_dir = "Mirai_Workload_Inputs";
    std::string features_file = "assets/datasets/Mirai_first_batch_32K.bin";
    std::string output_dir = "Mirai_Workload_Inputs";

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg.rfind("--input-file=", 0) == 0) {
            features_file = arg.substr(13);
        } else if (arg.rfind("--output-dir=", 0) == 0) {
            output_dir = arg.substr(13);
        } else if (arg.rfind("--keys-dir=", 0) == 0) {
            keys_dir = arg.substr(11);
        } else {
            features_to_encrypt = std::atoi(argv[i]);
            if (features_to_encrypt != 50) {
                std::cerr << "ERROR: Number of features must be 50 (FULL)" << std::endl;
                std::cerr << "Usage: " << argv[0] << " [num_features] [--input-file=PATH] [--output-dir=DIR] [--keys-dir=DIR]" << std::endl;
                return 1;
            }
        }
    }

    const char* mode_name = "FULL";
    std::cout << "=== " << mode_name << " Mirai Encryption (" << features_to_encrypt << " features) ===" << std::endl;

    std::cout << "Encrypting " << features_to_encrypt << " features per packet" << std::endl;

    // Load crypto context
    auto t_key_load = std::chrono::steady_clock::now();
    CryptoContext<DCRTPoly> cc;
    if (!Serial::DeserializeFromFile(keys_dir + "/cryptocontext.bin", cc, SerType::BINARY)) {
        std::cerr << "Failed to load crypto context" << std::endl;
        return 1;
    }
    std::cout << "✓ Loaded crypto context" << std::endl;

    // Load public key
    PublicKey<DCRTPoly> pk;
    if (!Serial::DeserializeFromFile(keys_dir + "/public_key.bin", pk, SerType::BINARY)) {
        std::cerr << "Failed to load public key" << std::endl;
        return 1;
    }
    std::cout << "✓ Loaded public key" << std::endl;
    std::cout << "[TIMING] key_load_ms: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - t_key_load).count()
              << std::endl;

    // Determine packet count based on profile
    // FULL (50 features): ring_dim=65536 → batch_size=32768 packets
    const int num_packets = (features_to_encrypt == 2) ? 1024 : 32768;
    const int num_features = 50;

    std::ifstream fin(features_file, std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to open features file: " << features_file << std::endl;
        return 1;
    }

    std::vector<double> all_data(num_packets * num_features);
    fin.read(reinterpret_cast<char*>(all_data.data()),
             num_packets * num_features * sizeof(double));
    fin.close();

    std::cout << "✓ Read " << num_packets << " packets × " << num_features << " features" << std::endl;
    if (features_to_encrypt == 2) {
    }
    std::cout << "  First feature of packet 0: " << all_data[0] << std::endl;
    std::cout << "  First feature of packet 50: " << all_data[50 * num_features] << std::endl;

    // Create output directory
    fs::create_directories(output_dir);

    // Encrypt specified number of features
    double percentage = (features_to_encrypt * 100.0) / num_features;
    std::cout << "\n✓ Encrypting " << features_to_encrypt << " features ("
              << (int)percentage << "% of full workload)..." << std::endl;

    auto t_encrypt = std::chrono::steady_clock::now();
    for (int feat = 0; feat < features_to_encrypt; feat++) {
        // Extract this feature from all packets
        std::vector<double> feature_vector(num_packets);
        for (int pkt = 0; pkt < num_packets; pkt++) {
            feature_vector[pkt] = all_data[pkt * num_features + feat];
        }

        // Encode and encrypt
        Plaintext ptxt = cc->MakeCKKSPackedPlaintext(feature_vector);
        auto ctxt = cc->Encrypt(pk, ptxt);

        // Serialize
        std::string filename = output_dir + "/feature_ciphertext_" + std::to_string(feat) + ".bin";
        Serial::SerializeToFile(filename, ctxt, SerType::BINARY);

        std::cout << "  Encrypted feature " << feat << " → " << filename << std::endl;
    }

    std::cout << "[TIMING] encrypt_ms: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - t_encrypt).count()
              << std::endl;

    // Copy crypto context and keys (only if source != destination)
    if (keys_dir != output_dir) {
        std::cout << "\n✓ Copying crypto materials..." << std::endl;
        fs::copy_file(keys_dir + "/cryptocontext.bin", output_dir + "/cryptocontext.bin",
                      fs::copy_options::overwrite_existing);
        fs::copy_file(keys_dir + "/relinearization_key.bin", output_dir + "/relinearization_key.bin",
                      fs::copy_options::overwrite_existing);
        fs::copy_file(keys_dir + "/secret_key.bin", output_dir + "/secret_key.bin",
                      fs::copy_options::overwrite_existing);
        std::cout << "  Copied cryptocontext, relin key, secret key" << std::endl;
    } else {
        std::cout << "\n✓ Crypto materials already in output directory" << std::endl;
    }

    // Model and norm files should already exist in Mirai_Workload_Inputs
    // (placed there by unit_keygen_fullscale or copied manually)
    std::cout << "\n✓ Model and normalization files assumed present in " << output_dir << std::endl;

    std::cout << "\n=== " << mode_name << " WORKLOAD READY! ===" << std::endl;
    std::cout << "Location: " << output_dir << "/" << std::endl;
    std::cout << "Features: " << features_to_encrypt << " (vs 50 full)" << std::endl;

    std::cout << "\nNext step: run the workload via harness/run_submission.py" << std::endl;

    return 0;
}
