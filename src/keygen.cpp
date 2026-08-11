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

// FHE Key Generation Tool
// Generates crypto context, public/secret keys, and relinearization key
// Supports multiple datasets and custom output directories
//
// Usage: ./keygen [OPTIONS]
//   --profile {dataset}  : Dataset name (default: full)
//   --output-dir <path>  : Output directory (default: Mirai_Workload_Inputs)
//
// Datasets (crypto parameters):
//   full, mirai, unit:
//          Standard (ring_dim=65536, batch_size=32768, mult_depth=22)
//          All share same crypto parameters - can reuse keys!
//
// Examples:
//   ./keygen --profile unit --output-dir Mirai_Workload_Inputs

#include <chrono>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include "openfhe.h"
#include "cryptocontext-ser.h"
#include "ciphertext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;
namespace fs = std::filesystem;

std::string get_arg(int argc, char** argv, const std::string& flag, const std::string& default_val) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], flag.c_str()) == 0) {
            if (i + 1 < argc) return argv[i + 1];
        }
    }
    return default_val;
}

std::string get_profile(int argc, char** argv) {
    // Check command-line first
    std::string profile = get_arg(argc, argv, "--profile", "");
    if (profile.empty()) profile = get_arg(argc, argv, "-p", "");

    // Fallback to env var
    if (profile.empty()) {
        const char* env = std::getenv("NIOBIUM_PROFILE");
        profile = env ? env : "full";  // Default: full
    }
    return profile;
}

int main(int argc, char* argv[]) {
    std::string profile = get_profile(argc, argv);
    std::string output_dir = get_arg(argc, argv, "--output-dir", "Mirai_Workload_Inputs");

    // Normalize profiles for crypto parameter selection
    // All profiles use standard params (ring_dim=65536): full, mirai, unit
    std::string normalized_profile;
    if (profile == "full" || profile == "mirai" || profile == "unit") {
        normalized_profile = "standard";  // All use same crypto params
    } else {
        std::cerr << "ERROR: Invalid dataset '" << profile << "'." << std::endl;
        std::cerr << "Available datasets: full, mirai, unit" << std::endl;
        return 1;
    }

    // Create output directory
    fs::create_directories(output_dir);

    std::cout << "============================================================" << std::endl;
    std::string profile_upper = profile;
    std::transform(profile_upper.begin(), profile_upper.end(), profile_upper.begin(), ::toupper);
    std::cout << "FHE Key Generation - " << profile_upper << " Dataset" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Output directory: " << output_dir << std::endl;
    std::cout << std::endl;

    // Set CKKS parameters based on profile
    CCParams<CryptoContextCKKSRNS> parameters;

    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingModSize(54);           // Matches original setup
    parameters.SetScalingTechnique(FLEXIBLEAUTO);

    // STANDARD params for all profiles (full, mirai, unit): ring_dim=65536,
    // shared keys across datasets, matching networkmonitor.py training parameters.
    parameters.SetSecurityLevel(HEStd_128_classic);
    parameters.SetMultiplicativeDepth(22);  // Required for KitNET Chebyshev approximations (see commit 4171d4a)
    parameters.SetRingDim(65536);           // N = 2^16
    parameters.SetBatchSize(32768);         // 2^15 slots

    std::cout << "CKKS Parameters:" << std::endl;
    std::cout << "  Dataset: " << profile << std::endl;
    // All profiles are 50 features
    int features = 50;
    std::cout << "  Features: " << features << std::endl;
    std::cout << "  Batch Size: " << parameters.GetBatchSize() << std::endl;
    std::cout << "  Ring Dimension: " << parameters.GetRingDim() << std::endl;
    std::cout << "  Multiplicative Depth: " << parameters.GetMultiplicativeDepth() << std::endl;
    std::cout << "  Scaling Modulus Size: 54" << std::endl;
    std::cout << "  Scaling Technique: FLEXIBLEAUTO" << std::endl;
    std::cout << "  Security Level: " << "HEStd_128_classic" << std::endl;
    std::cout << std::endl;

    // Generate crypto context
    std::cout << "Generating crypto context..." << std::endl;
    auto t_gen_ctx = std::chrono::steady_clock::now();
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    std::cout << "[TIMING] gen_crypto_context_ms: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - t_gen_ctx).count()
              << std::endl;

    // Enable required features for KitNET FHE operations
    cc->Enable(PKE);           // Public key encryption
    cc->Enable(KEYSWITCH);     // Key switching for rotations
    cc->Enable(LEVELEDSHE);    // Leveled homomorphic operations
    cc->Enable(ADVANCEDSHE);   // Advanced operations (automorphisms, etc.)

    std::cout << "✓ Crypto context generated successfully" << std::endl;
    std::cout << "  Ring Dimension: " << cc->GetRingDimension() << std::endl;
    std::cout << "  Batch Size: " << cc->GetEncodingParams()->GetBatchSize() << std::endl;
    std::cout << std::endl;

    // Serialize crypto context
    fs::path cc_path = fs::path(output_dir) / "cryptocontext.bin";
    if (!Serial::SerializeToFile(cc_path.string(), cc, SerType::BINARY)) {
        std::cerr << "ERROR: Failed to serialize crypto context to " << cc_path << std::endl;
        return 1;
    }
    std::cout << "✓ Saved crypto context: " << cc_path << std::endl;

    // Generate a fresh key pair on every run (never reuse keys across runs).
    std::cout << std::endl;
    std::cout << "Generating FRESH key pair..." << std::endl;
    auto t_keygen = std::chrono::steady_clock::now();
    KeyPair<DCRTPoly> keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    std::cout << "[TIMING] keygen_ms: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - t_keygen).count()
              << std::endl;
    std::cout << "✓ Key pair generated (fresh random keys)" << std::endl;
    std::cout << "✓ Relinearization key generated" << std::endl;
    std::cout << std::endl;

    // Serialize relinearization key
    auto t_key_serialize = std::chrono::steady_clock::now();
    fs::path relin_path = fs::path(output_dir) / "relinearization_key.bin";
    std::ofstream relin_stream(relin_path, std::ios::out | std::ios::binary);
    if (!relin_stream.is_open()) {
        std::cerr << "ERROR: Failed to open file: " << relin_path << std::endl;
        return 1;
    }
    if (!cc->SerializeEvalMultKey(relin_stream, SerType::BINARY)) {
        std::cerr << "ERROR: Failed to serialize relinearization key" << std::endl;
        return 1;
    }
    relin_stream.close();

    // Get file sizes for reporting
    auto get_file_size = [](const fs::path& p) -> std::string {
        auto size = fs::file_size(p);
        if (size < 1024) return std::to_string(size) + " B";
        if (size < 1024*1024) return std::to_string(size/1024) + " KB";
        return std::to_string(size/(1024*1024)) + " MB";
    };

    std::cout << "✓ Saved relinearization key: " << relin_path
              << " (" << get_file_size(relin_path) << ")" << std::endl;

    // Serialize public key (for encryption tool)
    fs::path pk_path = fs::path(output_dir) / "public_key.bin";
    if (!Serial::SerializeToFile(pk_path.string(), keyPair.publicKey, SerType::BINARY)) {
        std::cerr << "ERROR: Failed to serialize public key to " << pk_path << std::endl;
        return 1;
    }
    std::cout << "✓ Saved public key: " << pk_path
              << " (" << get_file_size(pk_path) << ")" << std::endl;

    // Serialize secret key (for decryption)
    fs::path sk_path = fs::path(output_dir) / "secret_key.bin";
    if (!Serial::SerializeToFile(sk_path.string(), keyPair.secretKey, SerType::BINARY)) {
        std::cerr << "ERROR: Failed to serialize secret key to " << sk_path << std::endl;
        return 1;
    }
    std::cout << "✓ Saved secret key: " << sk_path
              << " (" << get_file_size(sk_path) << ")" << std::endl;
    std::cout << "[TIMING] key_serialize_ms: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - t_key_serialize).count()
              << std::endl;

    std::cout << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Key Generation Complete - " << "FULL" << " Profile" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Generated files in " << output_dir << ":" << std::endl;
    std::cout << "  - cryptocontext.bin       (crypto parameters)" << std::endl;
    std::cout << "  - relinearization_key.bin (for multiplication)" << std::endl;
    std::cout << "  - public_key.bin          (for encryption)" << std::endl;
    std::cout << "  - secret_key.bin          (for decryption)" << std::endl;
    std::cout << std::endl;
    std::cout << "NOTE: These are FRESH keys generated specifically for this run." << std::endl;
    std::cout << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "  Run the workload:  python3 harness/run_submission.py --profile <full|unit|mirai>" << std::endl;
    std::cout << std::endl;

    return 0;
}
