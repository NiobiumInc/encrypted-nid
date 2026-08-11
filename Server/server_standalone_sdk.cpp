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
//
// server_standalone_sdk — the FHE server stage: encrypted KitNET (NID) inference.
//
// Linked against the niobium-client SDK (libnbfhetch). On a cache miss it records
// the computation trace; on a cache hit replay() ships the trace over the FHETCH
// transport (in-process sim, a replay server, or the Fog) and returns the
// encrypted result. Uses the niobium-fhetch API: capture_crypto_context /
// tag_input / tag_keys. See NIOBIUM_INTEGRATION.md.

#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>

// Helper: emit [TIMING] line compatible with the benchmark tooling.
static auto now_ms() { return std::chrono::high_resolution_clock::now(); }
static long long elapsed_ms(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t0).count();
}
static void emit_timing(const char* key, long long ms) {
    std::cout << "[TIMING] " << key << ": " << ms << std::endl;
}

#include "openfhe.h"
#include "cryptocontext-ser.h"
#include "ciphertext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

#include "Kitnet.h"
#include "Packetdata.h"
#include "tobytes.h"

#ifdef NIOBIUM_COMPILER
#include "niobium/compiler.h"   // resolved from niobium-fhetch/include (SDK version)
#else
#error "server_standalone_sdk.cpp must be built with NIOBIUM_COMPILER"
#endif

using namespace lbcrypto;

int main(int argc, char *argv[])
{
    // init() consumes Niobium flags (--target / -O / --opt-level); the target
    // drives the hardware data format (no --niobium_hw).
    niobium::compiler().init(argc, argv);

    // Profile selection: env var first, overridable by --profile.
    const char* profile_env = std::getenv("NIOBIUM_PROFILE");
    std::string profile = profile_env ? profile_env : "full";
    std::string input_dir;            // empty => default Mirai_Workload_Inputs/
    std::string model_path_override;  // empty => derive from profile
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg.rfind("--input-dir=", 0)  == 0) { input_dir = arg.substr(12);
            std::cout << "Using input directory: " << input_dir << std::endl; }
        else if (arg.rfind("--profile=", 0)    == 0) { profile = arg.substr(10);
            std::cout << "Using profile: " << profile << std::endl; }
        else if (arg.rfind("--model-path=", 0) == 0) { model_path_override = arg.substr(13);
            std::cout << "Using model path: " << model_path_override << std::endl; }
        else if (arg == "--niobium_hw") { /* no-op: target drives HW format */ }
        else if (arg == "--target" && i + 1 < argc) { ++i; /* consumed by init() */ }
        else if (arg == "--opt-level" && i + 1 < argc) { ++i; /* consumed by init() */ }
        else if (arg.size() == 3 && arg.substr(0, 2) == "-O") { /* consumed by init() */ }
    }

    std::string workload_name = "NID_Mirai_" + profile + "_workload";
    niobium::compiler().set_program_info(
        workload_name.c_str(), "1.0",
        "Network intrusion detection workload (single batch, SDK)");
    niobium::compiler().set_build_info(__FILE__, __LINE__, __TIMESTAMP__);

    niobium::Compiler::CacheParameters params = {{"profile", profile}};
    niobium::compiler().cache_parameters(params);

    // ── Load the precomputed KitNET model ────────────────────────────
    std::filesystem::path cwd = std::filesystem::current_path();
    std::filesystem::path model_file_path;
    if (!model_path_override.empty()) {
        auto mp = std::filesystem::path(model_path_override);
        model_file_path = mp.is_relative() ? cwd / mp : mp;
    } else if (profile == "unit")       model_file_path = cwd / "Datasets/Unit/Unit_model.bin";
    else if (profile == "mirai")        model_file_path = cwd / "Datasets/Mirai/Mirai_model.bin";
    else                                model_file_path = cwd / "assets/models/Mirai_model_FULL.bin";

    std::unique_ptr<KitNET> model;
    {
        auto _t = now_ms();
        std::ifstream sourcefile(model_file_path, std::ios::binary | std::ios::in);
        if (!sourcefile.is_open()) {
            std::cerr << "Unable to load model file from: " << model_file_path << std::endl;
            return 1;
        }
        model.reset(new KitNET(sourcefile));
        emit_timing("model_load_ms", elapsed_ms(_t));
    }
    std::cout << "[NID_WORKLOAD] Loaded model file from: " << model_file_path << std::endl;

    // ── Server-side crypto material (share directory) ────────────────
    std::filesystem::path share_path = cwd / "Mirai_Workload_Inputs";
    if (!std::filesystem::is_directory(share_path)) {
        std::cerr << "Share directory is not a directory: " << share_path << std::endl;
        return 1;
    }
    std::filesystem::path cryptocontext_path        = share_path / "cryptocontext.bin";
    std::filesystem::path relinearization_key_path  = share_path / "relinearization_key.bin";

    CryptoContext<DCRTPoly> cryptocontext;
    {
        auto _t = now_ms();
        cryptocontext->ClearEvalMultKeys();
        cryptocontext->ClearEvalAutomorphismKeys();
        CryptoContextFactory<DCRTPoly>::ReleaseAllContexts();
        if (!Serial::DeserializeFromFile(cryptocontext_path, cryptocontext, SerType::BINARY)) {
            std::cerr << "ERROR: cannot read cryptocontext: " << cryptocontext_path << std::endl;
            return 1;
        }
        cryptocontext->Enable(PKE);
        cryptocontext->Enable(KEYSWITCH);
        cryptocontext->Enable(LEVELEDSHE);
        cryptocontext->Enable(ADVANCEDSHE);
        emit_timing("cc_load_ms", elapsed_ms(_t));
    }

    // Deserialize the relinearization (EvalMult) key into the context; tag_keys()
    // below captures it from the loaded context for the transport request.
    {
        auto _t = now_ms();
        std::ifstream rk(relinearization_key_path, std::ios::in | std::ios::binary);
        if (!rk.is_open()) {
            std::cerr << "ERROR: cannot open relin key: " << relinearization_key_path << std::endl;
            return 1;
        }
        if (!cryptocontext->DeserializeEvalMultKey(rk, SerType::BINARY)) {
            std::cerr << "ERROR: cannot deserialize relin key: " << relinearization_key_path << std::endl;
            return 1;
        }
        emit_timing("eval_key_load_ms", elapsed_ms(_t));
    }
    std::cout << "[NID_WORKLOAD] Loaded relinearization key" << std::endl;

    // ── Niobium (SDK) capture: context first ─────────────────────────
    niobium::compiler().capture_crypto_context(cryptocontext);

    int model_num_feat = model->get_num_feat();
    int actual_features = model_num_feat;

    std::filesystem::path feature_input_dir = share_path;
    if (!input_dir.empty()) {
        feature_input_dir = std::filesystem::path(input_dir);
        if (feature_input_dir.is_relative()) feature_input_dir = cwd / feature_input_dir;
    }

    // ── Load + tag the encrypted feature ciphertexts (inputs) ────────
    std::vector<Ciphertext<DCRTPoly>> feature_ciphertexts(model_num_feat);
    {
        auto _t = now_ms();
        for (int feature = 0; feature < actual_features; ++feature) {
            auto ct_path = feature_input_dir /
                ("feature_ciphertext_" + std::to_string(feature) + ".bin");
            std::string name = "input_" + std::to_string(feature);
            if (std::filesystem::exists(ct_path)) {
                if (!Serial::DeserializeFromFile(ct_path, feature_ciphertexts[feature], SerType::BINARY)) {
                    std::cerr << "ERROR: cannot read " << ct_path << std::endl;
                    return 1;
                }
                niobium::compiler().tag_input(name, feature_ciphertexts[feature], ct_path);
            } else if (!niobium::compiler().is_cache_valid()) {
                std::cerr << "ERROR: input file not found during recording: " << ct_path << std::endl;
                return 1;
            }
        }
        emit_timing("input_load_ms", elapsed_ms(_t));
    }
    std::cout << "[NID_WORKLOAD] Loaded feature ciphertexts" << std::endl;

    // Keys last (order: context -> inputs -> keys).
    niobium::compiler().tag_keys(cryptocontext);

    // ── Record XOR replay ────────────────────────────────────────────
    Ciphertext<DCRTPoly> score;
    if (!niobium::compiler().is_cache_valid()) {
        std::cout << "[compute-sdk] recording..." << std::endl;
        // Record in hollow mode; the verified result comes from replay.
        // Must be OFF for probe()/stop().
        niobium::compiler().enable_hollow_mode(true);
        niobium::compiler().start();

        auto _t = now_ms();
        if (!model->execute_ckks(cryptocontext, feature_ciphertexts, score)) {
            std::cerr << "ERROR: FHE execute_ckks failed" << std::endl;
            return 1;
        }
        emit_timing("compute_ms", elapsed_ms(_t));

        niobium::compiler().enable_hollow_mode(false);   // MUST be OFF for probe/stop
        niobium::compiler().probe("output_result", score);
        niobium::compiler().stop();

        // Placeholder result; the verified output comes from replay
        // (decrypt --use-replay). Written for API symmetry.
        auto out = share_path / "score_ciphertext_fhe.bin";
        if (!Serial::SerializeToFile(out, score, SerType::BINARY)) {
            std::cerr << "ERROR: cannot write " << out << std::endl;
            return 1;
        }
        std::cout << "[compute-sdk] record done -> " << out << std::endl;
    } else {
        std::cout << "[compute-sdk] replaying over transport..." << std::endl;
        auto _t = now_ms();
        if (!niobium::compiler().replay()) {
            std::cerr << "[compute-sdk] replay failed" << std::endl;
            return 1;
        }
        if (!niobium::compiler().result(cryptocontext, "output_result", score)) {
            std::cerr << "[compute-sdk] result() failed" << std::endl;
            return 1;
        }
        emit_timing("replay_ms", elapsed_ms(_t));

        auto out = share_path / "score_ciphertext_replay.bin";
        if (!Serial::SerializeToFile(out, score, SerType::BINARY)) {
            std::cerr << "ERROR: cannot write " << out << std::endl;
            return 1;
        }
        std::cout << "[compute-sdk] replay done -> " << out << std::endl;
    }

    std::cout << "[NID_WORKLOAD] Single batch server computation completed" << std::endl;
    return 0;
}