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

// validation_sampling.h - Deterministic sampling for FHE correctness validation
//
// Provides compact, reproducible validation across batch without printing 32K values.
// Used by: decrypt_probe.cpp

#ifndef VALIDATION_SAMPLING_H
#define VALIDATION_SAMPLING_H

#include <vector>
#include <set>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>

namespace validation {

/**
 * Get deterministic sample indices for validation
 *
 * Strategy:
 * - Head: First 10 packets [0..9]
 * - Middle: Around N/2 ± 5 packets
 * - Tail: Last 10 packets [N-10..N-1]
 * - Random: K additional deterministic samples (fixed seed=42)
 *
 * @param N Total batch size (e.g., 32768)
 * @param K Number of additional random samples (default 10)
 * @param seed Fixed seed for reproducibility (default 42)
 * @return Sorted vector of unique sample indices
 */
inline std::vector<size_t> get_sample_indices(
    size_t N,
    size_t K = 10,
    uint32_t seed = 42
) {
    if (N == 0) return {};

    std::set<size_t> indices;

    // Head: First 10 packets
    for (size_t i = 0; i < std::min(size_t(10), N); i++) {
        indices.insert(i);
    }

    // Middle: N/2 ± 5 packets
    if (N >= 20) {
        size_t mid = N / 2;
        for (int offset = -5; offset <= 5; offset++) {
            int64_t idx = static_cast<int64_t>(mid) + offset;
            if (idx >= 0 && idx < static_cast<int64_t>(N)) {
                indices.insert(static_cast<size_t>(idx));
            }
        }
    }

    // Tail: Last 10 packets
    size_t tail_start = (N > 10) ? (N - 10) : 0;
    for (size_t i = tail_start; i < N; i++) {
        indices.insert(i);
    }

    // Deterministic random samples (fixed seed for CI stability)
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, N - 1);

    size_t target_size = std::min(N, indices.size() + K);
    size_t attempts = 0;
    while (indices.size() < target_size && attempts < K * 3) {
        indices.insert(dist(rng));
        attempts++;
    }

    // Convert to sorted vector
    std::vector<size_t> result(indices.begin(), indices.end());
    std::sort(result.begin(), result.end());

    return result;
}

/**
 * Summary statistics for validation
 */
struct ValidationStats {
    size_t count;
    double min_fhe;
    double max_fhe;
    double mean_fhe;
    double std_fhe;
    double min_plain;
    double max_plain;
    double mean_plain;
    double std_plain;
    double max_abs_error;
    double max_rel_error_pct;
    double mean_rel_error_pct;

    static ValidationStats compute(
        const std::vector<double>& fhe_values,
        const std::vector<double>& plain_values,
        const std::vector<size_t>& sample_indices
    ) {
        ValidationStats stats = {};
        stats.count = sample_indices.size();

        if (sample_indices.empty()) return stats;

        // Initialize min/max
        stats.min_fhe = std::numeric_limits<double>::max();
        stats.max_fhe = std::numeric_limits<double>::lowest();
        stats.min_plain = std::numeric_limits<double>::max();
        stats.max_plain = std::numeric_limits<double>::lowest();
        stats.max_abs_error = 0.0;
        stats.max_rel_error_pct = 0.0;

        double sum_fhe = 0.0;
        double sum_plain = 0.0;
        double sum_rel_error = 0.0;

        // First pass: compute sums, min/max, errors
        for (size_t i = 0; i < sample_indices.size(); i++) {
            size_t idx = sample_indices[i];
            if (idx >= fhe_values.size() || i >= plain_values.size()) continue;

            double fhe = fhe_values[idx];
            double plain = plain_values[i];  // plain_values indexed by sample position

            stats.min_fhe = std::min(stats.min_fhe, fhe);
            stats.max_fhe = std::max(stats.max_fhe, fhe);
            stats.min_plain = std::min(stats.min_plain, plain);
            stats.max_plain = std::max(stats.max_plain, plain);

            sum_fhe += fhe;
            sum_plain += plain;

            double abs_error = std::abs(fhe - plain);
            stats.max_abs_error = std::max(stats.max_abs_error, abs_error);

            double rel_error_pct = (plain != 0.0) ? (abs_error / std::abs(plain)) * 100.0 : 0.0;
            stats.max_rel_error_pct = std::max(stats.max_rel_error_pct, rel_error_pct);
            sum_rel_error += rel_error_pct;
        }

        stats.mean_fhe = sum_fhe / stats.count;
        stats.mean_plain = sum_plain / stats.count;
        stats.mean_rel_error_pct = sum_rel_error / stats.count;

        // Second pass: compute std dev
        double sum_sq_fhe = 0.0;
        double sum_sq_plain = 0.0;

        for (size_t i = 0; i < sample_indices.size(); i++) {
            size_t idx = sample_indices[i];
            if (idx >= fhe_values.size() || i >= plain_values.size()) continue;

            double fhe = fhe_values[idx];
            double plain = plain_values[i];

            sum_sq_fhe += (fhe - stats.mean_fhe) * (fhe - stats.mean_fhe);
            sum_sq_plain += (plain - stats.mean_plain) * (plain - stats.mean_plain);
        }

        stats.std_fhe = std::sqrt(sum_sq_fhe / stats.count);
        stats.std_plain = std::sqrt(sum_sq_plain / stats.count);

        return stats;
    }
};

/**
 * Print compact validation table with sampled indices
 */
inline void print_validation_table(
    const std::vector<double>& fhe_values,
    const std::vector<double>& plain_values,
    const std::vector<size_t>& sample_indices,
    bool verbose = false
) {
    std::cout << "\n=== Sampled Validation (n=" << sample_indices.size() << ") ===" << std::endl;
    std::cout << std::setw(8) << "Index"
              << std::setw(16) << "FHE MSE"
              << std::setw(16) << "Plain MSE"
              << std::setw(14) << "Abs Error"
              << std::setw(14) << "Rel Error (%)" << std::endl;
    std::cout << std::string(68, '-') << std::endl;

    // Print all samples if verbose, otherwise print head/mid/tail with ellipsis
    size_t print_head = verbose ? sample_indices.size() : std::min(size_t(5), sample_indices.size());
    size_t print_tail = verbose ? 0 : std::min(size_t(5), sample_indices.size());

    for (size_t i = 0; i < print_head; i++) {
        size_t idx = sample_indices[i];
        if (idx >= fhe_values.size() || i >= plain_values.size()) continue;

        double fhe = fhe_values[idx];
        double plain = plain_values[i];
        double abs_error = std::abs(fhe - plain);
        double rel_error_pct = (plain != 0.0) ? (abs_error / std::abs(plain)) * 100.0 : 0.0;

        std::cout << std::setw(8) << idx
                  << std::setw(16) << std::scientific << std::setprecision(6) << fhe
                  << std::setw(16) << plain
                  << std::setw(14) << std::fixed << std::setprecision(8) << abs_error
                  << std::setw(14) << std::setprecision(4) << rel_error_pct << std::endl;
    }

    if (!verbose && sample_indices.size() > 10) {
        std::cout << "  ...     (showing first/last 5 only, use --verbose for all)" << std::endl;

        // Print last 5
        for (size_t i = sample_indices.size() - print_tail; i < sample_indices.size(); i++) {
            size_t idx = sample_indices[i];
            if (idx >= fhe_values.size() || i >= plain_values.size()) continue;

            double fhe = fhe_values[idx];
            double plain = plain_values[i];
            double abs_error = std::abs(fhe - plain);
            double rel_error_pct = (plain != 0.0) ? (abs_error / std::abs(plain)) * 100.0 : 0.0;

            std::cout << std::setw(8) << idx
                      << std::setw(16) << std::scientific << std::setprecision(6) << fhe
                      << std::setw(16) << plain
                      << std::setw(14) << std::fixed << std::setprecision(8) << abs_error
                      << std::setw(14) << std::setprecision(4) << rel_error_pct << std::endl;
        }
    }
}

/**
 * Print validation summary statistics
 */
inline void print_validation_summary(
    const ValidationStats& stats,
    double threshold_pct = 1.0,
    const std::string& profile = "full"
) {
    std::cout << "\n=== Validation Summary ===" << std::endl;
    std::cout << "Samples validated : " << stats.count << std::endl;
    std::cout << "\nFHE MSE:" << std::endl;
    std::cout << "  Min             : " << std::scientific << std::setprecision(6) << stats.min_fhe << std::endl;
    std::cout << "  Max             : " << stats.max_fhe << std::endl;
    std::cout << "  Mean            : " << stats.mean_fhe << std::endl;
    std::cout << "  Std Dev         : " << stats.std_fhe << std::endl;

    std::cout << "\nPlaintext MSE:" << std::endl;
    std::cout << "  Min             : " << stats.min_plain << std::endl;
    std::cout << "  Max             : " << stats.max_plain << std::endl;
    std::cout << "  Mean            : " << stats.mean_plain << std::endl;
    std::cout << "  Std Dev         : " << stats.std_plain << std::endl;

    std::cout << "\nError Metrics:" << std::endl;
    std::cout << "  Max abs error   : " << std::fixed << std::setprecision(8) << stats.max_abs_error << std::endl;
    std::cout << "  Mean rel error  : " << std::setprecision(4) << stats.mean_rel_error_pct << "%" << std::endl;

    // Note: Max error is informational only - pass/fail uses mean error
}

/**
 * Print sampling strategy explanation
 */
inline void print_sampling_info(size_t N, const std::vector<size_t>& sample_indices) {
    std::cout << "\n=== Sampling Strategy ===" << std::endl;
    std::cout << "Total batch size  : " << N << std::endl;
    std::cout << "Samples selected  : " << sample_indices.size() << std::endl;
    std::cout << "Coverage          : " << std::fixed << std::setprecision(2)
              << (sample_indices.size() * 100.0 / N) << "%" << std::endl;
    std::cout << "\nRegions sampled:" << std::endl;
    std::cout << "  - Head: indices [0..9]" << std::endl;
    std::cout << "  - Middle: indices [" << (N/2 - 5) << ".." << (N/2 + 5) << "]" << std::endl;
    std::cout << "  - Tail: indices [" << (N - 10) << ".." << (N - 1) << "]" << std::endl;
    std::cout << "  - Random: 10 deterministic indices (seed=42 for CI stability)" << std::endl;
}

} // namespace validation

#endif // VALIDATION_SAMPLING_H
