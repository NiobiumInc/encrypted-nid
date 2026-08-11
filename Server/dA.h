// Denoising autoencoder (dA): C++ port adapted from Yusuke Sugomori's dA
// (https://github.com/yusugomori/DeepLearning), via Kitsune (ymirsky/Kitsune-py).
// Original portions Copyright (c) 2017 Yusuke Sugomori, MIT License; see Server/LICENSE.txt.
//
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

// C++ class definiion of the autoencoder

#pragma once
#ifndef AEMODEL   
#define AEMODEL

#include <vector>
#include <math.h>
#include <algorithm>

class dA
{
public:
    dA(int n_input, int n_hidden, bool usetanh);
    ~dA(void);

    double * execute(double * input);
    void load_parameters(double * transform, double * hidden_bias, double * visible_bias);

    // Set Chebyshev coefficients to match the FHE path (OpenFHE polynomial).
    // Once set, execute() uses Chebyshev approximations instead of exact sigmoid/tanh.
    void set_chebyshev_coeffs(const std::vector<double>& coeffs, double a, double b);

    // NEW: read-only accessors for FHE path
    int  visible_dim() const { return n_visible; }
    int  hidden_dim()  const { return n_hidden;  }
    bool uses_tanh()   const { return tanh_nonlin; }

    // Raw parameter buffers (read-only). tform is column-major by hidden index:
    // tform[ii + jj*n_visible] == W(visible=ii, hidden=jj)
    const double* tform_data() const { return tform; }
    const double* hbias_data() const { return hbias; }
    const double* rbias_data() const { return rbias; }

    // NEW: peek helpers
    double peek_weight(int vis_idx, int hid_idx) const { return tform[vis_idx + hid_idx * n_visible]; }
    double peek_hbias(int hid_idx) const           { return hbias[hid_idx]; }
    double peek_vbias(int vis_idx) const           { return rbias[vis_idx]; }

private:

    int n_visible;
    int n_hidden;
    bool tanh_nonlin;
    double * tform;
    double * hbias;
    double * rbias;

    // compute buffers
    double * y;
    double * z;
    double * e;

    // Chebyshev approximation coefficients (empty = use exact activation)
    std::vector<double> cheb_coeffs;
    double cheb_a = 0.0;
    double cheb_b = 0.0;

    void get_hidden_values(double * input);
    void get_reconstructed_input(double * hidden);

    // Evaluate Chebyshev series at x, mapped from [a,b] to [-1,1] (Clenshaw algorithm)
    double eval_chebyshev(double x) const
    {
        x = std::clamp(x, cheb_a, cheb_b);
        double t = (2.0 * x - cheb_a - cheb_b) / (cheb_b - cheb_a);
        int n = (int)cheb_coeffs.size();
        double b2 = 0.0, b1 = 0.0;
        for (int i = n - 1; i > 0; --i)
        {
            double b_new = cheb_coeffs[i] + 2.0 * t * b1 - b2;
            b2 = b1;
            b1 = b_new;
        }
        return cheb_coeffs[0] / 2.0 + t * b1 - b2;
    }

    void vect_activation(int len, double * x)
    {
        if (!cheb_coeffs.empty())
        {
            for (int ii = 0; ii < len; ++ii)
                x[ii] = eval_chebyshev(x[ii]);
        }
        else if (tanh_nonlin)
        {
            for (int ii = 0; ii < len; ++ii)
                x[ii] = tanh(x[ii]);
        }
        else
        {
            for (int ii = 0; ii < len; ++ii)
                x[ii] = 1.0 / (1.0 + exp(-x[ii]));
        }
    }

};
#endif
