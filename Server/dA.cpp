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

#include "dA.h"

// This class defines the model parameters and execution of each individual
// perceptron (autoencoder or anomaly detector).  It accepts three parameters:
// the input dimension, the hidden dimension, and a flag to indicate whether a
// sigmoid or tanh nonlinearity should be used. Sigmoids are used for the autoencoders
// whose inputs have a range of 0 to +1. Tanhs are used for the anomaly detector whose
// inputs have a range of -1 to +1. 
// The cloass holds the transform and bias parameters, which are loaded from KitNET.
// It also declares several working vectors for computation. 

dA::dA(int input_dim, int hidden_dim, bool usetanh)
{
    n_visible = input_dim;
    n_hidden = hidden_dim;
    tanh_nonlin = usetanh;

    //allocate memory for parameter arrays
    tform = new double [n_visible*n_hidden];
    hbias = new double[n_hidden];
    rbias = new double[n_visible];

    //allocate memory for computation arrays
    y = new double[n_hidden];
    z = new double[n_visible];
    e = new double[n_visible];
}

dA::~dA(void)
{
    delete [] e;
    delete [] y;
    delete [] z;
    delete[] rbias;
    delete[] hbias;
    delete[] tform;
}

double * dA::execute(double * input)
{
    dA::get_hidden_values(input);
    dA::vect_activation(n_hidden, y);

    dA::get_reconstructed_input(y);
    dA::vect_activation(n_visible, z);

    for (int ii=0; ii<n_visible; ++ii)
    {
        e[ii] = input[ii] - z[ii];
    }

    return e;
}

void dA::set_chebyshev_coeffs(const std::vector<double>& coeffs, double a, double b)
{
    cheb_coeffs = coeffs;
    cheb_a = a;
    cheb_b = b;
}

void dA::load_parameters(double * transform, double * hidden_bias, double * reconstruct_bias)
{
    for (int ii=0; ii < n_visible; ++ii)
    {
        for (int jj=0; jj<n_hidden; ++jj)
        {
            tform[ii+jj*n_visible] = transform[ii*n_hidden+jj];
        }
        rbias[ii] = reconstruct_bias[ii];
    }

    for (int jj=0; jj <n_hidden; ++jj)
    {
        hbias[jj] = hidden_bias[jj];
    }
}

void dA::get_hidden_values(double * input)
{
    for (int jj=0; jj<n_hidden; ++jj)
    {
        y[jj] = hbias[jj];
        for (int ii=0; ii<n_visible; ++ii)
        {
            y[jj] += tform[ii+jj*n_visible]*input[ii];
        }
    }
    return;
}
    
void dA::get_reconstructed_input(double * hidden)
{
    for (int ii=0; ii<n_visible; ++ii)
    {
        z[ii] = rbias[ii];
        for (int jj=0; jj<n_hidden; ++jj)
        {
            z[ii] += tform[ii+jj*n_visible]*hidden[jj];
        }
    }
    return;
}
