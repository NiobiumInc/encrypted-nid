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

// Kitnet - definition of anomaly detector Neural network
// Consists of an ensemble of autoencoders followed by a 3 layer anomaly detector

#pragma once
#ifndef KITNET   
#define KITNET

#include <string>
#include <vector>
#include <ios>
#include <streambuf>
#include <iostream>
#include <fstream>
#include <math.h>

#include "dA.h"
#include "openfhe.h"   // NEW: for FHE path
#include "math/chebyshev.h"



// model header parameters
#define HDRVALS 7
#define VALSIZE 2
#define MAXAE 5

struct model_header
{
    short int numAE;
    short int numfeat;
    short int visAE;
    short int hidAE;
    short int visAD;
    short int hidAD;
    short int apxord;
};
union bytestoheader
{
    struct model_header modelpar;
    char raws[HDRVALS*VALSIZE];
};

class KitNET
{
public:

    KitNET(std::ifstream &file);
    ~KitNET(void);

    double execute(double * input);

    int get_num_feat()
    {
        return num_feat;
    }

    // NEW: Encrypted execution on ciphertexts (CPU path).
    // Ef[f] is ciphertext for feature f across all packets.
    // score_ct: ciphertext where each slot holds MSE for that packet.
    bool execute_ckks(
        const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
        const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& Ef,
        lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& score_ct
    ) const;



    void compute_chebyshev();

private:

    int num_feat;
    int num_ae;
    int ae_vis_dim;
    int ae_hid_dim;
    int ad_vis_dim;
    int ad_hid_dim;
    int apx_ord;

    std::vector<double> sig_coeffs;
    std::vector<double> tanh_coeffs;

    std::vector<double> fhesig_coeffs;
    std::vector<double> fhetanh_coeffs;

    std::vector<std::vector<int>> map;
    
    std::vector<dA*> ensemble_layer;
    dA* output_layer;

    double * errvec;
    double * xi;

    // NEW: cached anomaly-detector parameters for encrypted path (row-major [vis][hid])
    std::vector<double> ad_W;   // size = ad_vis_dim * ad_hid_dim
    std::vector<double> ad_hb;  // size = ad_hid_dim
    std::vector<double> ad_rb;  // size = ad_vis_dim

    // functions to be estimated with Chebyshev approximation
    const double as = -5.0, bs = 5.0;   // sigmoid (AEs)
    const double at = -2.0, bt =  2.0;  // tanh   (AD)

    static double base_sigmoid(double x)
    {
        return 1.0/(1.0 + exp(-x));
    }

    static double base_tanh(double x)
    {
        return tanh(x);
    }


};
#endif
