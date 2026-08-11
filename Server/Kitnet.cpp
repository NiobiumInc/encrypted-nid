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

#include "Kitnet.h"
#include "tobytes.h"
#include <math.h>
#include <stdexcept>
#include <vector>
#include <functional>
#include <chrono>

using namespace lbcrypto;

// C++ version of the KitNet Neural network model
// Model consists of an ensemble of autoencoders (3 layer perceptron), followed by
// an anomaly detector (3 layer perceptron) that operates on the residual from the
// autoencoder ensemble. Individual perceptrons are defined in the dA class. The model
// parameters also include a map that specifies the feature indices that are assigned
// to each autoencoder.
// This implementation reads the model parameters from a file when the object
// is created. It creates a list of autoencoders and a final anomaly detector using
// the dA class. It loads the transform and bias parameters into each created object.
// Only the execute method, which applies the model to an input vector, is
// implemented.
// FHE: The FHE server can use this class, along with the dA class to read the model 

KitNET::KitNET(std::ifstream &model_file)
{
    // read header from open stream
    int hdrlen = HDRVALS*VALSIZE;
    union bytestoheader bits;

    model_file.read(bits.raws, hdrlen);

    //load header values into class variables
    num_feat   = bits.modelpar.numfeat;
    num_ae     = bits.modelpar.numAE;
    ae_vis_dim = bits.modelpar.visAE;
    ae_hid_dim = bits.modelpar.hidAE;
    ad_vis_dim = bits.modelpar.visAD;
    ad_hid_dim = bits.modelpar.hidAD;
    apx_ord    = bits.modelpar.apxord;

    // read and save Chebyshev coefficients for logistic approximations
    union bytestodouble coeffs;
    for (int ii=0; ii<=apx_ord; ++ii)
    {
        model_file.read(coeffs.raws, 8);
        if (ii==0)
            sig_coeffs.push_back(2.0*coeffs.val);
        else
            sig_coeffs.push_back(coeffs.val);
    }
    for (int ii=0; ii<=apx_ord; ++ii)
    {
        model_file.read(coeffs.raws, 8);
        tanh_coeffs.push_back(coeffs.val);
    }
    KitNET::compute_chebyshev();

    // read and store feature maps 
    union bytestoshort mapidxs;
    std::vector<int> onemap;
    for (int ii=0; ii<num_ae; ++ii)
    {
        for (int jj=0; jj<ae_vis_dim; ++jj)
        {
            model_file.read(mapidxs.raw, 2);
            onemap.push_back(mapidxs.val);
        }
        map.push_back(onemap);
        onemap.clear();
    }
    
    // read in the rest of the file. Allocate and fill the model components.
    ensemble_layer.reserve(num_ae);
    union bytestodouble parm;
    double * what  = new double[ae_vis_dim * ae_hid_dim];
    double * hbhat = new double[ae_hid_dim];
    double * rbhat = new double [ae_vis_dim];

    for (int kk=0; kk<num_ae; ++kk)
    {   
        // read transform for this autoencoder
        for (int ii=0; ii<ae_vis_dim; ++ii)
        {
            for (int jj=0; jj<ae_hid_dim; ++jj)
            {
                model_file.read(parm.raws, 8);
                what[ii*ae_hid_dim + jj] = parm.val;
            }
        }
        // read hidden bias for this autoencoder
        for (int jj=0; jj<ae_hid_dim; ++jj)
        {
            model_file.read(parm.raws, 8);
            hbhat[jj] = parm.val;
        }      
        // read reconstruction bias for this autoencoder
        for (int ii=0; ii<ae_vis_dim; ++ii)
        {
            model_file.read(parm.raws, 8);
            rbhat[ii] = parm.val;
        }

        // create and fill next autoencoder object
        ensemble_layer.push_back(new dA(ae_vis_dim, ae_hid_dim, false));
        ensemble_layer[kk]->load_parameters(what, hbhat, rbhat);
        ensemble_layer[kk]->set_chebyshev_coeffs(fhesig_coeffs, as, bs);
    }
    delete [] what;
    delete [] hbhat;
    delete [] rbhat;

    double  * w  = new double[ad_vis_dim * ad_hid_dim];
    double * hb = new double [ad_hid_dim];
    double * rb = new double [ad_vis_dim];

    // read transform for the anomaly detector
    for (int ii=0; ii<ad_vis_dim; ++ii)
    {
        for (int jj=0; jj<ad_hid_dim; ++jj)
        {
            model_file.read(parm.raws, 8);
            w[ii*ad_hid_dim + jj] = parm.val;
        }
    }
    // read hidden bias for this autoencoder
    for (int jj=0; jj<ad_hid_dim; ++jj)
    {
        model_file.read(parm.raws, 8);
        hb[jj] = parm.val;
    }      
    // read reconstruction bias for this autoencoder
    for (int ii=0; ii<ad_vis_dim; ++ii)
    {
        model_file.read(parm.raws, 8);
        rb[ii] = parm.val;
    }
    output_layer = new dA(ad_vis_dim, ad_hid_dim, true);
    output_layer->load_parameters(w,hb,rb);
    output_layer->set_chebyshev_coeffs(fhetanh_coeffs, at, bt);

    // NEW: AD parameters for encrypted path
    ad_W.assign(w,  w  + (ad_vis_dim * ad_hid_dim));
    ad_hb.assign(hb, hb + ad_hid_dim);
    ad_rb.assign(rb, rb + ad_vis_dim);

    delete [] w;
    delete [] hb;
    delete [] rb;

    //allocate data for computations
    errvec = new double[num_feat];
    xi = new double[ae_vis_dim];


}

KitNET::~KitNET(void)
{
    delete [] xi;
    delete [] errvec;


}

double KitNET::execute(double * input)
{
    double * ei;
    double * result;

    for (int ii=0; ii<num_feat; ++ ii)
    {
        errvec[ii] = 0.0;
    }
    //loop thought the ensemble, calling execute for each autoencoder
    for (int kk=0; kk<num_ae; ++kk)
    {
        for (int ii=0; ii<ae_vis_dim; ++ii)
            xi[ii] = input[map[kk][ii]];

        ei = ensemble_layer[kk]->execute(xi);

        for (int ii=0; ii<ae_vis_dim; ++ii)
            errvec[ii+kk*ae_vis_dim] = ei[ii];
    }

    result = output_layer->execute(errvec);

    //compute and return the mean square error
    double mse = 0.0;
    for (int ii=0; ii<num_feat; ++ ii)
    {
        mse += result[ii]*result[ii];
    }
    mse = mse/(double) num_feat;
    return mse;
}

// FHE path

void KitNET::compute_chebyshev()
{
    //reading python generated Chebyshev coefficients is not compatible with openfhe
    //regenerate the coefficients using fhe EvalChebyshevCoefficients function
    std::function<double(double)> apxsig = KitNET::base_sigmoid;
    fhesig_coeffs = lbcrypto::EvalChebyshevCoefficients(apxsig, as, bs, 5);
    std::function<double(double)> apxtanh = KitNET::base_tanh;
    fhetanh_coeffs = lbcrypto::EvalChebyshevCoefficients(apxtanh, at, bt, 5);
}

// Linear combination without rotations
static Ciphertext<DCRTPoly> LinComb_NoRotate(
    const CryptoContext<DCRTPoly>& cc,
    const std::vector<Ciphertext<DCRTPoly>>& X,
    const std::vector<double>& w,
    double b) {

    if (X.empty() || w.empty() || X.size() != w.size())
        throw std::runtime_error("LinComb_NoRotate: size mismatch");

    Ciphertext<DCRTPoly> acc = cc->EvalMult(X[0], w[0]);
    for (size_t i = 1; i < w.size(); ++i)
        acc = cc->EvalAdd(acc, cc->EvalMult(X[i], w[i]));
    if (b != 0.0)
        acc = cc->EvalAdd(acc, b);
    return acc;
}

// Chebyshev nonlinearity
static inline Ciphertext<DCRTPoly> ActCheb(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& x,
    const std::vector<double>& coeffs,
    double a, double b) {

    return cc->EvalChebyshevSeries(x, coeffs, a, b);
}

bool KitNET::execute_ckks(
    const CryptoContext<DCRTPoly>& cc,
    const std::vector<Ciphertext<DCRTPoly>>& Ef,
    Ciphertext<DCRTPoly>& score_ct) const {

    if (!output_layer)
        return false;
    if ((int)Ef.size() != num_feat)
        return false;

    const std::vector<double>& sigC = fhesig_coeffs;
    const std::vector<double>& tanC = fhetanh_coeffs;

    // R_ct: residuals from AE ensemble, size = num_feat = num_ae * ae_vis_dim
    std::vector<Ciphertext<DCRTPoly>> R_ct(num_feat);

    int write_base = 0;
    for (int kk = 0; kk < num_ae; ++kk) {
        // gather feature-major inputs for this AE
        std::vector<Ciphertext<DCRTPoly>> X_ct(ae_vis_dim);
        for (int ii = 0; ii < ae_vis_dim; ++ii) {
            int fidx = map[kk][ii];
            X_ct[ii] = Ef[fidx];
        }

        // hidden layer
        std::vector<Ciphertext<DCRTPoly>> Y_ct(ae_hid_dim);
        for (int j = 0; j < ae_hid_dim; ++j) {
            std::vector<double> wcol(ae_vis_dim);
            for (int i = 0; i < ae_vis_dim; ++i)
                wcol[i] = ensemble_layer[kk]->peek_weight(i, j);
            double b_h = ensemble_layer[kk]->peek_hbias(j);

            auto lin = LinComb_NoRotate(cc, X_ct, wcol, b_h);
            Y_ct[j]  = ActCheb(cc, lin, sigC, as, bs);
            //Y_ct[j]  = lin;
        }

        // reconstruction layer
        std::vector<Ciphertext<DCRTPoly>> Z_ct(ae_vis_dim);
        for (int i = 0; i < ae_vis_dim; ++i) {
            std::vector<double> wrow(ae_hid_dim);
            for (int j = 0; j < ae_hid_dim; ++j)
                wrow[j] = ensemble_layer[kk]->peek_weight(i, j);
            double b_v = ensemble_layer[kk]->peek_vbias(i);

            auto lin = LinComb_NoRotate(cc, Y_ct, wrow, b_v);
            Z_ct[i]  = ActCheb(cc, lin, sigC, as, bs);
            //Z_ct[i]  = lin;
        }

        // residuals for this AE
        for (int i = 0; i < ae_vis_dim; ++i) {
            R_ct[write_base + i] = cc->EvalSub(X_ct[i], Z_ct[i]);
        }

        write_base += ae_vis_dim;
    }

    // Anomaly detector
    const int Vn = ad_vis_dim;
    const int Hn = ad_hid_dim;

    std::vector<Ciphertext<DCRTPoly>> YN_ct(Hn);
    for (int j = 0; j < Hn; ++j) {
        std::vector<double> wcolN(Vn);
        for (int i = 0; i < Vn; ++i)
            wcolN[i] = ad_W[i * ad_hid_dim + j];
        double b_h = ad_hb[j];

        auto lin = LinComb_NoRotate(cc, R_ct, wcolN, b_h);
        YN_ct[j]  = ActCheb(cc, lin, tanC, at, bt);
        //YN_ct[j]  = lin;
    }

    std::vector<Ciphertext<DCRTPoly>> ZN_ct(Vn);
    for (int i = 0; i < Vn; ++i) {
        std::vector<double> wrowN(Hn);
        for (int j = 0; j < Hn; ++j)
            wrowN[j] = ad_W[i * ad_hid_dim + j];
        double b_v = ad_rb[i];

        auto lin = LinComb_NoRotate(cc, YN_ct, wrowN, b_v);
        ZN_ct[i]  = ActCheb(cc, lin, tanC, at, bt);
        //ZN_ct[i]  = lin;
    }

    // AD residuals: E[i] = R[i] - ZN[i]
    // MSE = (1/Vn) * sum_i E[i]^2   (elementwise across slots / packets)
    auto diff0 = cc->EvalSub(R_ct[0], ZN_ct[0]);
    Ciphertext<DCRTPoly> sumsq = cc->EvalMult(diff0, diff0);
    for (int i = 1; i < Vn; ++i) {
        auto diff = cc->EvalSub(R_ct[i], ZN_ct[i]);
        sumsq = cc->EvalAdd(sumsq, cc->EvalMult(diff, diff));
    }
    auto mse = cc->EvalMult(sumsq, 1.0 / static_cast<double>(Vn));

    score_ct = mse;
    return true;
}


