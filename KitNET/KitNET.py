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
#
# Original KitNET code Copyright (c) 2017 Yisroel Mirsky (MIT License)
# See end of file for full MIT license text.

import numpy as np
import KitNET.dA as AE
import sys
import struct
from pathlib import Path

# This class represents a KitNET machine learner.
# KitNET is a lightweight online anomaly detection algorithm based on an ensemble of autoencoders.

class KitNET:
    #num_feat: the number of features in input dataset (i.e., x \in R^n)
    #feature_map: map of features to autoencoders. The map must be a list,
    #           where the i-th entry contains a list of the feature indices to be assingned to the i-th autoencoder in the ensemble.
    #           For example, [[2,5,3],[4,0,1],[6,7]]
    #model_file: path to the binary file that holds a trained model parameters.
    #learning_rate: the default stochastic gradient descent learning rate for all autoencoders in the KitNET instance.
    #hidden_ratio: the default ratio of hidden to visible neurons. E.g., 0.75 will cause roughly a 25% compression in the hidden layer.
    def __init__(self,num_feat=0,feature_map=None, model_file=None,learning_rate=0.1,hidden_ratio=0.75):
        # Parameters:
        self.lr = learning_rate
        self.hr = hidden_ratio
        self.n = num_feat

        # Variables
        self.n_trained = 0 # the number of training instances so far
        self.n_executed = 0 # the number of executed instances so far
        self.ensembleLayer = []
        self.outputLayer = None
        self.logistic_apx = AE.dA_logistic()   #computes Chebyshev polynomial approximation coefficients
        self.apx_order = self.logistic_apx.get_order()

        #create fixed file packing formats for header information       
        self.hdrlen = 7*2
        self.hdrfmt = '@'+'7h'
       
         
        #kitNET must be initialized with either a feature map or a model file. Creation fails if neither one is available. 
        #The model is created from the file only if the feature map is None.
        if (feature_map is None) | (self.n == 0):
            #check to see if model file is present. If so, create the model from the file parameters. If not, abort.
            if (model_file is None):
                raise RuntimeError('Can not create KitNET model. Specify either a feature map or a model file')
            elif (not model_file.exists()):  # model parameter file does not exist
                raise RuntimeError('File: '+ str(model_file) + ' does not exist. Run fhe-NetworkMonitor with the -train flag to generate.')
            else:
                self.map = []
                self.load_model(model_file)
        else:
            self.map = feature_map
            self.__createAD__()
 
    #train KitNET on x
    def train(self,x):
        ## Ensemble Layer
        residual = np.zeros(self.n)
        sidx = 0
        for a in range(len(self.ensembleLayer)):
            # make sub instance for autoencoder 'a'
            xi = x[self.map[a]]
            residual[sidx:sidx+self.AEvis] = self.ensembleLayer[a].train(xi)
            sidx += self.AEvis
        ## OutputLayer
        rawerr = self.outputLayer.train(residual)

        self.n_trained += 1
        return np.sqrt(np.mean(rawerr**2))

    #execute KitNET on x
    def execute(self,x):
        if self.map is None:
            raise RuntimeError('KitNET Cannot execute x, because a feature mapping has not yet been learned or provided')
        else:
            self.n_executed += 1
            ## Ensemble Layer
            residual = np.zeros(self.n)
            sidx = 0
            for a in range(len(self.ensembleLayer)):
                # make sub inst
                xi = x[self.map[a]]
                residual[sidx:sidx+self.AEvis] = self.ensembleLayer[a].execute(xi)
                sidx += self.AEvis
            ## OutputLayer
            rawerr = self.outputLayer.execute(residual)
            return np.sqrt(np.mean(rawerr**2))

    def __createAD__(self):
        # construct ensemble layer
        for map in self.map:
            params = AE.dA_params(n_visible=len(map), n_hidden=0, lr=self.lr, hiddenRatio=self.hr, chebyapx=self.logistic_apx.get_sigmoid_apx())
            self.ensembleLayer.append(AE.dA(params))
        self.AEvis = params.n_visible
        self.AEhid = params.n_hidden

        # construct output layer
        params = AE.dA_params(self.n, n_hidden=len(self.map), lr=self.lr, chebyapx=self.logistic_apx.get_tanh_apx(), nonlin='tanh')
        self.outputLayer = AE.dA(params)
        self.ADvis = params.n_visible
        self.ADhid = params.n_hidden

    def print_AE_limits(self):
        for a in range(len(self.ensembleLayer)):
            aelimits = self.ensembleLayer[a].get_limits()
            print(f"AE {a} : {aelimits}", file=sys.stderr)
        
        aelimits = self.outputLayer.get_limits()
        print(f"Anom Det : {aelimits}", file=sys.stderr)

    def print_AE_parameters(self):
        for a in range(len(self.ensembleLayer)):
            [W, hbias, rbias] = self.ensembleLayer[a].get_parameters()
            print(f" AE {a}\n")
            print(f"Weights: {W} \n")
            print(f"hidden bias: {hbias} \n")
            print(f"reconstruct bias {rbias} \n")

        [W, hbias, rbias] = self.outputLayer.get_parameters()
        print(f" Anomaly Detector\n", file=sys.stderr)
        print(f"Weights: {W} \n", file=sys.stderr)
        print(f"hidden bias: {hbias} \n", file=sys.stderr)
        print(f"reconstruct bias {rbias} \n", file=sys.stderr)

    def save_model(self, filename:Path):
        #model parameters use short int (2 bytes) for all integer values and double (8 bytes) for all floating point values
        numAE = len(self.map)
        sigcoeff = self.logistic_apx.get_sigmoid_apx()
        tanhcoeff = self.logistic_apx.get_tanh_apx()

        #Create the packing formats for the model file. 
        numcoeff = self.apx_order + 1
        apxfmt = '@'+str(numcoeff)+'d'
        mapfmt = '@'+str(self.AEvis)+'h'
        aehidfmt = '@'+str(self.AEhid)+'d'
        aevisfmt = '@'+str(self.AEvis)+'d'
        adhidfmt = '@'+str(self.ADhid)+'d'
        advisfmt = '@'+str(self.ADvis)+'d'

        with open(filename, 'wb') as fp:
            #write header and logistic approximation coefficents
            fp.write(struct.pack(self.hdrfmt, numAE, self.n, self.AEvis, self.AEhid, self.ADvis, self.ADhid, self.apx_order))
            fp.write(struct.pack(apxfmt, *sigcoeff))
            fp.write(struct.pack(apxfmt, *tanhcoeff))

            #write feature to autoencoder maps
            for v in self.map:
                fp.write(struct.pack(mapfmt, *v))

            #write AE transform and biases
            for a in range(len(self.ensembleLayer)):
                [W, hbias, rbias] = self.ensembleLayer[a].get_parameters()
                for ii in range(self.AEvis):
                    fp.write(struct.pack(aehidfmt, *W[ii]))
                fp.write(struct.pack(aehidfmt, *hbias))
                fp.write(struct.pack(aevisfmt, *rbias))

            # write AD transform and biases
            [W, hbias, rbias] = self.outputLayer.get_parameters()
            for ii in range(self.ADvis):
                    fp.write(struct.pack(adhidfmt, *W[ii]))
            fp.write(struct.pack(adhidfmt, *hbias))
            fp.write(struct.pack(advisfmt, *rbias))


    def load_model(self, filename:Path):
 
        #This function creates a new NN model with the parameters from the model file.
        with open(filename, 'rb') as fp:
            #get header
            bits = fp.read(self.hdrlen)
            [K, self.n, dae, hae, dad, had, ord] = struct.unpack(self.hdrfmt, bits)
            #if (len(self.map) != K) | (self.AEvis != dae) | (self.AEhid != hae) | (self.ADvis !=dad) | (self.ADhid != had):
            #   print('Model does not match expected architecture. Aborting load')
            #    return -1
            print(f'header: AE: {K} num feat {self.n} vis dim {dae} hid dim {hae} AD: vis dim {dad} hid dim {had}', file=sys.stderr)

            #create read formats from the header information
            apxlen = (ord+1)*8
            numcoeff = ord+1
            apxfmt = '@'+str(numcoeff)+'d'
            maplen = dae*2
            mapfmt = '@'+str(dae)+'h'
            aehidlen = hae*8
            aehidfmt = '@'+str(hae)+'d'
            aevislen = dae*8
            aevisfmt = '@'+str(dae)+'d'
            adhidlen = had*8
            adhidfmt = '@'+str(had)+'d'
            advislen = dad*8
            advisfmt = '@'+str(dad)+'d'

            #read logistic approximation coefficients. As long as we are using python, we recalcuate the approximation coefficients
            #so we can read and discard these. However, a C++ server should read and load these parameters. 
            bits = fp.read(apxlen)
            *scoeff, = struct.unpack(apxfmt, bits)
            #print(f'sigmoid parm: order {so} coeff {scoeff}')
            bits = fp.read(apxlen)
            *tcoeff, = struct.unpack(apxfmt, bits)
            #print(f'tanh parm: order {to} coeff {tcoeff}')

            #read and load the feature maps
            for ii in range(K):
                bits = fp.read(maplen)
                *mapfeats, = struct.unpack(mapfmt, bits)
                self.map.append(mapfeats)

            self.__createAD__()     #create empty NN model now that we have read the feature map

            #read the autoencoder parameters: transform, hidden layer bias, and reconstruction layer bias
            for a in range(len(self.ensembleLayer)):
                What = np.zeros((self.AEvis, self.AEhid))
                for ii in range(self.AEvis):
                    bits = fp.read(aehidlen)
                    *ww, = struct.unpack(aehidfmt, bits)
                    What[ii] = ww

                bits = fp.read(aehidlen)
                *hbiashat, = struct.unpack(aehidfmt, bits)
                bits = fp.read(aevislen)
                *rbiashat, = struct.unpack(aevisfmt, bits)
                self.ensembleLayer[a].load_parameters(What, hbiashat, rbiashat)


            #read the anomaly parameters: transform, hidden layer bias, and reconstruciton layer bias
            What = np.zeros((self.ADvis, self.ADhid))
            for ii in range(self.ADvis):
                bits = fp.read(adhidlen)
                *ww, = struct.unpack(adhidfmt, bits)
                What[ii] = ww

            bits = fp.read(adhidlen)
            *hbiashat, = struct.unpack(adhidfmt, bits)
            bits = fp.read(advislen)
            *rbiashat, = struct.unpack(advisfmt, bits)
            self.outputLayer.load_parameters(What, hbiashat, rbiashat)

    def get_num_features(self):
        return self.n

#  Copyright (c) 2017 Yisroel Mirsky
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.