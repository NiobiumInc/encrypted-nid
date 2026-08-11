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

import sys
import numpy as np
import struct

# Class to train the normalization statistics for the features that will be input to KitNET
class normStat:
    #storage packing
    packpar = struct.Struct("@dd")    #packer for one feature: mean scale
    packhdr = struct.Struct("@i")     #packer for header: number of features

    def __init__(self,n, norm_file=None):

        #parameter:
        self.n = n     #number of features
        
        #varaibles
        self.norm_cnt = np.zeros(self.n)
        self.norm_avg = np.zeros(self.n)
        self.norm_res = np.zeros(self.n)
        self.norm_mean = np.zeros(self.n)
        self.norm_scale = np.zeros(self.n)

        if (not norm_file is None):
            self.load_parm(norm_file)

    # x: a numpy vector of length n
    def update(self,x):
        #statistics for logistic norm
        self.norm_cnt += np.ones(self.n)
        self.norm_avg += x
        self.norm_mean = self.norm_avg / self.norm_cnt
        resid = x - self.norm_mean
        self.norm_res += resid * resid
        self.norm_scale = 4.0*np.sqrt(self.norm_res/self.norm_cnt)


    def get_logisticnorm(self,x):
        xhat = (x-self.norm_mean)/self.norm_scale
        if (np.exp(-xhat).any() == -1):
            print('Error: failed normalization sigmoid', file=sys.stderr)
        return 1. / (1 + np.exp(-xhat))
    
    def save_parm(self, filename):
        with (open(filename, 'bw') as fp):
            fp.write(self.packhdr.pack(self.n))
            for ii in range(self.n):
                fp.write(self.packpar.pack(self.norm_mean[ii], self.norm_scale[ii]))
    
    def load_parm(self, filename):
        with (open(filename, 'br') as fp):
            size = self.packhdr.size
            numfeat, = self.packhdr.unpack(fp.read(size))
            #check that number of features matches the expected number
            if (numfeat != self.n):
                print(f'number of feature mismatch read{numfeat} should be {self.n}', file=sys.stderr)
                return -1
            else:
                self.n = numfeat
            
            size = self.packpar.size
            for ii in range(numfeat):
                mean, scale = self.packpar.unpack(fp.read(size))
                self.norm_mean[ii] = mean
                self.norm_scale[ii] = scale
            return numfeat
    
