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
# Portions derived from Kitsune (https://github.com/ymirsky/kitsune-py),
# Copyright (c) 2018 Yisroel Mirsky, MIT License. See NOTICE for full text.

import numpy as np

import Features.IncrStat as ics

#
# MIT License
#
# Copyright (c) 2018 Yisroel mirsky
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class netStat:
    #Datastructure for efficent network stat queries
    # HostLimit: no more that this many Host identifiers will be tracked
    # HostSimplexLimit: no more that this many outgoing channels from each host will be tracked (purged periodically)
    # Lambdas: a list of 'window sizes' (decay factors) to track for each stream. nan resolved to default [5,3,1,.1,.01]
    def __init__(self, Lambdas = None, HostLimit=255,HostSimplexLimit=1000):
        #Lambdas
        if Lambdas is None:
            self.Lambdas = [5,3,1,.1,.01]
        else:
            self.Lambdas = Lambdas
        self.numLambdas = len(self.Lambdas)

        #HT Limits
        self.HostLimit = HostLimit
        self.SessionLimit = HostSimplexLimit*self.HostLimit*self.HostLimit #*2 since each dual creates 2 entries in memory
        self.MAC_HostLimit = self.HostLimit*10

        self.NUM_1D_FEAT = 3
        self.NUM_2D_FEAT = 4
        #self.NUM_1D2D_FEAT =  2*self.NUM_1D_FEAT + self.NUM_2D_FEAT
        self.NUM_1D2D_FEAT =  self.NUM_1D_FEAT + self.NUM_2D_FEAT
        self.numFeatures = 0

        #Dictionaries of network statistics. Number of features must match those returned in updateGetStats below.
        self.HT_jit = ics.incStat_db(feat_type='timdif', limit=self.HostLimit*self.HostLimit)#H-H Jitter Stats. 1D 
        self.numFeatures += self.NUM_1D_FEAT * self.numLambdas
        #self.HT_MI = ics.incStat_db(limit=self.MAC_HostLimit)#MAC-IP relationships 1D
        #self.numFeatures += self.NUM_1D_FEAT * self.numLambdas
        self.HT_H = ics.incStat_db(feat_type = 'pktlen', limit=self.HostLimit) #Source Host BW Stats 1D2D
        self.numFeatures += self.NUM_1D2D_FEAT * self.numLambdas
        #self.HT_Hp = ics.incStat_db(feat_type = 'pktlen', limit=self.SessionLimit)#Source Host BW Stats 1D2D
        #self.numFeatures += self.NUM_1D2D_FEAT * self.numLambdas


    def updateGetStats(self, pktidx, content, srcIP, dstIP, datagramSize, timestamp):
        # Host BW: Stats on the srcIP's general Sender Statistics
        # Hstat = np.zeros((3*len(self.Lambdas,)))
        # for i in range(len(self.Lambdas)):
        #     Hstat[(i*3):((i+1)*3)] = self.HT_H.update_get_1D_Stats(srcIP, timestamp, datagramSize, self.Lambdas[i])

        #MAC.IP: Stats on src MAC-IP relationships
       # MIstat =  np.zeros((3*len(self.Lambdas,)))
       # for i in range(len(self.Lambdas)):
       #     MIstat[(i*3):((i+1)*3)] = self.HT_MI.update_get_1D_Stats(srcMAC+srcIP, timestamp, datagramSize, self.Lambdas[i])

        # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
        HHstat =  np.zeros((self.NUM_1D2D_FEAT*len(self.Lambdas,)))
        for ii in range(len(self.Lambdas)):
            HHstat[(ii*self.NUM_1D2D_FEAT):((ii+1)*self.NUM_1D2D_FEAT)] = self.HT_H.update_get_1D2D_Stats(srcIP, dstIP,timestamp,datagramSize,self.Lambdas[ii])

        # Host-Host Jitter:
        HHstat_jit =  np.zeros((self.NUM_1D_FEAT*len(self.Lambdas,)))
        for ii in range(len(self.Lambdas)):
            HHstat_jit[(ii*self.NUM_1D_FEAT):((ii+1)*self.NUM_1D_FEAT)] = self.HT_jit.update_get_1D_Stats(srcIP, timestamp, 0.0, self.Lambdas[ii])

        # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
       # HpHpstat =  np.zeros((self.NUM_1D2D_FEAT*len(self.Lambdas,)))
       # if srcProtocol == 'arp':
       #     for i in range(len(self.Lambdas)):
       #         HpHpstat[(i*self.NUM_1D2D_FEAT):((i+1)*self.NUM_1D2D_FEAT)] = self.HT_Hp.update_get_1D2D_Stats(srcMAC, dstMAC, timestamp, datagramSize, self.Lambdas[i])
       # else:  # some other protocol (e.g. TCP/UDP)
       #     for i in range(len(self.Lambdas)):
       #         HpHpstat[(i*self.NUM_1D2D_FEAT):((i+1)*self.NUM_1D2D_FEAT)] = self.HT_Hp.update_get_1D2D_Stats(srcIP + srcProtocol, dstIP + dstProtocol, timestamp, datagramSize, self.Lambdas[i])

        return np.concatenate((HHstat, HHstat_jit))  # concatenation of stats into one stat vector

    def getNetStatNumFeatures(self):
        return self.numFeatures
