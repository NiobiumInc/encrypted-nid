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

import math
import numpy as np


class incStat:
    def __init__(self, Lambda, ID, init_time=0):  # timestamp is creation time
        self.ID = ID
        self.CF1 = 0.0  # linear sum
        self.res = 0.0
        self.CF2 = 0.0  # sum of squares
        self.w = 1e-20  # weight
        self.Lambda = Lambda  # Decay Factor
        self.lastTimestamp = init_time
        self.cur_mean = np.nan
        self.cur_var = np.nan
        self.cur_std = np.nan
        self.covs = [] # a list of incStat_covs (references) with relate to this incStat

    def update(self, v, t=0):  # v is a scalar, t is v's arrival timestamp
        self.processDecay(t)

        # update with v
        self.w += 1
        self.CF1 += v
        self.cur_mean = np.nan
        self.res = v - self.mean()
        self.CF2 += self.res * self.res
        
        #force recalculations
        self.cur_var = np.nan
        self.cur_std = np.nan

    def processDecay(self, timestamp):
        factor=1
        # check for decay
        timeDiff = timestamp - self.lastTimestamp
        if timeDiff > 0:
            factor = math.pow(2, (-self.Lambda * timeDiff))
            self.CF1 = self.CF1 * factor
            self.CF2 = self.CF2 * factor
            self.w = self.w * factor
            self.lastTimestamp = timestamp
        return factor

    def weight(self):
        return self.w

    def mean(self):
        if math.isnan(self.cur_mean):    # recalculate only when new data
            if (self.w == 0):
                print('Error: count of zero')
                self.cur_mean = 0.0
            else:
                self.cur_mean = self.CF1 / self.w
        return self.cur_mean

    def var(self):
        if math.isnan(self.cur_var):  # calculate it only once when necessary
            if (self.w == 0):
                self.cur_var = 0.0
            else:
                self.cur_var = self.CF2 / self.w
                if (self.cur_var < 0):
                    self.cur_var = 0.0
        return self.cur_var

    def std(self):
        if math.isnan(self.cur_std):  # calculate it only once when necessary
            self.cur_std = math.sqrt(self.var())
        return self.cur_std
    
    def pkts(self, other_incStats): # total number of packets in correlation stats
        P = self.weight()
        for incS in other_incStats:
            P += incS.weight()
        return P

    def radius(self, other_incStats):  # the radius of a set of incStats (variance of variable sum)
        A = self.var()
        for incS in other_incStats:
            A += incS.var()
        return A

    def magnitude(self, other_incStats):  # the magnitude of a set of incStats
        A = math.pow(self.mean(), 2)
        for incS in other_incStats:
            A += math.pow(incS.mean(), 2)
        return math.sqrt(A)

    #calculates and pulls all stats on this stream
    def allstats_1D(self):
        #self.cur_mean = self.CF1 / self.w
        #self.cur_var = abs(self.CF2 / self.w - math.pow(self.cur_mean, 2))
        return [self.w, self.mean(), self.var()]


#like incStat, but maintains stats between two streams
class incStat_cov:
    def __init__(self, incS1, incS2, init_time = 0):
        # store references to the streams' incStats
        self.incStats = [incS1,incS2]
        self.lastRes = [0,0]
        # init extrapolators
        #self.EXs = [extrapolator(),extrapolator()]

        # init sum product residuals
        self.CF3 = 0 # sum of residule products (A-uA)(B-uB)
        self.w3 = 1e-20
        self.lastTimestamp_cf3 = init_time

    # ID: the stream ID which produced (v,t)
    # assumes incStat "ID" has ALREADY been updated with (t,v) [performed in method incStat.update()]
    def update_cov(self, ID, v, t):  
        # find incStat
        if ID == self.incStats[0].ID:
            inc = 0
        elif ID == self.incStats[1].ID:
            inc = 1
        else:
            print("update_cov ID error")
            return ## error

        # Decay other incStat
        #self.incStats[not(inc)].processDecay(t)

        # Decay residules
        self.processDecay(t,inc)

        # Compute and update residule
        res = (v - self.incStats[inc].mean())
        resid = (v - self.incStats[inc].mean()) * self.lastRes[not(inc)]
        self.CF3 += resid
        self.w3 += 1
        self.lastRes[inc] = res

    def processDecay(self,t,micro_inc_indx):
        factor = 1
        # check for decay cf3
        timeDiffs_cf3 = t - self.lastTimestamp_cf3
        if timeDiffs_cf3 > 0:
            factor = math.pow(2, (-(self.incStats[micro_inc_indx].Lambda) * timeDiffs_cf3))
            self.CF3 *= factor
            self.w3 *= factor
            self.lastTimestamp_cf3 = t
            self.lastRes[micro_inc_indx] *= factor
        return factor

    #todo: add W3 for cf3

    #covariance approximation
    def cov(self):
        if (self.w3 == 0):
            return 0.0
        else:
            return self.CF3 / self.w3

    # Pearson corl. coef
    def pcc(self):
        ss = self.incStats[0].std() * self.incStats[1].std()
        if ss != 0:
            pccval = self.cov() / ss
            if (pccval > 1.0):
                return 0.0
            elif (pccval < -1.0):
                return 0.0
            else:
                return pccval
        else:
            return 0.0

    # calculates and pulls just correlative stats
    def get_stats1(self):
        return [self.cov(), self.pcc()]

    # calculates and pulls all correlative stats AND 2D stats from both streams (incStat)
    def allstats_2D(self):
        #return [self.incStats[0].pkts([self.incStats[1]]),self.incStats[0].radius([self.incStats[1]]),self.incStats[0].magnitude([self.incStats[1]]),self.cov(), self.pcc()]
        return [self.incStats[0].radius([self.incStats[1]]),self.incStats[0].magnitude([self.incStats[1]]),self.cov(), self.pcc()]

class incStat_db:
    # default_lambda: use this as the lambda for all streams. If not specified, then you must supply a Lambda with every query.
    def __init__(self, feat_type='pktlen', limit=np.inf,default_lambda=np.nan):
        self.HT = dict()
        self.limit = limit
        self.df_lambda = default_lambda
        if (feat_type == 'timdif'):
            self.timdif = True
        elif (feat_type == 'pktlen'):
            self.timdif = False
        else:
            self.timdif = False

    def get_lambda(self,Lambda):
        if not np.isnan(self.df_lambda):
            Lambda = self.df_lambda
        return Lambda

    # Registers a new stream. init_time: init lastTimestamp of the incStat
    def register(self,ID,Lambda=1,init_time=0):
        #Default Lambda?
        Lambda = self.get_lambda(Lambda)

        #Retrieve incStat
        key = ID+"_"+str(Lambda)
        incS = self.HT.get(key)
        if incS is None: #does not already exist
            if len(self.HT) + 1 > self.limit:
                raise LookupError(
                    'Adding Entry:\n' + key + '\nwould exceed incStatHT 1D limit of ' + str(
                        self.limit) + '.\nObservation Rejected.')
            incS = incStat(Lambda, ID, init_time)
            self.HT[key] = incS #add new entry
        return incS

    # Registers covariance tracking for two streams, registers missing streams
    def register_cov(self,ID1,ID2,Lambda=1,init_time=0):
        #Default Lambda?
        Lambda = self.get_lambda(Lambda)

        # Lookup both streams
        incS1 = self.register(ID1,Lambda,init_time)
        incS2 = self.register(ID2,Lambda,init_time)

        #check for pre-exiting link
        for cov in incS1.covs:
            if cov.incStats[0].ID == ID2 or cov.incStats[1].ID == ID2:
                return [incS1, incS2, cov]  #there is a pre-exiting link

        # Does not exist. Instatiate a new inc_cov object and link to both incStats
        incCov = incStat_cov(incS1,incS2,init_time)
        incS1.covs.append(incCov)
        incS2.covs.append(incCov)
        return [incS1, incS2, incCov]

    # Updates and then pulls current 1D stats from the given ID. Automatically registers previously unknown stream IDs
    def update_get_1D_Stats(self, ID,t,vin,Lambda=1):  # weight, mean, std
        # get incremental stats object
        incS = self.register(ID,Lambda,t)

        #calculate value based on type
        if self.timdif:
            dif = t - incS.lastTimestamp
            if dif > 0:
                v = dif
            else:
                v = 0
        else:
            v = vin   

        #update statistics accumulators
        incS.update(v,t)

        return incS.allstats_1D()

    # Updates and then pulls current 1D and 2D stats from the given IDs. Automatically registers previously unknown stream IDs
    def update_get_1D2D_Stats(self, ID1,ID2,t,vin,Lambda=1): 
        # register the IDs and find the individual src and dst objects, and the cov (dual) object
        [incS, incD, incCov] = self.register_cov(ID1, ID2, Lambda,  t)

        #calculate value based on type
        if self.timdif:
            dif = t - incS.lastTimestamp
            if dif > 0:
                v = dif
            else:
                v = 0
        else:
            v = vin

        #update and get indivdual statistics for the source
        incS.update(v, t)
        src_feat = incS.allstats_1D()

        #decay statistics for the destination prior to covariance calculation.
        #get current values (input) stats to add to feature vector
        incD.processDecay(t)
        #dst_feat = incD.allstats_1D()

        #Update the covariance statistics for this source + destination pair
        incCov.update_cov(ID1,v,t)
        dual_feat = incCov.allstats_2D()

        #return np.concatenate((src_feat, dst_feat, dual_feat))
        return np.concatenate((src_feat, dual_feat))
