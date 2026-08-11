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

#Import dependencies
from pathlib import Path
import Features.NetStat as ns
import numpy as np
from . import processpcap as pp
from . import ConnStat as cns

#Extracts statistical features from given pcap file one packet at a time using "get_next_vector()"
# Wireshark must be installed (tshark); it is used to parse the pcap file into a tsv file for ease of processing
# Once the tsv file is created, it can be used in subsequent runs to reduce processing time.
class featExt:
    def __init__(self,file_path:Path,limit=np.inf):
        self.path = file_path
        self.limit = limit
        self.parse_type = None #unknown

        # get and prep input pcap file. 
        self.inptsv = pp.pcapParse(file_path, limit)

        ### Prep Network Statistics extractor 
        maxHost = 100000000000
        maxSess = 100000000000

        self.Lambdas = [5,3,1,.1,.01]
        self.max_AE_size = 2*len(self.Lambdas)
        self.nstat = ns.netStat(self.Lambdas, maxHost, maxSess)
        self.conn = cns.connStat_db()
        self.connID: list[int] = []

 
    def get_next_vector(self):
        pkt = self.inptsv.get_next_vector()
        if (pkt.len == 0):
            #end of file reached. return empty feature vector
            self.inptsv.close()
            return [[],0,0]

        #process connection
        cnid = self.conn.update(pkt.srcaddr, pkt.dstaddr, pkt.proto, pkt.len, pkt.timestamp) 
        self.connID.append(cnid)  
         ### Extract Features
        try:
            return [self.nstat.updateGetStats(pkt.idx, pkt.content, pkt.srcaddr, pkt.dstaddr, pkt.len, pkt.timestamp), cnid, pkt.timestamp]
        except Exception as e:
            print(e)
            self.inptsv.close()
            return [[],0,0]


    def get_num_features(self):
        return self.nstat.getNetStatNumFeatures()
    
    def get_autoencoder_size(self):
        return self.max_AE_size
    
    def get_num_packets(self):
        return self.inptsv.get_num_packets()
    
    def get_connections(self):
        return self.connID
    
    def close_tsv(self):
        self.inptsv.close()
