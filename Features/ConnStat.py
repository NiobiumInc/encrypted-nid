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

"""
Connection Database and Statistics

A PCAP file consists of one or more connections between a source address
and a destination address. this class tabulates the unique connections and
assigns a unique integer ID to each one

It also keeps track of the duration and average incoming and outgoing
packet lengths
"""

import sys

class connStat:
    def __init__(self, connection:int, time:float, src:str, dst:str, proto:str):
        self.ID = connection
        self.starttime = time
        self.lasttime = time
        self.srcaddr = src
        self.dstaddr = dst
        self.protocol = proto

        self.in_cnt:int = 0
        self.out_cnt:int = 0
        self.in_acc:float = 0.0
        self.out_acc:float = 0.0

    def update(self, v:int, t:float, direction: str='out'):
        self.lasttime = t
        
        if (direction == 'out'):
            self.out_cnt +=1
            self.out_acc += v
        elif (direction == 'in'):
            self.in_cnt += 1
            self.in_acc += v
        else:
            print("unrecognized connection direction " + direction, file=sys.stderr)

    def get_connID(self):
        return self.ID
    
    def get_protocol(self):
        return self.protocol
    
    def get_duration(self):
        return self.lasttime - self.starttime
    
    def get_length(self, direction:str='out'):
        if (direction == 'out'):
            if (self.out_cnt > 0):
                return self.out_acc/self.out_cnt
            else:
                return 0.0
        elif (direction == 'in'):
            if (self.in_cnt > 0):
                return self.in_acc/self.in_cnt
            else:
                return 0.0
        else:
            print("unrecognized connection direction " + direction, file=sys.stderr)
            return 0.0
       

class connStat_db:
    def __init__(self):
        self.HT = dict()
        self.last_connID = 0

    # registers a new connection or finds an existing connection
    def register(self, src:str, dst:str, t:float, proto: str):
        key = src+"_"+dst
        idxC = self.HT.get(key)
        if (idxC is None):
            oppkey = dst+"_"+src
            idxC = self.HT.get(oppkey)
            if (idxC is None):
                idxC = connStat(self.last_connID, t, src, dst, proto)
                self.HT[key] = idxC
                self.last_connID += 1
                direction = 'out'
            else:
                direction = 'in'
        else:
            direction = 'out'

        return [idxC, direction]
    
    def update(self, src:str, dst:str, proto:str, v:int, t:float):
        [idxC, inout] = self.register(src, dst, t, proto)

        idxC.update(v, t, inout)

        return idxC.get_connID()
 

    def get_Stats(self, src:str, dst:str):
        [idxC, _] = self.register(src, dst, 0.0, '')

        outmn = idxC.get_length('out')
        inmn = idxC.get_length('in')
        dur = idxC.get_duration()
        connID = idxC.get_connID()
        proto = idxC. get_protocol()

        return [connID, proto, dur, outmn, inmn]
    
    def get_num_conn(self):
        return self.last_connID


