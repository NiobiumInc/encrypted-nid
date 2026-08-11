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

## process Kitsune label file
## csv files with packet index and a normal/malicious label

import csv
import sys
from pathlib import Path

class labelParse:
    def __init__(self, file_path:Path):
        self.path = file_path
        self.labels = []
        self.totalpkts = 0
        self.firstmal = 0
        self.curidx = 0

        self.__prep__()

    def __prep__(self):
        # Find file: 
        if not self.path.exists():  # file does not exist
            raise RuntimeError("File: {self.path} does not exist.")
        
        ### open reader ##
        maxInt: int = sys.maxsize
        decrement: bool = True
        while decrement:
            # decrease the maxInt value by factor 10
            # as long as the OverflowError occurs.
            try:
                csv.field_size_limit(maxInt)
                decrement = False
            except OverflowError:
                maxInt = int(maxInt / 10)

        self.csvlabf = open(self.path, "rt", encoding="utf8")
        self.csvlab= csv.reader(self.csvlabf, delimiter=",")
        self.get_next_pkt()   #read and discard the header

    def get_next_pkt(self):
        try:
            row = self.csvlab.__next__()
            self.curidx += 1
            return  row[-1]
        except StopIteration:
            self.csvlabf.close()
            return -1
        
    def read_labels(self):
        # read in labels
        while True:
            v = int(self.get_next_pkt())
            if (v == -1):
                #end of file
                self.totalpkts = self.labels.__len__()
                print(f"There are {self.totalpkts} packets.", file=sys.stderr)
                break
            else:
                self.labels.append(v)
        
        #find first malicious packet
        self.firstmal = self.labels.index(1) 
        return self.totalpkts

    def get_firstmal(self):
        return self.firstmal 
    
    def get_label(self, index):
        return self.labels[index]

    def write_labels(self, out_path, first_idx, last_idx):
        if (last_idx == -1):
            last_idx = self.totalpkts

        with open(out_path, "wt", newline='', encoding="utf8") as outlabf:
            outlab= csv.writer(outlabf, delimiter=",")

            header = ["", "x"]
            outlab.writerow(header)
            for ii in range(first_idx, last_idx):
                thislabel = [ii-first_idx+1, self.labels[ii]]
                outlab.writerow(thislabel)

        return last_idx - first_idx

