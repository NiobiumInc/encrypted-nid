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

#include "Packetdata.h"
#include "tobytes.h"

//This class is used to read and manage in-the-clear, batched packet feature data.
// It includes a method to read the file and store it in a local array. The
// data is organized as [number features][number of packets] in the storage file.
// It also defines a method to extract the feature vector (column) from the storage
// array for a specified packet

DataArray::DataArray(int numrow, int numcol)
{
    num_feat = numrow;
    num_pkt = numcol;
    datafromfile = new double[numrow*numcol];
    pkt = new double[numrow];

};  

DataArray::~DataArray(void)
{
    delete [] pkt;
    delete [] datafromfile;
};

void DataArray::read_values(std::ifstream &datafile)
{
    union bytestodouble vals;

    //read data file into batch object
    for (int ii=0; ii<num_feat; ++ii)
    {
        for (int jj=0; jj<num_pkt; ++jj)
        {
            datafile.read(vals.raws, 8);
            datafromfile[ii*num_pkt+jj] = vals.val;
        }
    }

    //std::cerr << "sampled data" << datafromfile[0] << " " << datafromfile[20*num_pkt+1024] << " " << datafromfile[45*num_pkt+4096] << std::endl;
    return;
};

double * DataArray::get_vector(int pktidx)
{
    //extract data from the specified column (i.e., packet) into a vector
    for (int ii=0; ii<num_feat; ++ii)
    {
        pkt[ii] = datafromfile[ii*num_pkt+pktidx];
    }
    return pkt;
};