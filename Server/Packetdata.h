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

// two dimensional data array 
// data is read from file
// Class is for in-the-clear batched feature data

#pragma once
#ifndef TWODATA   
#define TWODATA

#include <ios>
#include <streambuf>
#include <iostream>
#include <fstream>

class DataArray
{
public:
    DataArray(int numrow, int numcol);
    ~DataArray(void);

    void read_values(std::ifstream &file);
    double * get_vector(int colidx);

private:
    double * datafromfile;
    int num_feat;
    int num_pkt;

    double * pkt;
};

#endif