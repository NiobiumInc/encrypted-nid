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

# Script to pull out labels for Unit test examples
# this is a convienence script and is not part of the application.

import processlabel as pl
from pathlib import Path

wd = Path.cwd()
data_path = "Mirai"
base_path = wd.joinpath(Path("Datasets"), data_path)

label_name = data_path + "_labels" + ".csv"
label_path = base_path.joinpath(Path(label_name))


# Read label file and find the transition to malware
LF = pl.labelParse(label_path)
totalpkts = LF.read_labels()
firstmalpkt = LF.get_firstmal()

#write label file for unit test examples
#Records >= 71621 and <= 137156
out_name = "Unit_labels.csv"
out_path = base_path.joinpath(Path(out_name))
num_labels = LF.write_labels(out_path, 71621, 162157)
#num_labels = LF.write_labels(out_path, 0, -1)
print("wrote file " + str(out_path) + "with " + str(num_labels) + " records" )
