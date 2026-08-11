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

import argparse
import os
import subprocess
import platform
from pathlib import Path


def get_wireshark_program_path(program: str) -> Path:
    """
    Finds the path to a specific Wireshark program.

    Input:
        program: str. A Wireshark program such as: 'tshark', 'mergecap', or 'editcap'.

    Output:
        str: System path to the Wireshark system excecutable.
    """

    system_path: str = os.environ["PATH"]
    path_separator: str = os.pathsep
    individual_paths: list[str] = system_path.split(path_separator)

    for path in individual_paths:
        if platform.system() == "Windows":
            exe_path = os.path.join(path, program + ".exe")
        else:
            exe_path = os.path.join(path, program)

        if os.path.isfile(exe_path):
            return Path(exe_path)

    raise OSError(
        f"Wireshark not found in PATH: {system_path}. Ensure the path to Wireshark in PATH."
    )


def merge_pcaps(
    mergecap_program_path: Path, pcap_folder_path: Path, out_path: Path
) -> Path:
    """
    merge_pcaps gathers PCAP files in pcap_folder_path and uses the Wireshark program `mergecap` located at: mergecap_program_path to merge all PCAP files.
    """

    # Find all PCAP files in data_path
    pcaps = pcap_folder_path / "*.pcap"

    # Build single string command.
    cmd: str = (
        '"' + str(mergecap_program_path) + '" -w  ' + str(out_path) + " " + str(pcaps)
    )

    subprocess.run(cmd, shell=True)

    return out_path


def pcap_to_tsv(
    tshark_program_path: Path, merged_pcap_path: Path, tsv_out_path: Path
) -> Path:

    # Gather all fields
    fields: list[str] = [
        "frame.time_epoch",
        "frame.len",
        "eth.src",
        "eth.dst",
        "ip.src",
        "ip.dst",
        "tcp.srcport",
        "tcp.dstport",
        "udp.srcport",
        "udp.dstport",
        "icmp.type",
        "icmpv6.type",
        "arp.opcode",
        "arp.src.hw_mac",
        "arp.src.proto_ipv4",
        "arp.dst.hw_mac",
        "arp.dst.proto_ipv4",
        "ipv6.src",
        "ipv6.dst",
        "tcp.flags",
        "tls.record.content_type",
        "tls.record.opaque_type",
        "tls.handshake",
        "frame.protocols",
    ]

    # Convert to string
    fields_str: str = "-e " + " -e ".join([a for a in fields])

    # Build single string command to pipe output.
    cmd: str = (
        '"'
        + str(tshark_program_path)
        + '" -r '
        + str(merged_pcap_path)
        + " -T fields "
        + fields_str
        + " -E header=y -E occurrence=f > "
        + str(tsv_out_path)
    )

    subprocess.run(cmd, shell=True)

    return tsv_out_path


def main(pcap_folder_path: Path, tsv_out_path: Path) -> Path:

    # Find mergecap and merge all PCAPs in data_path.
    mergecap_program_path: Path = get_wireshark_program_path("mergecap")

    # Merge all PCAPs.
    data_prefix: str = pcap_folder_path.stem
    merged_pcap_path: Path = pcap_folder_path.parent.joinpath(
        f"{data_prefix}_merged.pcap"
    )
    _ = merge_pcaps(mergecap_program_path, pcap_folder_path, merged_pcap_path)

    # Find tshark and parse merged PCAP into merged TSV.
    tshark_program_path: Path = get_wireshark_program_path("tshark")

    _ = pcap_to_tsv(tshark_program_path, merged_pcap_path, tsv_out_path)

    return tsv_out_path


if __name__ == "__main__":
    """
    This file contains code to read in a batch of PCAP files and concatenate them into one PCAP file, then output that PCAP file into a TSV file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_data", help="Path to folder containing PCAPs to merge.", type=Path
    )
    parser.add_argument(
        "tsv_out_path",
        help="Where to write a merged TSV file.",
        type=Path,
    )
    args = parser.parse_args()

    path_to_data = args.path_to_data
    tsv_out_path = args.tsv_out_path

    _ = main(path_to_data, tsv_out_path)
