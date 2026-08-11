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

from typing import Optional
import csv
import sys
from pathlib import Path

import numpy as np

from . import mergepcaps


class packetData:
    def __init__(self):
        self.idx: int = 0  # packet index in the current file. Used for book-keeping
        self.proto: str = (
            ""  # protocol used in the current packet. Used for packet understanding.
        )
        self.timestamp: float = (
            0.0  # time of the current packet. Used to find jitter, i.e., delta time, for statistics calculation
        )
        self.len: int = (
            0  # number of bytes in the current packet. Used for statistics calculations
        )
        self.flags: int = (
            0  # tcp flags field from the current packet. Used to recognize SYN and ACK packets
        )
        self.content: str = (
            ""  # tls content field from the current packet. Used for packet understanding
        )
        self.srcaddr: str = (
            ""  # source IP addr string. Used to separate statistics calculations by connection
        )
        self.dstaddr: str = (
            ""  # destination IP addr string. Used to separate statistics calculations by connection
        )

    def clear(self):
        self.idx = 0
        self.proto = ""
        self.timestamp = 0.0
        self.len = 0
        self.flags = 0
        self.content = ""
        self.srcaddr = ""
        self.dstaddr = ""


class pcapParse:
    def __init__(self, file_path: Path, limit=np.inf):
        self.path = file_path
        self.limit = limit
        self.parse_type = ""
        self.curPacketIndx = 0
        self.numConn = 0
        self.curConn = 0
        self.pkt = packetData()

        ### Prep pcap  creates self.tsvin and self.tsvinf ##
        self.__prep__()

    def __prep__(self):
        ### Find file: ###
        if not self.path.exists():  # file does not exist
            raise RuntimeError("File: "+str(self.path)+" does not exist.")

        ### check file type ###
        filetype = self.path.suffix

        ##If file is TSV (pre-parsed by wireshark script)
        if filetype == ".tsv":
            self.parse_type = "tsv"

        ##If file is pcap
        elif filetype == ".pcap" or filetype == ".pcapng":
            tsv_out_path: Path = self.path.with_suffix(".tsv")
            if not tsv_out_path.exists():
                tshark_path = mergepcaps.get_wireshark_program_path("tshark")
                _ = mergepcaps.pcap_to_tsv(
                    tshark_path, self.path, tsv_out_path
                )  # creates local tsv file
            self.path = tsv_out_path
            self.parse_type = "tsv"

        else:
            raise RuntimeError("File: "+str(self.path)+" is not a tsv or pcap file.")

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

        self.num_lines = sum(1 for _ in open(self.path))
        #print(f"There are {self.num_lines-1} packets.", file=sys.stderr)
        self.limit = min(self.limit, self.num_lines)
        self.tsvinf = open(self.path, "rt", encoding="utf8")
        self.tsvin = csv.reader(self.tsvinf, delimiter="\t")

    def get_next_packet(self) -> list[Optional[str]]:
        if self.curPacketIndx == self.limit:
            if self.parse_type == "tsv":
                self.tsvinf.close()
            return []

        ### Parse next packet ###
        if self.parse_type == "tsv":
            try:
                row: list = self.tsvin.__next__()
                self.curPacketIndx += 1
                return row
            except StopIteration:
                self.tsvinf.close()
                return []
        else:
            return []

    def interpret_packet(self, row) -> packetData:
        self.pkt.idx = self.curPacketIndx
        # get protocol information
        protostring = row[23]
        if "tcp" in protostring:
            if "tls" in protostring:
                self.pkt.proto = "tls"
            else:
                self.pkt.proto = "tcp"
        elif "udp" in protostring:
            self.pkt.proto = "udp"
        elif "arp" in protostring:
            self.pkt.proto = "arp"
        elif "icmp" in protostring:
            self.pkt.proto = "icmp"
        elif "igmp" in protostring:
            self.pkt.proto = "gmp"
        else:
            self.pkt.proto = ""
        if row[19] != "":
            self.pkt.flags = int(row[19], 16)  # tcp flags. 0x010 ACK, 0x002 SYN, 0x008 PSH
        else:
            self.pkt.flags = 0

        if row[20] != "":
            self.pkt.content = row[20            ]  # tls.record.content_type 20 or 22 for tls connection
        elif row[21] != "":
            self.pkt.content = row[21]  # tls.record.opaque_type 23 for application data
        else:
            syn = self.pkt.flags & 0x002
            ack = self.pkt.flags & 0x010
            psh = self.pkt.flags & 0x008
            if syn and not (ack):
                self.pkt.content = "02"  # initial tcp synchronization packet
            elif syn and ack:
                self.pkt.content = "12"  # return tcp synchronization packet
            elif psh and ack:
                self.pkt.content = "23"  # application data, but without tls information
            elif ack:
                self.pkt.content = "10"  # any other tcp ACK packet
            else:
                self.pkt.content = ""

        # get time and packet length
        self.pkt.timestamp = float(row[0])
        self.pkt.len = int(row[1])

        # get source and destination address info
        self.pkt.srcaddr = ""
        self.pkt.dstaddr = ""
        if row[4] != "":  # IPv4
            srcIP = row[4]
            dstIP = row[5]
        elif row[17] != "":  # ipv6
            srcIP = row[17]
            dstIP = row[18]
        elif row[12] != "":  # ARP
            srcIP = row[14]  # src IP (ARP)
            dstIP = row[16]  # dst IP (ARP)
        else:
            srcIP = row[2]  # src MAC
            dstIP = row[3]  # dst MAC

        srcport = row[6] + row[8]  # UDP or TCP port: the concatenation of the two port strings will will results in an OR "[tcp|udp]"
        dstport = row[7] + row[9]  # UDP or TCP port

        #self.pkt.srcaddr = srcIP + ":" + srcport
        #self.pkt.dstaddr = dstIP + ":" + dstport
        self.pkt.srcaddr = srcIP
        self.pkt.dstaddr = dstIP


        return self.pkt

    def get_next_vector(self) -> packetData:
        if self.curPacketIndx == 0:
            _ = self.get_next_packet()
            self.curPacketIndx += 1

        row = self.get_next_packet()
        if len(row) == 0:
            self.pkt.clear()
            return self.pkt

        self.interpret_packet(row)
        return self.pkt

    def get_num_packets(self) -> int:
        totalpackets = int(self.num_lines - 1)
        return totalpackets

    def close(self):
        if self.tsvinf.closed == False:
            self.tsvinf.close()

    def rewind(self):
        self.tsvinf.seek(0)
        _ = self.tsvin.__next__()
        self.curPacketIndx = 1
