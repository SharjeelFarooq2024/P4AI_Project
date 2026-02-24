"""Packet capture helpers built around Scapy."""
from typing import List, Optional

from scapy.all import Packet, rdpcap, sniff, wrpcap


def capture_packets(
    iface: Optional[str] = None,
    duration: Optional[int] = None,
    count: int = 0,
    bpf_filter: Optional[str] = None,
    promisc: bool = True,
    output_pcap: Optional[str] = None,
) -> List[Packet]:
    """
    Sniff packets from an interface.
    - duration: seconds to sniff; None means run until count is reached or manually stopped.
    - count: maximum number of packets (0 means unlimited within duration).
    - bpf_filter: optional BPF filter string (e.g., "tcp", "port 80").
    - promisc: toggle promiscuous mode.
    - output_pcap: optional path to write captured packets.
    """
    packets = sniff(
        iface=iface,
        count=count if count > 0 else 0,
        timeout=duration,
        filter=bpf_filter,
        store=True,
        promisc=promisc,
    )

    if output_pcap:
        wrpcap(output_pcap, packets)

    return packets


def read_pcap(pcap_path: str) -> List[Packet]:
    """Load packets from an existing pcap file."""
    return rdpcap(pcap_path)
