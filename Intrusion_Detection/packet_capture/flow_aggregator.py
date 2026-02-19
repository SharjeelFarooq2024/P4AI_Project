"""Flow aggregation utilities to turn packets into bi-directional flows."""
from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Tuple

from scapy.all import IP, IPv6, Packet, TCP, UDP


class FlowRecord:
    """Mutable container for flow stats keyed by the first-seen 5-tuple."""

    def __init__(self, key: Tuple[str, int, str, int, str], ts: float) -> None:
        self.key = key
        self.src, self.sport, self.dst, self.dport, self.proto = key
        self.start_ts = ts
        self.last_ts = ts

        # Directional counters relative to the first packet's direction.
        self.src_bytes = 0
        self.dst_bytes = 0
        self.src_pkts = 0
        self.dst_pkts = 0

        self.src_ttls: List[int] = []
        self.dst_ttls: List[int] = []

        self.src_last_ts: Optional[float] = None
        self.dst_last_ts: Optional[float] = None
        self.src_intervals: List[float] = []
        self.dst_intervals: List[float] = []

        self.src_win: Optional[int] = None
        self.dst_win: Optional[int] = None
        self.src_tcp_base_seq: Optional[int] = None
        self.dst_tcp_base_seq: Optional[int] = None

        self.syn_ts: Optional[float] = None
        self.synack_ts: Optional[float] = None
        self.ack_ts: Optional[float] = None

        self.http_methods = set()
        self.ftp_cmds = 0
        self.state_flags = set()

    # pylint: disable=too-many-branches
    def update(self, pkt: Packet, ts: float) -> None:
        """Update counters for a packet that belongs to this flow."""
        self.last_ts = ts

        src_ip, dst_ip, src_port, dst_port, proto = _packet_tuple(pkt)
        if src_ip is None:
            return

        direction_src = src_ip == self.src and src_port == self.sport
        byte_len = int(len(pkt))

        # Direction-sensitive counters.
        if direction_src:
            self.src_bytes += byte_len
            self.src_pkts += 1
            if self.src_last_ts is not None:
                self.src_intervals.append(max(0.0, ts - self.src_last_ts))
            self.src_last_ts = ts
        else:
            self.dst_bytes += byte_len
            self.dst_pkts += 1
            if self.dst_last_ts is not None:
                self.dst_intervals.append(max(0.0, ts - self.dst_last_ts))
            self.dst_last_ts = ts

        # TTL tracking.
        if IP in pkt:
            (self.src_ttls if direction_src else self.dst_ttls).append(int(pkt[IP].ttl))
        elif IPv6 in pkt:
            (self.src_ttls if direction_src else self.dst_ttls).append(int(pkt[IPv6].hlim))

        # TCP-specific details.
        if TCP in pkt:
            tcp_seg = pkt[TCP]
            self.state_flags.add(tcp_seg.flags)
            if direction_src:
                self.src_win = int(tcp_seg.window)
                if self.src_tcp_base_seq is None:
                    self.src_tcp_base_seq = int(tcp_seg.seq)
            else:
                self.dst_win = int(tcp_seg.window)
                if self.dst_tcp_base_seq is None:
                    self.dst_tcp_base_seq = int(tcp_seg.seq)

            # Estimate handshake timings where possible.
            flags = tcp_seg.flags
            if flags & 0x02 and not flags & 0x10:  # SYN without ACK
                if direction_src:
                    self.syn_ts = self.syn_ts or ts
            if flags & 0x12:  # SYN-ACK
                if not direction_src:
                    self.synack_ts = self.synack_ts or ts
            if flags & 0x10 and self.synack_ts is not None and self.ack_ts is None:
                self.ack_ts = ts

            # Naive HTTP method detection for mapping to ct_flw_http_mthd.
            if pkt.haslayer("Raw"):
                payload: bytes = bytes(pkt["Raw"].load)
                for method in (b"GET", b"POST", b"PUT", b"DELETE", b"HEAD", b"OPTIONS"):
                    if payload.startswith(method + b" "):
                        self.http_methods.add(method.decode("ascii", errors="ignore"))
                        break

            # FTP command counter if either port is 21.
            if 21 in (src_port, dst_port) and pkt.haslayer("Raw"):
                self.ftp_cmds += 1

    def to_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "src": self.src,
            "sport": self.sport,
            "dst": self.dst,
            "dport": self.dport,
            "proto": self.proto,
            "start_ts": self.start_ts,
            "last_ts": self.last_ts,
            "src_bytes": self.src_bytes,
            "dst_bytes": self.dst_bytes,
            "src_pkts": self.src_pkts,
            "dst_pkts": self.dst_pkts,
            "src_ttls": self.src_ttls,
            "dst_ttls": self.dst_ttls,
            "src_intervals": self.src_intervals,
            "dst_intervals": self.dst_intervals,
            "src_win": self.src_win,
            "dst_win": self.dst_win,
            "src_tcp_base_seq": self.src_tcp_base_seq,
            "dst_tcp_base_seq": self.dst_tcp_base_seq,
            "syn_ts": self.syn_ts,
            "synack_ts": self.synack_ts,
            "ack_ts": self.ack_ts,
            "http_methods": list(self.http_methods),
            "ftp_cmds": self.ftp_cmds,
            "state_flags": list(self.state_flags),
        }


class FlowAggregator:
    """Collect packets into flows with idle and active timeouts."""

    def __init__(self, active_timeout: int = 60, idle_timeout: int = 15) -> None:
        self.active_timeout = active_timeout
        self.idle_timeout = idle_timeout
        self.flows: Dict[Tuple[str, int, str, int, str], FlowRecord] = {}
        self.completed: List[FlowRecord] = []

    def _touch(self, key: Tuple[str, int, str, int, str], pkt: Packet, ts: float) -> FlowRecord:
        if key not in self.flows:
            self.flows[key] = FlowRecord(key, ts)
        record = self.flows[key]
        record.update(pkt, ts)
        return record

    def ingest(self, pkt: Packet) -> None:
        ts = float(pkt.time)
        key = _packet_tuple(pkt)
        if key[0] is None:
            return

        # Bidirectional merge: reuse reversed flow if present.
        reverse_key = (key[2], key[3], key[0], key[1], key[4])
        chosen_key = key if key in self.flows else (reverse_key if reverse_key in self.flows else key)
        record = self._touch(chosen_key, pkt, ts)

        # Close flows that are idle or exceed active window.
        self._evict_expired(ts)

    def _evict_expired(self, now: float) -> None:
        to_close = []
        for key, record in self.flows.items():
            idle = now - record.last_ts > self.idle_timeout
            active = now - record.start_ts > self.active_timeout
            if idle or active:
                to_close.append(key)
        for key in to_close:
            self.completed.append(self.flows.pop(key))

    def flush(self) -> List[FlowRecord]:
        """Return all flows, including completed and currently open ones."""
        self.completed.extend(self.flows.values())
        self.flows = {}
        out, self.completed = self.completed, []
        return out

    def process_packets(self, packets: Iterable[Packet]) -> List[FlowRecord]:
        for pkt in packets:
            self.ingest(pkt)
        return self.flush()


def _packet_tuple(pkt: Packet) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[int], str]:
    """Extract a normalized 5-tuple; returns (None, ...) if not IP."""
    if IP in pkt:
        proto_name = _proto_name(pkt[IP].proto)
        sport = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
        dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
        return pkt[IP].src, sport, pkt[IP].dst, dport, proto_name
    if IPv6 in pkt:
        proto_name = _proto_name(pkt[IPv6].nh)
        sport = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
        dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
        return pkt[IPv6].src, sport, pkt[IPv6].dst, dport, proto_name
    return None, None, None, None, "UNKNOWN"


def _proto_name(proto_num: int) -> str:
    if proto_num == 6:
        return "tcp"
    if proto_num == 17:
        return "udp"
    if proto_num == 1:
        return "icmp"
    return str(proto_num)
