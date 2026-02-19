"""Derive UNSW-NB15-style features from aggregated flows."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from packet_capture.flow_aggregator import FlowRecord
from src.data_loader import load_feature_names

# Minimal port-to-service hints for the "service" feature.
_PORT_SERVICE_MAP = {
    20: "ftp-data",
    21: "ftp",
    22: "ssh",
    23: "telnet",
    25: "smtp",
    53: "dns",
    80: "http",
    110: "pop3",
    143: "imap",
    443: "https",
    993: "imaps",
    995: "pop3s",
    3306: "mysql",
    3389: "rdp",
}


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: List[float]) -> float:
    return float(np.std(values)) if len(values) > 1 else 0.0


def _service_from_ports(sport: int, dport: int) -> str:
    if sport in _PORT_SERVICE_MAP:
        return _PORT_SERVICE_MAP[sport]
    if dport in _PORT_SERVICE_MAP:
        return _PORT_SERVICE_MAP[dport]
    return "-"


def flow_to_feature_row(flow: FlowRecord, feature_names: List[str]) -> Dict[str, object]:
    data: Dict[str, object] = {name: 0 for name in feature_names}

    duration = max(0.0, flow.last_ts - flow.start_ts)
    smean_sz = flow.src_bytes / flow.src_pkts if flow.src_pkts else 0.0
    dmean_sz = flow.dst_bytes / flow.dst_pkts if flow.dst_pkts else 0.0

    # TTL fallbacks: if one side is missing, reuse the other side's last TTL to avoid zeros.
    sttl_val = flow.src_ttls[-1] if flow.src_ttls else (flow.dst_ttls[-1] if flow.dst_ttls else 0)
    dttl_val = flow.dst_ttls[-1] if flow.dst_ttls else (flow.src_ttls[-1] if flow.src_ttls else 0)

    data.update(
        {
            "srcip": flow.src,
            "sport": flow.sport,
            "dstip": flow.dst,
            "dsport": flow.dport,
            "proto": flow.proto,
            "state": "CON",
            "dur": duration,
            "sbytes": flow.src_bytes,
            "dbytes": flow.dst_bytes,
            "sttl": sttl_val,
            "dttl": dttl_val,
            "sloss": 0,
            "dloss": 0,
            "service": _service_from_ports(flow.sport, flow.dport),
            "Sload": (flow.src_bytes * 8 / duration) if duration > 0 else 0.0,
            "Dload": (flow.dst_bytes * 8 / duration) if duration > 0 else 0.0,
            "Spkts": flow.src_pkts,
            "Dpkts": flow.dst_pkts,
            "swin": flow.src_win or 0,
            "dwin": flow.dst_win or 0,
            "stcpb": flow.src_tcp_base_seq or 0,
            "dtcpb": flow.dst_tcp_base_seq or 0,
            "smeansz": smean_sz,
            "dmeansz": dmean_sz,
            "trans_depth": 0,
            "res_bdy_len": 0,
            "Sjit": _safe_std(flow.src_intervals),
            "Djit": _safe_std(flow.dst_intervals),
            "Stime": flow.start_ts,
            "Ltime": flow.last_ts,
            "Sintpkt": _safe_mean(flow.src_intervals),
            "Dintpkt": _safe_mean(flow.dst_intervals),
            "tcprtt": (flow.ack_ts - flow.syn_ts) if flow.ack_ts and flow.syn_ts else 0.0,
            "synack": (flow.synack_ts - flow.syn_ts) if flow.synack_ts and flow.syn_ts else 0.0,
            "ackdat": (flow.ack_ts - flow.synack_ts) if flow.ack_ts and flow.synack_ts else 0.0,
            "is_sm_ips_ports": int(flow.src == flow.dst and flow.sport == flow.dport),
            "ct_state_ttl": len(flow.state_flags),
            "ct_flw_http_mthd": len(flow.http_methods),
            "is_ftp_login": 1 if flow.ftp_cmds > 0 and flow.dport == 21 else 0,
            "ct_ftp_cmd": flow.ftp_cmds,
            "ct_srv_src": 0,
            "ct_srv_dst": 0,
            "ct_dst_ltm": 0,
            "ct_src_ltm": 0,
            "ct_src_dport_ltm": 0,
            "ct_dst_sport_ltm": 0,
            "ct_dst_src_ltm": 0,
        }
    )

    # Keep zero defaults for any feature not mapped above.
    return {name: data.get(name, 0) for name in feature_names}


def flows_to_dataframe(flows: Iterable[FlowRecord]) -> pd.DataFrame:
    feature_names = load_feature_names()

    # Drop label-like columns we cannot determine during live capture.
    feature_names = [
        name for name in feature_names if name.lower() not in {"label", "attack_cat"}
    ]
    rows = [flow_to_feature_row(flow, feature_names) for flow in flows]
    df = pd.DataFrame(rows, columns=feature_names)

    # Ensure numeric columns are properly typed where obvious.
    non_numeric = {"proto", "state", "service", "srcip", "dstip"}
    numeric_cols = [col for col in df.columns if col not in non_numeric]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["proto"] = df["proto"].astype(str)
    df["state"] = df["state"].astype(str)
    df["service"] = df["service"].astype(str)
    return df


def features_from_packets(packets: Iterable) -> pd.DataFrame:
    """Convenience wrapper to aggregate packets then compute the feature frame."""
    from packet_capture.flow_aggregator import FlowAggregator

    aggregator = FlowAggregator()
    flows = aggregator.process_packets(packets)
    return flows_to_dataframe(flows)
