"""Phase 2: build the clean record table from the raw Edge-IIoTset DNN CSV.

    python scripts/phase2_clean.py --csv data/DNN-EdgeIIoT-dataset.csv

Outputs
  data/processed/processed_data.csv   clean table (git-ignored, ~2.1M rows)
  data/processed/excluded_rows.csv    rows left out, with the reason
  results/phase2_summary.csv          counts per Attack_type and segment
  results/phase2_columns.csv          every raw column: kept / dropped and why

What "clean" means here (see results/phase2_columns.csv for the column list):
  * exact duplicate rows are dropped, first occurrence kept (Phase 1 rule);
  * DDoS_UDP and MITM rows are excluded: their fields are shifted between
    columns in every CSV release of the dataset (Phase 1 finding), so no
    field in them can be trusted;
  * nothing is scaled or encoded. Categorical fields stay as raw strings so
    encoders can be fit on the training split only (Phase 3 onward).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TIME_RE = r"^\s*\d{4} \d\d:\d\d:\d\d\.\d+\s*$"
CORRUPT_CLASSES = {"DDoS_UDP": "fields shifted between columns in source CSV",
                   "MITM": "fields shifted between columns in source CSV"}
NULL_IPS = {"0", "0.0", ""}

# A new recording starts when the clock jumps back more than this, or forward
# by more than NEW_SEGMENT_FWD. Small backward steps are packet reordering.
NEW_SEGMENT_BACK = 600.0
NEW_SEGMENT_FWD = 3600.0
DAY = 86400.0

NUMERIC_FIELDS = [
    "tcp.srcport", "tcp.dstport", "udp.port", "tcp.flags", "tcp.flags.ack", "tcp.len",
    "tcp.seq", "tcp.ack", "tcp.connection.fin", "tcp.connection.rst",
    "tcp.connection.syn", "tcp.connection.synack", "udp.time_delta",
    "icmp.seq_le", "arp.opcode", "arp.hw.size",
    "http.content_length", "http.response", "http.tls_port",
    "dns.qry.qu", "dns.qry.type", "dns.retransmission", "dns.retransmit_request",
    "mqtt.hdrflags", "mqtt.len", "mqtt.msgtype", "mqtt.conflags",
    "mqtt.conflag.cleansess", "mqtt.proto_len", "mqtt.topic_len", "mqtt.ver",
    "mbtcp.len", "mbtcp.trans_id", "mbtcp.unit_id",
]
CATEGORICAL_FIELDS = ["http.request.method", "http.request.version"]
DROPPED = {
    "frame.time": "replaced by Timestamp / time_s / segment_id",
    "ip.src_host": "renamed DeviceSrc",
    "ip.dst_host": "renamed DeviceDst",
    "arp.dst.proto_ipv4": "IP address: device identity, leaks label (Phase 1 finding 3)",
    "arp.src.proto_ipv4": "IP address: device identity, leaks label (Phase 1 finding 3)",
    "icmp.checksum": "checksum: per-packet noise",
    "tcp.checksum": "checksum: per-packet noise",
    "tcp.ack_raw": "raw sequence number: per-connection random",
    "icmp.transmit_timestamp": "absolute clock value, not a behaviour feature",
    "icmp.unused": "always empty",
    "udp.stream": "capture-tool stream index: depends on recording, not behaviour",
    "dns.retransmit_request_in": "frame index into the capture file",
    "http.file_data": "free-text payload",
    "http.request.uri.query": "free-text payload",
    "http.request.full_uri": "free-text payload, contains target IP",
    "http.referer": "free-text payload",
    "tcp.options": "raw hex bytes",
    "tcp.payload": "raw hex payload",
    "mqtt.msg": "raw hex payload (sensor readings)",
    "mqtt.topic": "single constant topic name",
    "mqtt.protoname": "constant 'MQTT' (protocol column covers it)",
    "mqtt.msg_decoded_as": "constant",
    "mqtt.conack.flags": "hex string, constant",
    "dns.qry.name": "free-text query name",
    "dns.qry.name.len": "holds query names in this release (mislabelled column)",
    "Attack_label": "kept, see output",
    "Attack_type": "kept, see output",
}


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def protocol(df: pd.DataFrame) -> pd.Series:
    """Highest-level protocol seen in the packet."""
    n = {c: to_num(df[c]) for c in ["arp.opcode", "mqtt.msgtype", "mqtt.len", "mbtcp.len",
                                    "http.response", "dns.qry.type", "icmp.seq_le",
                                    "tcp.flags", "tcp.srcport", "tcp.dstport", "udp.port"]}
    has_http = df["http.request.method"].fillna("0").ne("0") & df["http.request.method"].fillna("0").ne("0.0")
    conds = [
        n["arp.opcode"] > 0,
        (n["mqtt.msgtype"] > 0) | (n["mqtt.len"] > 0),
        n["mbtcp.len"] > 0,
        has_http | (n["http.response"] > 0),
        n["dns.qry.type"] > 0,
        n["icmp.seq_le"] > 0,
        (n["tcp.flags"] > 0) | (n["tcp.srcport"] > 0) | (n["tcp.dstport"] > 0),
        n["udp.port"] > 0,
    ]
    names = ["ARP", "MQTT", "MODBUS", "HTTP", "DNS", "ICMP", "TCP", "UDP"]
    return pd.Series(np.select(conds, names, default="OTHER"), index=df.index)


def segments(tod: np.ndarray, block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split each class block into recordings and unwrap midnight.

    Returns (segment index within block, unwrapped seconds).
    """
    seg = np.zeros(len(tod), dtype=np.int64)
    unwrapped = tod.copy()
    for b in np.unique(block):
        idx = np.flatnonzero(block == b)
        t = tod[idx]
        s, offset = 0, 0.0
        out_seg, out_t = np.empty(len(t), np.int64), np.empty(len(t))
        out_seg[0], out_t[0] = 0, t[0]
        for i in range(1, len(t)):
            d = t[i] - t[i - 1]
            if d < -NEW_SEGMENT_BACK:
                if t[i - 1] > DAY - NEW_SEGMENT_BACK and t[i] < NEW_SEGMENT_BACK:
                    offset += DAY            # crossed midnight, same recording
                else:
                    s, offset = s + 1, 0.0
            elif d > NEW_SEGMENT_FWD:
                s, offset = s + 1, 0.0
            out_seg[i], out_t[i] = s, t[i] + offset
        seg[idx], unwrapped[idx] = out_seg, out_t
    return seg, unwrapped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/DNN-EdgeIIoT-dataset.csv")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    out, res = Path(args.out), Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.csv, dtype=str, keep_default_na=False, na_values=[""])
    raw.insert(0, "record_id", np.arange(len(raw)))
    n_raw = len(raw)

    label = raw["Attack_type"]
    raw["capture_id"] = (label != label.shift()).cumsum() - 1   # file block order

    dup = raw.drop(columns=["record_id", "capture_id"]).duplicated()
    corrupt = label.isin(CORRUPT_CLASSES)
    bad_time = ~raw["frame.time"].fillna("").str.match(TIME_RE) & ~corrupt
    reason = np.select([dup, corrupt, bad_time],
                       ["exact duplicate", "corrupt class", "unparseable timestamp"], "")
    excluded = raw.loc[reason != "", ["record_id", "capture_id", "Attack_type"]].copy()
    excluded["reason"] = reason[reason != ""]
    excluded.loc[excluded.reason == "corrupt class", "reason"] = (
        "corrupt class: " + excluded["Attack_type"].map(CORRUPT_CLASSES))
    excluded.to_csv(out / "excluded_rows.csv", index=False)

    df = raw.loc[reason == ""].reset_index(drop=True)
    tod = pd.to_timedelta(df["frame.time"].str.strip().str.slice(5)).dt.total_seconds().to_numpy()
    seg, t_unwrapped = segments(tod, df["capture_id"].to_numpy())

    clean = pd.DataFrame({
        "record_id": df["record_id"],
        "capture_id": df["capture_id"],
        "segment_id": df["capture_id"].astype(str) + "-" + pd.Series(seg).astype(str),
        "Timestamp": df["frame.time"].str.strip().str.slice(5),
        "time_s": np.round(t_unwrapped, 6),
        "DeviceSrc": df["ip.src_host"].where(~df["ip.src_host"].isin(NULL_IPS), "none"),
        "DeviceDst": df["ip.dst_host"].where(~df["ip.dst_host"].isin(NULL_IPS), "none"),
        "protocol": protocol(df),
    })
    for c in NUMERIC_FIELDS:
        clean[c] = to_num(df[c])
    for c in CATEGORICAL_FIELDS:
        v = df[c].fillna("none")
        clean[c] = v.where(~v.isin(["0", "0.0"]), "none")
    clean["Attack_label"] = df["Attack_label"].astype(int)
    clean["Attack_type"] = df["Attack_type"]
    clean.to_csv(out / "processed_data.csv", index=False)

    cols = [(c, "kept", "") for c in NUMERIC_FIELDS + CATEGORICAL_FIELDS]
    cols += [(c, "transformed" if c in ("frame.time", "ip.src_host", "ip.dst_host") else
              ("kept" if c.startswith("Attack") else "dropped"), why)
             for c, why in DROPPED.items()]
    missing = set(raw.columns) - {"record_id", "capture_id"} - {c for c, _, _ in cols}
    assert not missing, f"columns not accounted for: {missing}"
    pd.DataFrame(cols, columns=["raw_column", "status", "reason"]).to_csv(
        res / "phase2_columns.csv", index=False)

    seg_summary = clean.groupby(["Attack_type", "segment_id"]).agg(
        records=("record_id", "size"), start=("Timestamp", "first"), end=("Timestamp", "last"),
        duration_s=("time_s", lambda s: round(s.max() - s.min(), 1)))
    seg_summary.to_csv(res / "phase2_segments.csv")

    devices = set(clean["DeviceSrc"]) | set(clean["DeviceDst"])
    devices.discard("none")
    hosts = sorted(d for d in devices if d.startswith("192.168."))
    per_class = clean["Attack_type"].value_counts().rename("records").to_frame()
    summary = pd.concat([
        pd.DataFrame({"item": ["raw records", "excluded records", "clean records",
                               "unique devices (all IPs)", "LAN hosts (192.168.*)",
                               "segments (recordings)"],
                      "value": [n_raw, len(excluded), len(clean), len(devices),
                                f"{len(hosts)}: {' '.join(hosts)}",
                                clean["segment_id"].nunique()]}),
        per_class.reset_index().rename(columns={"Attack_type": "item", "records": "value"})
        .assign(item=lambda d: "records: " + d["item"]),
    ])
    summary.to_csv(res / "phase2_summary.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_colwidth", 120):
        print(summary.to_string(index=False))
        print()
        print(excluded["reason"].value_counts().to_string())
        print()
        print(seg_summary.to_string())
        print()
        print("protocol mix:\n", pd.crosstab(clean["Attack_type"], clean["protocol"]).to_string())
        print()
        print("example row:\n", clean.iloc[1000].to_string())


if __name__ == "__main__":
    main()
