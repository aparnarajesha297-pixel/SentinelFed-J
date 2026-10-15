"""Phase 1: audit the raw Edge-IIoTset DNN CSV before anything is built on it.

    python scripts/phase1_audit.py --csv data/DNN-EdgeIIoT-dataset.csv

Writes results/dataset_audit.csv (the guide's table) and
results/dataset_audit_per_class.csv (counts, time windows and file positions
per Attack_type), and prints both.
"""

import argparse
import datetime as dt
import hashlib
from pathlib import Path

import pandas as pd

TIME_RE = r"^\s*\d{4} \d\d:\d\d:\d\d\.\d+\s*$"
IMPORTANT = ["frame.time", "ip.src_host", "ip.dst_host", "tcp.srcport", "tcp.dstport",
             "tcp.flags", "tcp.len", "udp.port", "Attack_label", "Attack_type"]
NULL_IPS = {"0", "0.0", ""}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/DNN-EdgeIIoT-dataset.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--downloaded", default=None, help="download date, YYYY-MM-DD")
    args = ap.parse_args()

    path, out = Path(args.csv), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Read everything as text: several columns mix numbers and strings, and the
    # audit must see the raw values, not pandas' guesses.
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    n = len(df)

    label = df["Attack_type"]
    is_normal = label == "Normal"
    ts_ok = df["frame.time"].fillna("").str.match(TIME_RE)
    tod = pd.to_timedelta(df.loc[ts_ok, "frame.time"].str.strip().str.slice(5))

    src, dst = df["ip.src_host"].fillna(""), df["ip.dst_host"].fillna("")
    ips = (set(src) | set(dst)) - NULL_IPS
    private = {ip for ip in ips if ip.startswith("192.168.")}

    dup_all = int(df.duplicated().sum())
    dup_no_time = int(df.drop(columns="frame.time").duplicated().sum())
    runs = int((label != label.shift()).sum())

    missing = df[IMPORTANT].isna().sum()
    zero_ip = int(((src.isin(NULL_IPS)) | (dst.isin(NULL_IPS))).sum())
    attack_counts = label[~is_normal].value_counts()

    rows = [
        ("Dataset version", f"{path.name}; sha256 {sha256(path)}; "
                            f"{path.stat().st_size} bytes; downloaded "
                            f"{args.downloaded or dt.date.today().isoformat()} from Kaggle "
                            "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot"),
        ("Total records", n),
        ("Unique devices", f"{len(ips)} distinct IPs in ip.src_host/ip.dst_host "
                           f"({len(private)} in 192.168.*; the rest are mostly spoofed "
                           "sources in DDoS_UDP). Excludes placeholder 0 / 0.0"),
        ("Normal records", int(is_normal.sum())),
        ("Attack records", int((~is_normal).sum())),
        ("Attack categories", f"{len(attack_counts)}: " +
                              "; ".join(f"{k} {v}" for k, v in attack_counts.items())),
        ("Time range", f"frame.time has year + time of day only (no month/day). "
                       f"Parseable on {int(ts_ok.sum())} rows: {tod.min()} to {tod.max()} "
                       f"time of day, year 2021. Unparseable on {int((~ts_ok).sum())} rows "
                       "(all DDoS_UDP and MITM rows, plus a handful of others)"),
        ("Missing values", "; ".join(f"{k} {v}" for k, v in missing.items()) +
                           f". Separately, {zero_ip} rows have 0/0.0 as an IP placeholder"),
        ("Duplicate records", f"{dup_all} exact duplicates over all 63 columns; "
                              f"{dup_no_time} if frame.time is ignored. "
                              "Rule: drop exact duplicates, keep first occurrence"),
        ("Class imbalance", f"attack : normal = {(~is_normal).sum() / is_normal.sum():.3f} : 1 "
                            f"({(~is_normal).mean():.1%} attack)"),
        ("File structure", f"Rows form {runs} contiguous label blocks in file order, "
                           "one per Attack_type: separate captures concatenated"),
    ]
    audit = pd.DataFrame(rows, columns=["Audit item", "Value"])
    audit.to_csv(out / "dataset_audit.csv", index=False)

    pos = pd.Series(range(n), index=df.index)
    per = pd.DataFrame({
        "records": label.value_counts(),
        "first_row": pos.groupby(label).min(),
        "last_row": pos.groupby(label).max(),
        "timestamp_ok": ts_ok.groupby(label).sum(),
        "duplicates": df.duplicated().groupby(label).sum(),
    })
    tod_h = (tod.dt.total_seconds() / 3600).groupby(label[ts_ok])
    per["tod_start"] = tod_h.min().map(lambda h: f"{int(h):02d}:{int(h % 1 * 60):02d}")
    per["tod_end"] = tod_h.max().map(lambda h: f"{int(h):02d}:{int(h % 1 * 60):02d}")
    per = per.sort_values("first_row")
    per.index.name = "Attack_type"
    per.to_csv(out / "dataset_audit_per_class.csv")

    with pd.option_context("display.max_colwidth", 200, "display.width", 200):
        print(audit.to_string(index=False))
        print()
        print(per.to_string())


if __name__ == "__main__":
    main()
