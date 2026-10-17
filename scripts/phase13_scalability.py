"""Phase 13: scalability with 10, 50, 100, 200 and 500 devices per network.

    python scripts/phase13_scalability.py            # all sizes, one subprocess each
    python scripts/phase13_scalability.py --report

Edge-IIoTset has only four real LAN hosts, so networks of N devices are
synthesised by replaying real traffic: a 120 s slice of normal sensor<->
broker conversation from the test split is the template; every synthetic
sensor replays it with its own random time offset (wrapping around the
slice), talking to one of ceil(N/10) brokers. Each size runs in a fresh
process so peak memory is measured cleanly.

Processing is streamed slot by slot (1 s), as a live system would:
for every slot, build the windows/ego graphs/activity patterns ending in
that slot is done once for the whole stream (graph construction time is
reported in total and per window), then LC + head score that slot's
windows using the history already computed. Latency = graph-construction
share + scoring time for a slot. Model weights do not affect timing; the
trained Phase 5 modules are used.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

SIZES = [10, 50, 100, 200, 500]


def rss_mb():
    """Current resident memory (the high-water mark is set by importing torch)."""
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
SLICE_S = 120
C1, C2, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


TEMPLATE = Path("data/phase13/template.pkl")
SENSOR_IP, BROKER_IP = "192.168.0.101", "192.168.0.128"


def make_template():
    """Cut the 120 s replay template once, so each size's process loads only
    the template and peak memory reflects the pipeline, not CSV loading."""
    if TEMPLATE.exists():
        return
    test = pd.read_csv("data/splits/S_test.csv", low_memory=False,
                       dtype={"segment_id": str, "http.request.method": str, "http.request.version": str})
    seg = test[test["segment_id"] == "0-0"].sort_values("time_s")
    t0 = seg["time_s"].iloc[0]
    tpl = seg[seg["time_s"] < t0 + SLICE_S].copy()
    tpl["time_s"] -= t0
    ends = [SENSOR_IP, BROKER_IP]
    tpl = tpl[tpl["DeviceSrc"].isin(ends) & tpl["DeviceDst"].isin(ends)]
    TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
    tpl.to_pickle(TEMPLATE)


def synthesise(n_devices, seed=0):
    tpl = pd.read_pickle(TEMPLATE)
    sensor_ip, broker_ip = SENSOR_IP, BROKER_IP
    n_brokers = max(1, int(np.ceil(n_devices / 10)))
    n_sensors = n_devices - n_brokers
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(n_sensors):
        d = tpl.copy()
        d["time_s"] = (d["time_s"] + rng.uniform(0, SLICE_S)) % SLICE_S
        s, b = f"10.{i // 250}.{i % 250}.2", f"10.200.{i % n_brokers}.1"
        d["DeviceSrc"] = np.where(d["DeviceSrc"] == sensor_ip, s, b)
        d["DeviceDst"] = np.where(d["DeviceDst"] == sensor_ip, s, b)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True).sort_values("time_s", kind="stable").reset_index(drop=True)
    df["record_id"] = np.arange(len(df))
    df["segment_id"] = "syn"
    df["split"] = "test"
    return df, n_sensors, n_brokers


def run_size(n):
    from sentinelfed.edges import edge_features
    from sentinelfed.graphs import GraphConfig, WindowBuilder, record_matrix
    from sentinelfed.train import LCClassifier, SupervisedHead, collate, head_inputs, lc_batches

    torch.set_num_threads(1)          # one core: comparable across sizes
    base_rss_mb = rss_mb()            # after imports, before any data
    samples = [base_rss_mb]
    wall0, cpu0 = time.perf_counter(), sum(os.times()[:2])
    df, n_sensors, n_brokers = synthesise(n)
    samples.append(rss_mb())
    feats = edge_features(df)
    df = df.set_index("record_id").loc[feats["record_id"]].reset_index()
    pre = joblib.load("artifacts/preprocess_supervised.joblib")
    P = pre["preprocessor"].transform(df[pre["features"]])
    P = P.toarray() if hasattr(P, "toarray") else P
    X = record_matrix(feats, P.astype(np.float32), joblib.load("artifacts/edge_scaler.joblib"))
    samples.append(rss_mb())

    tb = time.perf_counter()
    built = WindowBuilder(feats, X, GraphConfig()).build()
    build_s = time.perf_counter() - tb
    samples.append(rss_mb())
    w, ego, act, seq_idx = built["windows"], built["ego"], built["activity"], built["seq_idx"]
    per_window_build_ms = 1000 * build_s / len(w)

    lc_ck = torch.load("artifacts/phase5/lc.pt", weights_only=False)
    hd_ck = torch.load("artifacts/phase5/head.pt", weights_only=False)
    lc = LCClassifier(lc_ck["node_dim"], lc_ck["edge_dim"], 64, len(lc_ck["classes"]))
    lc.load_state_dict(lc_ck["state"])
    head = SupervisedHead(64, hd_ck["seq_len"], hd_ck["pattern_len"], len(hd_ck["classes"]), hd_ck["views"])
    head.load_state_dict(hd_ck["state"])
    lc.eval()
    head.eval()

    h_L = torch.zeros(len(w), 64)
    slots = w["end_slot"].to_numpy()
    lat, per_win = [], []
    ts0, cs0 = time.perf_counter(), sum(os.times()[:2])
    with torch.no_grad():
        for s in np.unique(slots):
            ids = np.flatnonzero(slots == s)
            t = time.perf_counter()
            h_L[ids] = lc.lc(collate(ego, ids), collate(ego, lc_batches(w, ids)))
            head(*head_inputs(h_L, seq_idx, act, ids))
            score_ms = 1000 * (time.perf_counter() - t)
            total = score_ms + per_window_build_ms * len(ids)
            lat.append(total)
            samples.append(rss_mb())
            per_win.append(total / len(ids))
    stream_wall = time.perf_counter() - ts0
    stream_cpu = sum(os.times()[:2]) - cs0
    lat = np.array(lat)
    return {
        "devices": n, "sensors": n_sensors, "brokers": n_brokers, "records": len(df),
        "windows": len(w), "slots": len(np.unique(slots)),
        "mean_latency_ms": float(lat.mean()), "p95_latency_ms": float(np.percentile(lat, 95)),
        "p99_latency_ms": float(np.percentile(lat, 99)),
        "mean_latency_per_window_ms": float(np.mean(per_win)),
        "cpu_utilisation_cores": stream_cpu / stream_wall,
        "cpu_seconds_total": sum(os.times()[:2]) - cpu0, "wall_seconds_total": time.perf_counter() - wall0,
        "peak_memory_MB": max(samples),
        "pipeline_memory_MB": max(samples) - base_rss_mb,
        "baseline_memory_MB": base_rss_mb,
        "graph_construction_s": build_s, "graph_construction_ms_per_window": per_window_build_ms,
        "mean_nodes": float(w["n_nodes"].mean()), "max_nodes": int(w["n_nodes"].max()),
    }


def report(res):
    t = pd.read_csv(res / "scalability_results.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    x = t["devices"]
    for col, lab, mk, c in [("mean_latency_ms", "Mean", "o", C1), ("p95_latency_ms", "P95", "s", C2),
                            ("p99_latency_ms", "P99", "^", "#1baf7a")]:
        a.plot(x, t[col], marker=mk, color=c, linewidth=1.8, markersize=6, markeredgecolor="white", label=lab)
    a.set_xlabel("Devices in the network")
    a.set_ylabel("Latency per 1 s slot (ms)")
    a.legend(frameon=False, fontsize=8)
    b.plot(x, t["pipeline_memory_MB"], marker="o", color=C1, linewidth=1.8, markersize=6, markeredgecolor="white")
    b.set_xlabel("Devices in the network")
    b.set_ylabel("Pipeline memory above baseline (MB)")
    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_xticks(list(x), [str(v) for v in x])
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED)
    fig.suptitle("Scalability: latency and memory vs devices (one CPU core, replayed traffic)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(res / "latency_vs_devices.png")
    plt.close(fig)
    with pd.option_context("display.width", 250):
        print(t.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--one", type=int, help="internal: run a single size and print JSON")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    res = Path(args.results)
    if args.one:
        print("RESULT " + json.dumps(run_size(args.one)), flush=True)
        return
    if not args.report:
        make_template()
        rows = []
        for n in args.sizes:
            out = subprocess.run([sys.executable, __file__, "--one", str(n)], capture_output=True, text=True)
            line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")]
            if not line:
                print(out.stderr[-2000:])
                raise SystemExit(f"size {n} failed")
            rows.append(json.loads(line[0][7:]))
            print(f"{n} devices: mean {rows[-1]['mean_latency_ms']:.1f} ms, p99 {rows[-1]['p99_latency_ms']:.1f} ms, "
                  f"pipeline memory {rows[-1]['pipeline_memory_MB']:.0f} MB", flush=True)
            pd.DataFrame(rows).to_csv(res / "scalability_results.csv", index=False)
        from sentinelfed.runlog import environment
        (res / "run_log_phase13.json").write_text(json.dumps({
            **environment(), "sizes": args.sizes, "slice_seconds": SLICE_S, "brokers": "ceil(N/10)",
            "threads": 1, "traffic": "replayed normal sensor<->broker traffic, test split, recording 0-0",
            "latency_definition": "per 1 s slot: graph-construction share + LC/head scoring of that slot's windows",
            "model": "Phase 5 trained LC + head (weights do not affect timing)"}, indent=2, default=str))
    report(res)


if __name__ == "__main__":
    main()
