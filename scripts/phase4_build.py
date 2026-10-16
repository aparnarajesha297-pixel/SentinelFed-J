"""Phase 4: build features and representations, checking shapes row by row.

    python scripts/phase4_build.py                         # everything
    python scripts/phase4_build.py --segments 0-4 12-0     # small dev run

Pipeline (each step prints its output shape):
  edge features -> windows -> ego graphs -> activity matrix -> LC -> CE -> GC -> fusion

LC, GC and fusion are not trained here; the supervised branch trains them
end to end in Phase 5. CE is pretrained here with the guide's masked
reconstruction objective on S_train windows only. The embeddings saved by
this phase are therefore pipeline outputs for shape checks, and Phase 5
regenerates them from the trained modules.
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from sentinelfed.edges import edge_features
from sentinelfed.graphs import GraphConfig, WindowBuilder, record_matrix
from sentinelfed.modules import CEModule, GCModule, LCModule, WeightedFusion, ego_batch

SPLIT_FILES = {"train": "S_train.csv", "val": "S_val.csv", "test": "S_test.csv"}


def log_shape(rows: list, name: str, value, note: str = "") -> None:
    shape = value if isinstance(value, tuple) else tuple(np.shape(value))
    rows.append({"component": name, "shape": "x".join(map(str, shape)), "note": note})
    print(f"  {name:28s} {str(shape):24s} {note}")


@torch.no_grad()
def run_lc(lc, ego, windows, batch=2048):
    out = []
    prev = windows["prev_window"].to_numpy()
    ids = np.arange(len(windows))
    for s in range(0, len(ids), batch):
        b = ids[s:s + batch]
        p = np.where(prev[b] >= 0, prev[b], b)       # no history: compare with itself
        out.append(lc(ego_batch(ego, b), ego_batch(ego, p)))
    return torch.cat(out)


def seq_tensor(h, seq_idx, rows):
    idx = torch.from_numpy(seq_idx[rows])
    pad = idx < 0
    seq = h[idx.clamp(min=0)]
    seq[pad] = 0.0
    return seq, pad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--out", default="data/phase4")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--results", default="results")
    ap.add_argument("--segments", nargs="*", default=None, help="restrict to these recordings")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--ce-epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--float16", action="store_true", help="store ego graphs in float16")
    ap.add_argument("--graphs-only", action="store_true",
                    help="stop after saving windows, graphs and activity (no demo embeddings)")
    args = ap.parse_args()
    out, art, res = Path(args.out), Path(args.artifacts), Path(args.results)
    for p in (out, art, res):
        p.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rows: list = []
    t0 = time.time()

    parts = []
    for split, fname in SPLIT_FILES.items():
        d = pd.read_csv(Path(args.splits) / fname, low_memory=False,
                        dtype={"segment_id": str, "http.request.method": str,
                               "http.request.version": str})
        if args.segments:
            d = d[d["segment_id"].isin(args.segments)]
        parts.append(d.assign(split=split))
    df = pd.concat(parts, ignore_index=True)
    print(f"records: {len(df):,} from {df['segment_id'].nunique()} recordings")

    # ---- 1. edge features ----------------------------------------------------
    feats = edge_features(df)
    df = df.set_index("record_id").loc[feats["record_id"]].reset_index()
    feats.to_csv(out / "features.csv", index=False)
    log_shape(rows, "features.csv", feats.shape, "one row per record")

    pre = joblib.load(art / "preprocess_supervised.joblib")
    packet_X = pre["preprocessor"].transform(df[pre["features"]])
    packet_X = packet_X.toarray() if hasattr(packet_X, "toarray") else packet_X
    train = (feats["split"] == "train").to_numpy()
    eng = np.column_stack([np.log1p(feats["time_gap"]), np.log(feats["byte_ratio"])])
    edge_scaler = StandardScaler().fit(eng[train])                 # S_train only
    joblib.dump(edge_scaler, art / "edge_scaler.joblib")
    X = record_matrix(feats, packet_X.astype(np.float32), edge_scaler)
    log_shape(rows, "record vectors", X.shape, "scaled packet fields + edge features")

    # ---- 2-4. windows, ego graphs, activity --------------------------------
    cfg = GraphConfig(window=args.window, hops=args.hops, float16=args.float16)
    built = WindowBuilder(feats, X, cfg).build()
    windows = built["windows"]
    print(f"  windows built in {time.time() - t0:.0f}s")
    with open(out / "windows.pkl", "wb") as f:
        pickle.dump({"windows": windows, "seq_idx": built["seq_idx"],
                     "center_feat": built["center_feat"], "config": built["config"]}, f)
    if args.float16:
        (out / "ego_graphs").mkdir(exist_ok=True)
        for k, v in built["ego"].items():
            np.save(out / "ego_graphs" / f"{k}.npy", v)
    else:
        with open(out / "ego_graphs.pkl", "wb") as f:
            pickle.dump(built["ego"], f)
    with open(out / "activity_flags.pkl", "wb") as f:
        pickle.dump(built["activity"], f)
    ego, act = built["ego"], built["activity"]
    log_shape(rows, "windows", windows.shape, f"w={cfg.window} one-second slots, per device")
    log_shape(rows, "ego graphs: node features", ego["node_feat"].shape,
              f"{len(windows)} graphs, mean {windows['n_nodes'].mean():.2f} nodes, "
              f"max {windows['n_nodes'].max()}, k={cfg.hops}")
    log_shape(rows, "ego graphs: edge attributes", ego["edge_attr"].shape,
              f"mean {windows['n_edges'].mean():.2f} edges per graph")
    log_shape(rows, "activity: own pattern", act["own"].shape, f"P={cfg.pattern_len} slots")
    log_shape(rows, "activity: other devices", act["others"].shape,
              f"M={cfg.max_others}; mean {act['others_mask'].sum(1).mean():.2f} others present")

    if args.graphs_only:
        pd.DataFrame(rows).to_csv(res / "phase4_shapes.csv", index=False)
        print(f"graphs only: done in {time.time() - t0:.0f}s")
        return

    # ---- 5. LC ---------------------------------------------------------------
    H = args.hidden
    node_dim, edge_dim = ego["node_feat"].shape[1], ego["edge_attr"].shape[1]
    lc = LCModule(node_dim, edge_dim, H).eval()
    h_L = run_lc(lc, ego, windows)
    log_shape(rows, "LC h_L", tuple(h_L.shape), "untrained (Phase 5 trains it)")

    # ---- 6. CE: masked-reconstruction pretraining on S_train ---------------
    ce = CEModule(H, cfg.seq_len)
    opt = torch.optim.Adam(ce.parameters(), lr=1e-3)
    tr = np.flatnonzero((windows["split"] == "train").to_numpy())
    rng = np.random.default_rng(args.seed)
    losses = []
    for epoch in range(args.ce_epochs):
        ce.train()
        perm = rng.permutation(tr)[:200_000]
        ep = []
        for s in range(0, len(perm), 512):
            seq, pad = seq_tensor(h_L, built["seq_idx"], perm[s:s + 512])
            loss = ce.masked_loss(seq, pad)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep.append(loss.item())
        losses.append(float(np.mean(ep)))
        print(f"  CE epoch {epoch + 1}: masked reconstruction MSE {losses[-1]:.4f}")
    ce.eval()
    with torch.no_grad():
        h_C = torch.cat([ce(*seq_tensor(h_L, built["seq_idx"], np.arange(s, min(s + 4096, len(windows)))))
                         for s in range(0, len(windows), 4096)])
    log_shape(rows, "CE h_C", tuple(h_C.shape), f"L={cfg.seq_len}; masked-recon MSE {losses[0]:.3f}->{losses[-1]:.3f}")

    # ---- 7. GC -----------------------------------------------------------------
    gc = GCModule(cfg.pattern_len, H).eval()
    with torch.no_grad():
        h_G = torch.cat([gc(torch.from_numpy(act["own"][s:s + 4096]),
                            torch.from_numpy(act["others"][s:s + 4096]),
                            torch.from_numpy(act["others_mask"][s:s + 4096]))
                         for s in range(0, len(windows), 4096)])
    log_shape(rows, "GC h_G", tuple(h_G.shape), "untrained (Phase 5 trains it)")

    # ---- 8. fusion -----------------------------------------------------------
    fusion = WeightedFusion(H).eval()
    with torch.no_grad():
        h, w = fusion([h_L, h_C, h_G])
    log_shape(rows, "Fusion h", tuple(h.shape), "untrained; view weights saved alongside")

    wid = windows["window_id"].to_numpy()
    for name, emb in [("lc", h_L), ("ce", h_C), ("gc", h_G)]:
        with open(out / f"{name}_embeddings.pkl", "wb") as f:
            pickle.dump({"window_id": wid, "embedding": emb.numpy()}, f)
    with open(out / "fused_embeddings.pkl", "wb") as f:
        pickle.dump({"window_id": wid, "embedding": h.numpy(), "view_weights": w.numpy(),
                     "views": ["LC", "CE", "GC"]}, f)
    torch.save({"lc": lc.state_dict(), "ce": ce.state_dict(), "gc": gc.state_dict(),
                "fusion": fusion.state_dict(), "node_dim": node_dim, "edge_dim": edge_dim,
                "hidden": H, "config": cfg.__dict__}, art / "phase4_modules.pt")

    tag = "_dev" if args.segments else ""
    pd.DataFrame(rows).to_csv(res / f"phase4_shapes{tag}.csv", index=False)
    ex = feats[feats["split"] == "train"].iloc[[min(5, (feats["split"] == "train").sum() - 1)]]
    ex.to_csv(res / f"phase4_feature_example{tag}.csv", index=False)
    win_counts = windows.groupby(["split", "attack_type"]).size().unstack(0).fillna(0).astype(int)
    win_counts.to_csv(res / f"phase4_window_counts{tag}.csv")
    (res / f"run_log_phase4{tag}.json").write_text(json.dumps({
        "phase": 4, "seed": args.seed, "segments": args.segments or "all",
        "graph_config": cfg.__dict__, "hidden": H, "ce_epochs": args.ce_epochs,
        "ce_masked_mse": losses, "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"done in {time.time() - t0:.0f}s")
    print(win_counts.to_string())
    print("example features.csv row:\n", ex.T.to_string())


if __name__ == "__main__":
    main()
