import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from sentinelfed.edges import edge_features
from sentinelfed.graphs import GraphConfig, WindowBuilder, record_matrix
from sentinelfed.modules import CEModule, GCModule, LCModule, WeightedFusion, ego_batch


def toy_records(n=300, seed=0):
    rng = np.random.default_rng(seed)
    devs = ["192.168.0.101", "192.168.0.128", "192.168.0.170", "10.0.0.9", "none"]
    src = rng.choice(devs[:4], n)
    dst = np.where(src == "192.168.0.128", rng.choice(devs, n), "192.168.0.128")
    t = np.sort(rng.uniform(0, 60, n))
    split = np.where(t < 40, "train", np.where(t < 50, "val", "test"))
    return pd.DataFrame({
        "record_id": np.arange(n), "split": split, "segment_id": "0-0", "time_s": t,
        "DeviceSrc": src, "DeviceDst": dst, "protocol": "TCP",
        "tcp.len": rng.integers(0, 100, n).astype(float),
        "tcp.connection.rst": 0.0, "tcp.connection.fin": 0.0, "tcp.connection.synack": 0.0,
        "tcp.connection.syn": rng.integers(0, 2, n).astype(float), "tcp.flags.ack": 0.0,
        "Attack_label": (src == "192.168.0.170").astype(int),
        "Attack_type": np.where(src == "192.168.0.170", "X", "Normal"),
    })


def test_edge_features_ignore_the_future():
    df = toy_records()
    full = edge_features(df).set_index("record_id")
    cut = 200
    future_changed = df.copy()
    future_changed.loc[cut:, "tcp.len"] = 9999.0
    future_changed.loc[cut:, "DeviceSrc"] = "192.168.0.101"
    part = edge_features(future_changed).set_index("record_id")
    cols = ["time_gap", "byte_ratio", "first_seen", "outcome"]
    pd.testing.assert_frame_equal(full.loc[: cut - 1, cols], part.loc[: cut - 1, cols])


def test_first_seen_and_time_gap():
    df = toy_records()
    f = edge_features(df)
    pair = f[["DeviceSrc", "DeviceDst"]].apply(lambda r: tuple(sorted(r)), axis=1)
    assert f.groupby(pair)["first_seen"].sum().eq(1).all()
    assert (f["time_gap"] >= 0).all()
    assert (f.loc[f.first_seen == 1, "time_gap"] == 0).all()


def build_toy():
    df = toy_records()
    feats = edge_features(df)
    df = df.set_index("record_id").loc[feats["record_id"]].reset_index()
    packet = np.random.default_rng(1).normal(size=(len(df), 5)).astype(np.float32)
    eng = np.column_stack([np.log1p(feats["time_gap"]), np.log(feats["byte_ratio"])])
    X = record_matrix(feats, packet, StandardScaler().fit(eng))
    return WindowBuilder(feats, X, GraphConfig(max_others=4, pattern_len=6)).build(), feats


def test_window_at_a_cut_sees_only_its_own_split():
    # Device A talks to C just before the train/val cut and to B just after,
    # inside the same one-second slot. The val window must not contain C.
    rec = pd.DataFrame({
        "record_id": [0, 1, 2, 3], "segment_id": "0-0",
        "time_s": [38.2, 40.1, 40.7, 40.9],
        "split": ["train", "train", "val", "val"],
        "DeviceSrc": ["A", "A", "A", "B"], "DeviceDst": ["C", "C", "B", "A"],
        "protocol": "TCP", "tcp.len": 10.0, "tcp.connection.rst": 0.0,
        "tcp.connection.fin": 0.0, "tcp.connection.synack": 0.0,
        "tcp.connection.syn": 0.0, "tcp.flags.ack": 1.0,
        "Attack_label": 0, "Attack_type": "Normal"})
    feats = edge_features(rec)
    eng = np.column_stack([np.log1p(feats["time_gap"]), np.log(feats["byte_ratio"])])
    X = record_matrix(feats, np.zeros((4, 1), np.float32), StandardScaler().fit(eng))
    wb = WindowBuilder(feats, X, GraphConfig(max_others=4, pattern_len=4))
    built = wb.build()
    w = built["windows"]
    a = wb.devices.get_loc("A")
    val_a = w[(w.split == "val") & (w.device_code == a)]
    assert len(val_a) == 1 and val_a["end_slot"].iat[0] == 40
    assert val_a["n_nodes"].iat[0] == 2                      # A and B, never C
    assert val_a["prev_window"].iat[0] == -1                 # train windows are not history
    i = val_a.index[0]
    assert built["activity"]["others_mask"][i].sum() == 1    # only B is visible to GC


def test_history_links_stay_inside_split_and_device():
    built, _ = build_toy()
    w = built["windows"]
    prev = w["prev_window"].to_numpy()
    has = prev >= 0
    assert (w["split"].to_numpy()[prev[has]] == w["split"].to_numpy()[has]).all()
    assert (w["device_code"].to_numpy()[prev[has]] == w["device_code"].to_numpy()[has]).all()
    assert (w["end_slot"].to_numpy()[prev[has]] < w["end_slot"].to_numpy()[has]).all()
    seq = built["seq_idx"]
    assert (seq[:, -1] == np.arange(len(w))).all()


def test_ego_graph_offsets_are_consistent():
    built, _ = build_toy()
    ego, w = built["ego"], built["windows"]
    assert ego["node_ptr"][-1] == len(ego["node_feat"])
    assert ego["edge_ptr"][-1] == ego["edge_index"].shape[1] == len(ego["edge_attr"])
    for i in range(len(w)):
        e = ego["edge_index"][:, ego["edge_ptr"][i]: ego["edge_ptr"][i + 1]]
        assert e.size == 0 or e.max() < w["n_nodes"][i]


def test_module_shapes():
    built, _ = build_toy()
    ego, act, w = built["ego"], built["activity"], built["windows"]
    ids = np.arange(min(16, len(w)))
    lc = LCModule(ego["node_feat"].shape[1], ego["edge_attr"].shape[1], hidden=32)
    h_L = lc(ego_batch(ego, ids), ego_batch(ego, ids))
    assert h_L.shape == (len(ids), 32)
    ce = CEModule(32, seq_len=8)
    seq = h_L[None].repeat(len(ids), 1, 1)[:, :8]
    pad = torch.zeros(seq.shape[:2], dtype=torch.bool)
    assert ce(seq, pad).shape == (len(ids), 32)
    assert ce.masked_loss(seq, pad).ndim == 0
    gc = GCModule(6, hidden=32)
    h_G = gc(torch.from_numpy(act["own"][ids]), torch.from_numpy(act["others"][ids]),
             torch.from_numpy(act["others_mask"][ids]))
    assert h_G.shape == (len(ids), 32) and torch.isfinite(h_G).all()
    h, weights = WeightedFusion(32)([h_L, h_G, h_G])
    assert h.shape == (len(ids), 32)
    assert torch.allclose(weights.sum(1), torch.ones(len(ids)))
