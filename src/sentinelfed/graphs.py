"""Windows, ego graphs and activity patterns (Phase 4).

Unit of analysis: a device-window = one device's traffic over ``w`` one-second
slots ending at a slot where it was active. Windows are built inside one
recording *and* one split, so no window ever straddles a train/val/test cut.

Device identity (IP address) is used only to group records. It is never a
feature: node features describe what a device did, not which device it is.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from .edges import OUTCOMES

NONE_DEVICE = "none"


@dataclass
class GraphConfig:
    slot_seconds: float = 1.0
    window: int = 3          # w, in slots
    hops: int = 1            # k
    max_nodes: int = 64      # cap on ego-graph size (DDoS victims see hundreds of peers)
    pattern_len: int = 16    # P, slots of activity history for GC
    max_others: int = 32     # M, other devices GC compares against
    seq_len: int = 8         # L, windows of history CE sees


INPUT_CLIP = 5.0


def record_matrix(feats: pd.DataFrame, packet_X: np.ndarray, edge_scaler) -> np.ndarray:
    """Model input per record: scaled packet fields + engineered edge features.

    Scaled values are clipped to +/-INPUT_CLIP. Fields that are almost always
    zero (Modbus unit id, DNS retransmissions) otherwise scale to values
    above 200 and dominate every loss and covariance downstream. The clip is
    a fixed constant, not fitted on any split.
    """
    eng = np.column_stack([np.log1p(feats["time_gap"].to_numpy()),
                           np.log(feats["byte_ratio"].to_numpy())])
    eng = edge_scaler.transform(eng)
    packet_X = np.clip(packet_X, -INPUT_CLIP, INPUT_CLIP)
    eng = np.clip(eng, -INPUT_CLIP, INPUT_CLIP)
    oc = pd.Categorical(feats["outcome"], categories=OUTCOMES).codes
    onehot = np.eye(len(OUTCOMES), dtype=np.float32)[oc]
    return np.hstack([packet_X, eng, feats[["first_seen"]].to_numpy(), onehot]).astype(np.float32)


def _aggregate(keys: np.ndarray, X: np.ndarray, n_keys: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean of X rows per key, plus counts."""
    M = sparse.csr_matrix((np.ones(len(keys), np.float32), (keys, np.arange(len(keys)))),
                          shape=(n_keys, len(keys)))
    counts = np.asarray(M.sum(1)).ravel()
    sums = M @ X
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts[:, None] > 0, sums / counts[:, None], 0.0)
    return mean.astype(np.float32), counts


class WindowBuilder:
    """Builds every artefact of the windowing stage from per-record data."""

    def __init__(self, feats: pd.DataFrame, X: np.ndarray, cfg: GraphConfig):
        self.cfg = cfg
        f = feats.reset_index(drop=True)
        self.E = X.shape[1]
        slot = np.floor(f["time_s"].to_numpy() / cfg.slot_seconds).astype(np.int64)
        group = f["segment_id"].astype(str) + "|" + f["split"]
        self.group_codes, self.group_names = pd.factorize(group)
        devices = pd.Index(pd.unique(np.concatenate([f["DeviceSrc"], f["DeviceDst"]])))
        src = devices.get_indexer(f["DeviceSrc"])
        dst = devices.get_indexer(f["DeviceDst"])
        none = devices.get_loc(NONE_DEVICE) if NONE_DEVICE in devices else -1
        self.devices = devices

        # ---- device-slot table: node features --------------------------------
        def key(g, d, s):
            return pd.MultiIndex.from_arrays([g, d, s])

        rec = np.arange(len(f))
        out_mask, in_mask = src != none, dst != none
        ds_keys = key(np.concatenate([self.group_codes[out_mask], self.group_codes[in_mask]]),
                      np.concatenate([src[out_mask], dst[in_mask]]),
                      np.concatenate([slot[out_mask], slot[in_mask]]))
        ds_codes, ds_index = pd.factorize(ds_keys)
        n_ds = len(ds_index)
        n_out = out_mask.sum()
        out_codes, in_codes = ds_codes[:n_out], ds_codes[n_out:]
        mean_out, cnt_out = _aggregate(out_codes, X[rec[out_mask]], n_ds)
        mean_in, cnt_in = _aggregate(in_codes, X[rec[in_mask]], n_ds)

        # distinct real peers per device-slot
        peer = np.concatenate([np.where(dst[out_mask] == none, -1, dst[out_mask]),
                               np.where(src[in_mask] == none, -1, src[in_mask])])
        pp = pd.DataFrame({"k": ds_codes, "p": peer})
        n_peers = pp[pp.p >= 0].drop_duplicates().groupby("k").size().reindex(
            range(n_ds), fill_value=0).to_numpy()
        self.node_feat = np.hstack([
            mean_out, mean_in,
            np.log1p(np.column_stack([cnt_out, cnt_in, n_peers])).astype(np.float32),
        ])
        self.F = self.node_feat.shape[1]
        self.ds = pd.DataFrame({
            "group": ds_index.get_level_values(0), "device": ds_index.get_level_values(1),
            "slot": ds_index.get_level_values(2)})

        # label of a device-slot: any attack record touching it
        lab = f["Attack_label"].to_numpy()
        ds_lab = np.zeros(n_ds, np.int8)
        np.maximum.at(ds_lab, ds_codes, np.concatenate([lab[out_mask], lab[in_mask]]).astype(np.int8))
        types = np.concatenate([f["Attack_type"].to_numpy()[out_mask], f["Attack_type"].to_numpy()[in_mask]])
        self.ds["label"] = ds_lab
        # Each recording holds one class, so the first type seen is the type.
        first = pd.DataFrame({"k": ds_codes, "t": types}).drop_duplicates("k").set_index("k")["t"]
        self.ds["attack_type"] = first.reindex(range(n_ds)).to_numpy()

        # ---- pair-slot table: edge attributes ---------------------------------
        both = (src != none) & (dst != none)
        ps_keys = key(self.group_codes[both], slot[both], src[both] * len(devices) + dst[both])
        ps_codes, ps_index = pd.factorize(ps_keys)
        ps_mean, ps_cnt = _aggregate(ps_codes, X[rec[both]], len(ps_index))
        pair = ps_index.get_level_values(2).to_numpy()
        self.ps = pd.DataFrame({"group": ps_index.get_level_values(0),
                                "slot": ps_index.get_level_values(1),
                                "u": pair // len(devices), "v": pair % len(devices),
                                "count": ps_cnt})
        self.ps_attr = ps_mean

    # -------------------------------------------------------------------------
    def build(self) -> dict:
        cfg = self.cfg
        ds = self.ds.sort_values(["group", "device", "slot"], kind="stable")
        order = ds.index.to_numpy()
        ds = ds.reset_index(drop=True)
        node_feat = self.node_feat[order]

        ps = self.ps.sort_values(["group", "slot"], kind="stable")
        ps_order = ps.index.to_numpy()
        ps = ps.reset_index(drop=True)
        ps_attr = self.ps_attr[ps_order]
        ps_logcnt = np.log1p(ps["count"].to_numpy()).astype(np.float32)[:, None]

        n = len(ds)
        prev_idx = np.full(n, -1, np.int64)
        seq_idx = np.full((n, cfg.seq_len), -1, np.int64)
        center_feat = np.zeros((n, self.F), np.float32)
        node_ptr, edge_ptr = [0], [0]
        g_nodes, g_edges, g_eattr = [], [], []
        own_pat = np.zeros((n, cfg.pattern_len), np.uint8)
        oth_pat = np.zeros((n, cfg.max_others, cfg.pattern_len), np.uint8)
        oth_mask = np.zeros((n, cfg.max_others), bool)

        ps_group = ps.groupby("group", sort=False).indices

        for gcode, rows in ds.groupby("group", sort=False).indices.items():
            rows = np.sort(rows)
            g_ds = ds.iloc[rows]
            dev_arr, slot_arr = g_ds["device"].to_numpy(), g_ds["slot"].to_numpy()
            # per-device slot arrays for fast range lookups
            per_dev = {d: rows[dev_arr == d] for d in np.unique(dev_arr)}
            per_dev_slots = {d: ds["slot"].to_numpy()[r] for d, r in per_dev.items()}

            prow = np.sort(ps_group.get(gcode, np.array([], np.int64)))
            p_slot = ps["slot"].to_numpy()[prow]
            p_u, p_v = ps["u"].to_numpy()[prow], ps["v"].to_numpy()[prow]
            p_cnt = ps["count"].to_numpy()[prow]

            # activity matrix for GC: device x slot, and a per-slot cache of the
            # most active devices over the trailing P slots
            s0, s1 = slot_arr.min(), slot_arr.max()
            udev = np.unique(dev_arr)
            dev_pos = {d: i for i, d in enumerate(udev)}
            A = sparse.csc_matrix((np.ones(len(rows), np.uint8),
                                   ([dev_pos[d] for d in dev_arr], slot_arr - s0)),
                                  shape=(len(udev), s1 - s0 + 1))
            top_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

            def node_mean(d: int, lo: int, hi: int) -> np.ndarray:
                sl = per_dev_slots[d]
                a, b = np.searchsorted(sl, lo), np.searchsorted(sl, hi, side="right")
                return node_feat[per_dev[d][a:b]].mean(0)

            for d, drows in per_dev.items():
                sl = per_dev_slots[d]
                for j, i in enumerate(drows):
                    # ego arrays are appended in loop order; it must be window order
                    assert i == len(node_ptr) - 1, "window order mismatch"
                    t = sl[j]
                    lo = t - cfg.window + 1
                    # CE history and LC's previous window (same device, earlier)
                    prev_idx[i] = drows[j - 1] if j > 0 else -1
                    hist = drows[max(0, j - cfg.seq_len + 1): j + 1]
                    seq_idx[i, cfg.seq_len - len(hist):] = hist

                    # ---- ego graph: BFS over this window's edges ------------
                    a, b = np.searchsorted(p_slot, lo), np.searchsorted(p_slot, t, side="right")
                    wu, wv, wc = p_u[a:b], p_v[a:b], p_cnt[a:b]
                    wrows = prow[a:b]
                    nodes = [d]
                    frontier = {d}
                    seen = {d}
                    for _ in range(cfg.hops):
                        touch = np.isin(wu, list(frontier)) | np.isin(wv, list(frontier))
                        cand = np.concatenate([wu[touch], wv[touch]])
                        weight = np.concatenate([wc[touch], wc[touch]])
                        if not len(cand):
                            break
                        uc, inv = np.unique(cand, return_inverse=True)
                        tot = np.bincount(inv, weights=weight)
                        fresh = ~np.isin(uc, list(seen))
                        uc, tot = uc[fresh], tot[fresh]
                        room = cfg.max_nodes - len(nodes)
                        new = uc[np.argsort(-tot, kind="stable")][:room].tolist()
                        nodes += new
                        seen.update(new)
                        frontier = set(new)
                        if room <= 0 or not new:
                            break
                    local = {x: k for k, x in enumerate(nodes)}
                    keep = np.isin(wu, nodes) & np.isin(wv, nodes)
                    eu, ev = wu[keep], wv[keep]
                    ew, er = wc[keep], wrows[keep]
                    if len(eu):
                        ekey = eu * len(self.devices) + ev
                        uk, inv = np.unique(ekey, return_inverse=True)
                        wsum = np.bincount(inv, weights=ew)
                        attr = np.zeros((len(uk), self.E), np.float64)
                        np.add.at(attr, inv, ps_attr[er] * ew[:, None])
                        attr /= wsum[:, None]
                        ei = np.stack([[local[x] for x in uk // len(self.devices)],
                                       [local[x] for x in uk % len(self.devices)]])
                        eattr = np.hstack([attr.astype(np.float32),
                                           np.log1p(wsum).astype(np.float32)[:, None]])
                    else:
                        ei = np.zeros((2, 0), np.int64)
                        eattr = np.zeros((0, self.E + 1), np.float32)
                    nf = np.stack([node_mean(x, lo, t) for x in nodes])
                    center_feat[i] = nf[0]
                    g_nodes.append(nf)
                    g_edges.append(ei)
                    g_eattr.append(eattr)
                    node_ptr.append(node_ptr[-1] + len(nodes))
                    edge_ptr.append(edge_ptr[-1] + ei.shape[1])

                    # ---- activity patterns for GC ----------------------------
                    c_lo = t - cfg.pattern_len + 1
                    if t not in top_cache:
                        c0, c1 = max(c_lo - s0, 0), t - s0 + 1
                        block = A[:, c0:c1]
                        active = np.asarray(block.sum(1)).ravel()
                        cand = np.flatnonzero(active)
                        cand = cand[np.argsort(-active[cand], kind="stable")][: cfg.max_others + 1]
                        dense = np.zeros((len(cand), cfg.pattern_len), np.uint8)
                        if len(cand):
                            sub = block[cand].toarray()
                            dense[:, cfg.pattern_len - sub.shape[1]:] = sub
                        top_cache[t] = (udev[cand], dense)
                    cdev, cpat = top_cache[t]
                    own = cpat[cdev == d]
                    if len(own):
                        own_pat[i] = own[0]
                    else:   # device not among the top: compute its own row
                        r = A[dev_pos[d], max(c_lo - s0, 0): t - s0 + 1].toarray().ravel()
                        own_pat[i, cfg.pattern_len - len(r):] = r
                    others = cpat[cdev != d][: cfg.max_others]
                    oth_pat[i, : len(others)] = others
                    oth_mask[i, : len(others)] = True

        node_ptr = np.asarray(node_ptr, np.int64)
        windows = pd.DataFrame({
            "window_id": np.arange(n),
            "group": [self.group_names[g] for g in ds["group"]],
            "device_code": ds["device"].to_numpy(),
            "end_slot": ds["slot"].to_numpy(),
            "label": ds["label"].to_numpy(),
            "attack_type": ds["attack_type"].to_numpy(),
            "prev_window": prev_idx,
            "n_nodes": np.diff(node_ptr),
            "n_edges": np.diff(np.asarray(edge_ptr)),
        })
        windows[["segment_id", "split"]] = windows["group"].str.split("|", expand=True)
        windows = windows.drop(columns="group")
        return {
            "windows": windows,
            "seq_idx": seq_idx,
            "center_feat": center_feat,
            "ego": {"node_feat": np.concatenate(g_nodes).astype(np.float32),
                    "edge_index": np.concatenate(g_edges, axis=1).astype(np.int64),
                    "edge_attr": np.concatenate(g_eattr).astype(np.float32),
                    "node_ptr": node_ptr, "edge_ptr": np.asarray(edge_ptr, np.int64)},
            "activity": {"own": own_pat, "others": oth_pat, "others_mask": oth_mask},
            "devices": list(self.devices),
            "config": cfg.__dict__.copy(),
        }
