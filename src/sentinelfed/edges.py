"""Per-record edge features (Phase 4, first row of the pipeline table).

Every feature here uses only the record itself and records earlier in the
same recording, so nothing depends on the future. Values computed for a
validation or test record may use training records from earlier in the same
recording; that is past information and is allowed.
"""

import numpy as np
import pandas as pd

OUTCOMES = ["none", "syn", "synack", "data", "finish", "reset"]


def outcome(df: pd.DataFrame) -> pd.Series:
    """Connection outcome from TCP flags on this record.

    "No reply" is not derived: it would need records from the future.
    """
    conds = [
        df["tcp.connection.rst"] > 0,
        df["tcp.connection.fin"] > 0,
        df["tcp.connection.synack"] > 0,
        (df["tcp.connection.syn"] > 0) & (df["tcp.flags.ack"] == 0),
        df["tcp.len"] > 0,
    ]
    return pd.Series(np.select(conds, ["reset", "finish", "synack", "syn", "data"], "none"),
                     index=df.index)


def edge_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute time_gap, byte_ratio, first_seen and outcome for every record.

    ``df`` must hold whole recordings (all splits of each segment_id), so the
    history before a validation or test record is available.
    """
    df = df.sort_values(["segment_id", "time_s", "record_id"], kind="stable")
    src, dst = df["DeviceSrc"].to_numpy(), df["DeviceDst"].to_numpy()
    forward = src <= dst                     # direction inside the unordered pair
    lo, hi = np.where(forward, src, dst), np.where(forward, dst, src)
    pair = df["segment_id"].to_numpy().astype(object) + "|" + lo + "|" + hi
    pair = pd.Series(pair, index=df.index)

    prev = df.groupby(pair, sort=False)["time_s"].shift()
    first_seen = prev.isna()
    time_gap = (df["time_s"] - prev).fillna(0.0)

    # Payload bytes each way, accumulated per pair up to and including this record.
    nbytes = df["tcp.len"].to_numpy(dtype=np.float64)
    fwd = pd.Series(np.where(forward, nbytes, 0.0), index=df.index).groupby(pair, sort=False).cumsum()
    bwd = pd.Series(np.where(forward, 0.0, nbytes), index=df.index).groupby(pair, sort=False).cumsum()
    sent = np.where(forward, fwd, bwd)
    received = np.where(forward, bwd, fwd)
    byte_ratio = (sent + 1.0) / (received + 1.0)   # +1 guards against dividing by zero

    out = df[["record_id", "split", "segment_id", "time_s", "DeviceSrc", "DeviceDst",
              "protocol", "Attack_label", "Attack_type"]].copy()
    out["time_gap"] = time_gap.to_numpy()
    out["byte_ratio"] = byte_ratio
    out["first_seen"] = first_seen.astype(np.int8).to_numpy()
    out["outcome"] = outcome(df).to_numpy()
    return out
