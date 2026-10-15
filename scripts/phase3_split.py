"""Phase 3: Leakage-Aware Temporal Splitting (LATS).

    python scripts/phase3_split.py

Edge-IIoTset has no calendar dates and each class was recorded separately
(Phase 1), so one global time cut would put whole classes in a single split.
Instead every recording (segment_id from Phase 2) is cut on its own clock:
the earliest ~70% of its records go to train, the next ~15% to validation,
the latest ~15% to test. Cut points are timestamps, and rows that share a
cut timestamp all go to the later split, so within a recording every train
row is strictly earlier than every validation row, and validation than test.

Supervised family:   S_train, S_val, S_test (normal + attack)
Unsupervised family: U_train = normal rows of S_train
                     U_val   = normal rows of S_val
                     U_test  = attack rows of S_test

The two families overlap on purpose (U_train is a subset of S_train);
records never cross train/val/test boundaries.

The preprocessor (scaling + encoding) is fit here on S_train only and saved,
so later phases apply it unchanged to validation and test data.
"""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
SPLITS = ["train", "val", "test"]

ID_COLS = ["record_id", "capture_id", "segment_id", "Timestamp", "time_s",
           "DeviceSrc", "DeviceDst"]
LABEL_COLS = ["Attack_label", "Attack_type"]
CATEGORICAL = ["protocol", "http.request.method", "http.request.version"]
# Columns computed from more than the row itself. None of these may ever be a
# model feature; check 4 enforces it.
NON_ROW_LOCAL = {"segment_id", "time_s", "capture_id"}


def cut_points(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seg, g in df.groupby("segment_id", sort=False):
        t = np.sort(g["time_s"].to_numpy())
        c1 = t[int(np.floor(TRAIN_FRAC * len(t)))]
        c2 = t[int(np.floor((TRAIN_FRAC + VAL_FRAC) * len(t)))]
        rows.append({"segment_id": seg, "Attack_type": g["Attack_type"].iloc[0],
                     "records": len(t), "start_s": t[0], "val_from_s": c1,
                     "test_from_s": c2, "end_s": t[-1]})
    return pd.DataFrame(rows)


def assign(df: pd.DataFrame, cuts: pd.DataFrame) -> pd.Series:
    c = df[["segment_id"]].merge(cuts, on="segment_id", how="left")
    t = df["time_s"].to_numpy()
    split = np.where(t < c["val_from_s"].to_numpy(), "train",
                     np.where(t < c["test_from_s"].to_numpy(), "val", "test"))
    return pd.Series(split, index=df.index)


def fmt(s: float) -> str:
    day, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{'+1d ' if day else ''}{int(h):02d}:{int(m):02d}:{s:09.6f}"


def build_preprocessor(features: list[str]) -> ColumnTransformer:
    numeric = [c for c in features if c not in CATEGORICAL]
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
    ])


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed/processed_data.csv")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", default="artifacts")
    args = ap.parse_args()
    out, res, models = Path(args.out), Path(args.results), Path(args.models)
    for p in (out, res, models):
        p.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.processed, low_memory=False,
                     dtype={"segment_id": str, "http.request.method": str,
                            "http.request.version": str})
    df = df.sort_values(["segment_id", "time_s", "record_id"], kind="stable")

    cuts = cut_points(df)
    df["split"] = assign(df, cuts)
    normal = df["Attack_type"] == "Normal"

    files = {
        "S_train": df[df.split == "train"],
        "S_val": df[df.split == "val"],
        "S_test": df[df.split == "test"],
        "U_train": df[(df.split == "train") & normal],
        "U_val": df[(df.split == "val") & normal],
        "U_test": df[(df.split == "test") & ~normal],
    }
    for name, part in files.items():
        part.drop(columns="split").to_csv(out / f"{name}.csv", index=False)

    # --- preprocessing, fit on training data only -------------------------
    features = [c for c in df.columns if c not in ID_COLS + LABEL_COLS + ["split"]]
    pre_s = build_preprocessor(features).fit(files["S_train"][features])
    pre_u = build_preprocessor(features).fit(files["U_train"][features])
    fit_ids = {
        "S": hashlib.sha256(np.sort(files["S_train"]["record_id"].to_numpy()).tobytes()).hexdigest(),
        "U": hashlib.sha256(np.sort(files["U_train"]["record_id"].to_numpy()).tobytes()).hexdigest(),
    }
    joblib.dump({"preprocessor": pre_s, "features": features, "fit_on": "S_train",
                 "fit_rows": len(files["S_train"]), "fit_ids_sha256": fit_ids["S"]},
                models / "preprocess_supervised.joblib")
    joblib.dump({"preprocessor": pre_u, "features": features, "fit_on": "U_train",
                 "fit_rows": len(files["U_train"]), "fit_ids_sha256": fit_ids["U"]},
                models / "preprocess_unsupervised.joblib")

    # --- the four mandatory checks ------------------------------------------
    ids = {k: set(v["record_id"]) for k, v in files.items()}
    per_seg = df.groupby(["segment_id", "split"])["time_s"].agg(["min", "max"]).unstack()
    order_ok = ((per_seg[("max", "train")] < per_seg[("min", "val")]) &
                (per_seg[("max", "val")] < per_seg[("min", "test")]))
    test_after_train = (per_seg[("min", "test")] > per_seg[("max", "train")])

    pairs = [("S_train", "S_val"), ("S_train", "S_test"), ("S_val", "S_test"),
             ("U_train", "U_val"), ("U_train", "U_test"), ("U_val", "U_test"),
             ("U_train", "S_val"), ("U_train", "S_test"), ("U_val", "S_train"),
             ("U_val", "S_test"), ("U_test", "S_train"), ("U_test", "S_val")]
    overlaps = {f"{a}&{b}": len(ids[a] & ids[b]) for a, b in pairs}

    scaler_s = pre_s.named_transformers_["num"]
    scaler_u = pre_u.named_transformers_["num"]
    fit_ok = (scaler_s.n_samples_seen_ == len(files["S_train"]) and
              scaler_u.n_samples_seen_ == len(files["U_train"]))
    leaky_features = sorted(NON_ROW_LOCAL & set(features))

    checks = pd.DataFrame([
        ("1. Earliest S_test timestamp is later than latest S_train timestamp",
         bool(test_after_train.all()),
         f"holds in {int(test_after_train.sum())}/{len(test_after_train)} recordings "
         "(checked per recording: there is no shared calendar across recordings); "
         f"train<val<test strictly in {int(order_ok.sum())}/{len(order_ok)}"),
        ("2. No record appears in more than one split",
         all(v == 0 for v in overlaps.values()),
         "; ".join(f"{k}={v}" for k, v in overlaps.items()) +
         ". U_train is a subset of S_train by design (separate families)"),
        ("3. No scaler or encoder fit on validation or test data",
         fit_ok,
         f"supervised preprocessor fit on S_train only ({int(scaler_s.n_samples_seen_)} rows); "
         f"unsupervised on U_train only ({int(scaler_u.n_samples_seen_)} rows); "
         "fitted row-id hashes stored in artifacts/*.joblib"),
        ("4. No feature built from information later than its record",
         not leaky_features,
         f"{len(features)} model features, all computed from the record's own packet "
         f"fields. Columns that use other records ({', '.join(sorted(NON_ROW_LOCAL))}) "
         "are identifiers for splitting only and are excluded from features"
         + (f"; LEAK: {leaky_features}" if leaky_features else "")),
    ], columns=["check", "passed", "evidence"])
    checks.to_csv(res / "split_checks.csv", index=False)

    counts = pd.DataFrame({k: v["Attack_type"].value_counts() for k, v in files.items()})
    counts = counts.fillna(0).astype(int)
    counts.loc["TOTAL"] = [len(v) for v in files.values()]
    counts.index.name = "Attack_type"
    counts.to_csv(res / "split_audit.csv")

    cut_tbl = cuts.copy()
    for c in ["start_s", "val_from_s", "test_from_s", "end_s"]:
        cut_tbl[c.replace("_s", "")] = cut_tbl[c].map(fmt)
    seg_counts = df.groupby(["segment_id", "split"]).size().unstack().fillna(0).astype(int)
    cut_tbl = cut_tbl.merge(seg_counts[SPLITS].add_prefix("n_"), on="segment_id")
    cut_tbl.to_csv(res / "split_cutpoints.csv", index=False)

    sha = hashlib.sha256(Path(args.processed).read_bytes()).hexdigest()
    log = {"phase": 3, "code_version": git_rev(), "seed": None,
           "data": {"processed_data.csv_sha256": sha, "rows": len(df)},
           "split_rule": f"per recording, time-ordered, {TRAIN_FRAC:.0%}/{VAL_FRAC:.0%}/"
                         f"{1 - TRAIN_FRAC - VAL_FRAC:.0%} of records; ties go to the later split",
           "cut_points_file": "results/split_cutpoints.csv",
           "features": features,
           "hardware": {"machine": platform.machine(), "processor": platform.processor(),
                        "python": platform.python_version()}}
    (res / "run_log_phase3.json").write_text(json.dumps(log, indent=2))

    with pd.option_context("display.width", 220, "display.max_colwidth", 200):
        print(checks.to_string(index=False))
        print()
        print(counts.to_string())
        print()
        print(cut_tbl[["segment_id", "Attack_type", "start", "val_from", "test_from", "end",
                       "n_train", "n_val", "n_test"]].to_string(index=False))


if __name__ == "__main__":
    main()
