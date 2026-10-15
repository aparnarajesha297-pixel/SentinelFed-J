"""Phase 8: calibration and Trust / Verify / Reject on the held-out test set.

    python scripts/phase8_calibration.py

Uses the protocol-blind scores saved by Phase 6 (--stress run). Calibration
maps are fitted on S_val only and evaluated on S_test. The decision policy
uses the thresholds chosen on S_val in that run, unchanged.
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from sentinelfed.detector import REJECT, TRUST, VERIFY, decide  # noqa: E402

N_BINS = 15
SERIES = {"Supervised branch alone": "#2a78d6", "Unsupervised branch alone": "#eb6834",
          "Dual Head": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def ece(y, p, bins=N_BINS):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return total


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def platt(score_val, y_val, score_test):
    lr = LogisticRegression(C=1e6).fit(score_val[:, None], y_val)
    return lr.predict_proba(score_test[:, None])[:, 1]


def reliability(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    xs, ys, ns = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 20:              # skip bins too small to estimate
            xs.append(p[m].mean())
            ys.append(y[m].mean())
            ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), ns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="data/phase6_stress")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    res = Path(args.results)
    files = sorted(glob.glob(f"{args.scores}/test_scores_seed*.npz"))
    classes = None
    rows, pol_rows, by_type, curves = [], [], [], {}

    for f in files:
        z = np.load(f)
        seed = int(Path(f).stem.split("seed")[1])
        y, yv = z["y"], z["val_y"]
        base = y.mean()
        sup_raw, uns_raw, risk = z["p_sup"], z["d_norm"], z["risk"]
        sup_cal = platt(z["val_p_sup"], yv, sup_raw)
        uns_cal = platt(z["val_d_norm"], yv, uns_raw)
        dec = decide(sup_raw, uns_raw, float(z["tau_sup"]), float(z["tau_unsup"]))
        verify = dec == VERIFY
        entries = [
            ("Supervised branch alone", sup_raw, sup_cal, None),
            ("Unsupervised branch alone", uns_raw, uns_cal, None),
            ("Dual Head", None, risk, verify),
        ]
        for name, raw, cal, ver in entries:
            row = {"Model / Branch": name, "seed": seed,
                   "ECE": ece(y, cal), "Brier": brier(y, cal),
                   "ECE_raw": ece(y, raw) if raw is not None else np.nan,
                   "Brier_raw": brier(y, raw) if raw is not None else np.nan,
                   "Verify %": 100 * ver.mean() if ver is not None else np.nan,
                   "Attack % among Verify": 100 * y[ver].mean() if ver is not None and ver.any() else np.nan,
                   "Overall attack %": 100 * base}
            rows.append(row)
            curves.setdefault(name, []).append(reliability(y, cal))
        for k in (TRUST, VERIFY, REJECT):
            m = dec == k
            pol_rows.append({"seed": seed, "decision": k, "windows": int(m.sum()),
                             "share %": 100 * m.mean(),
                             "attack % in bucket": 100 * y[m].mean() if m.any() else np.nan})
        if classes is None:
            import pickle
            W = pickle.load(open("data/phase4/windows.pkl", "rb"))["windows"]
            classes = ["Normal"] + sorted(c for c in W["attack_type"].unique() if c != "Normal")
        t = pd.crosstab(np.array(classes)[z["y_multi"]], dec)
        t["seed"] = seed
        by_type.append(t)

    sw = pd.DataFrame(rows)
    order = list(SERIES)
    g = sw.groupby("Model / Branch")
    table = pd.DataFrame({"Model / Branch": order})
    for c in ["ECE", "Brier", "ECE_raw", "Brier_raw", "Verify %", "Attack % among Verify"]:
        mean, std = g[c].mean().reindex(order), g[c].std().reindex(order).fillna(0)
        table[c] = ["n/a" if np.isnan(a) else f"{a:.4f} ± {b:.4f}" for a, b in zip(mean, std)]
    table["Overall attack % (test)"] = f"{sw['Overall attack %'].iloc[0]:.2f}"
    table["seeds"] = g.size().reindex(order).values
    table.to_csv(res / "calibration_results.csv", index=False)
    sw.to_csv(res / "calibration_seedwise.csv", index=False)

    pol = pd.DataFrame(pol_rows).groupby("decision")[["windows", "share %", "attack % in bucket"]]
    pol = pol.agg(["mean", "std"]).reindex([TRUST, VERIFY, REJECT])
    pol.to_csv(res / "phase8_policy_test.csv")
    bt = pd.concat(by_type).groupby(level=0).mean().drop(columns="seed").round(1)
    bt = bt.reindex(columns=[TRUST, VERIFY, REJECT]).fillna(0)
    bt.index.name = "attack_type"
    bt.to_csv(res / "phase8_policy_by_type_test.csv")

    # reliability diagram (seed 0 curves, all seeds' ECE in the legend)
    fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=150)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Perfect calibration")
    # Dual Head is drawn first and wider: its curve coincides with the
    # supervised branch, which is drawn dashed on top so both stay visible.
    style = {"Dual Head": dict(linewidth=4.5, zorder=2),
             "Supervised branch alone": dict(linewidth=1.8, linestyle=(0, (4, 2)), zorder=3),
             "Unsupervised branch alone": dict(linewidth=2, zorder=3)}
    for name in ["Dual Head", "Supervised branch alone", "Unsupervised branch alone"]:
        xs, ys, _ = curves[name][0]
        e = g["ECE"].mean()[name]
        label = f"{name} (ECE {e:.4f})"
        if name == "Supervised branch alone":
            label += ", same curve as Dual Head"
        ax.plot(xs, ys, color=SERIES[name], marker="o", markersize=5, markeredgecolor="white",
                markeredgewidth=1, label=label, **style[name])
    ax.text(0.98, 0.03, "Bins with fewer than 20 windows are omitted;\nsupervised scores sit almost "
            "all near 0 or 1.", ha="right", va="bottom", fontsize=7.5, color=MUTED,
            transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted attack probability (calibrated on S_val)")
    ax.set_ylabel("Observed attack rate on S_test")
    ax.set_title("Reliability diagram, protocol-blind model", fontsize=10, color=INK, loc="left")
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(res / "reliability_diagram.png")
    plt.close(fig)

    (res / "run_log_phase8.json").write_text(json.dumps({
        "scores_from": args.scores, "seeds": sorted(sw["seed"].unique().tolist()),
        "ece_bins": N_BINS, "calibration": "Platt scaling fitted on S_val, evaluated on S_test",
        "reliability_bins": "10 equal-width, bins with <20 windows omitted"}, indent=2))
    with pd.option_context("display.width", 250):
        print(table.to_string(index=False))
        print(pol.round(3).to_string())
        print(bt.to_string())


if __name__ == "__main__":
    main()
