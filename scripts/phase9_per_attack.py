"""Phase 9: attack-wise evaluation on S_test.

    python scripts/phase9_per_attack.py

Classes are read from the data (12 attack types + Normal after Phase 2
excluded the corrupted DDoS_UDP and MITM rows). Main model: protocol-blind
(Phase 6 --stress scores); the all-features model's F1 is shown alongside.
Counts are test *windows*, the unit every model is scored on.
"""

import argparse
import glob
import json
import pickle
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support  # noqa: E402

SMALL = 500          # test windows below this are flagged as statistically unreliable
F1_C, REC_C, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def per_class(files, n_cls):
    out = []
    for f in files:
        z = np.load(f)
        y, pred = z["y_multi"], z["prob"].argmax(1)
        p, r, f1, n = precision_recall_fscore_support(y, pred, labels=range(n_cls), zero_division=0)
        flagged = z["p_sup"] >= float(z["tau_sup"])
        det = np.array([flagged[y == c].mean() if (y == c).any() else np.nan for c in range(n_cls)])
        hits = np.array([((pred == c) & (y == c)).sum() for c in range(n_cls)])
        out.append({"p": p, "r": r, "f1": f1, "n": n, "det": det, "hits": hits})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="data/phase6_stress")
    ap.add_argument("--compare", default="data/phase6")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    res = Path(args.results)

    W = pickle.load(open("data/phase4/windows.pkl", "rb"))["windows"]
    classes = ["Normal"] + sorted(c for c in W["attack_type"].unique() if c != "Normal")
    files = sorted(glob.glob(f"{args.scores}/test_scores_seed*.npz"))
    cmp_files = sorted(glob.glob(f"{args.compare}/test_scores_seed*.npz"))
    main_runs, cmp_runs = per_class(files, len(classes)), per_class(cmp_files, len(classes))

    def ms(runs, key, c):
        v = np.array([r[key][c] for r in runs])
        return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0

    rows = []
    for c, name in enumerate(classes):
        n = int(main_runs[0]["n"][c])
        p, ps = ms(main_runs, "p", c)
        r, rs = ms(main_runs, "r", c)
        f, fs = ms(main_runs, "f1", c)
        d, _ = ms(main_runs, "det", c)
        fa, fas = ms(cmp_runs, "f1", c)
        lo, hi = wilson(int(round(np.mean([x["hits"][c] for x in main_runs]))), n)
        rows.append({
            "Attack type": name, "Test samples": n,
            "Precision": f"{p:.4f} ± {ps:.4f}", "Recall": f"{r:.4f} ± {rs:.4f}",
            "F1": f"{f:.4f} ± {fs:.4f}",
            "Recall 95% CI": f"[{lo:.3f}, {hi:.3f}]",
            "Flagged as any attack": "n/a (benign)" if name == "Normal" else f"{d:.4f}",
            "F1 (all-features model)": f"{fa:.4f} ± {fas:.4f}",
            "Reliability": (f"UNRELIABLE: only {n} test windows" if n < SMALL else "ok"),
            "_f1": f, "_f1s": fs, "_r": r, "_rs": rs})
    table = pd.DataFrame(rows)
    order = [0] + list(table.iloc[1:].sort_values("Test samples", ascending=False).index)
    table = table.loc[order]
    table.drop(columns=[c for c in table if c.startswith("_")]).to_csv(res / "per_attack_results.csv", index=False)

    # plot: attack types only. A dot plot, not bars: the axis is zoomed in,
    # and bars that do not start at zero would exaggerate the differences.
    t = table[table["Attack type"] != "Normal"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 5.6), dpi=150)
    yy = np.arange(len(t))
    for off, key, err, c, lab in [(0.14, "_f1", "_f1s", F1_C, "F1"), (-0.14, "_r", "_rs", REC_C, "Recall")]:
        ax.errorbar(t[key], yy + off, xerr=t[err], fmt="o", color=c, markersize=6,
                    markeredgecolor="white", markeredgewidth=1, ecolor=c, elinewidth=1.2,
                    capsize=2, label=lab)
    labels = [f"{a}  (n={n:,})" for a, n in zip(t["Attack type"], t["Test samples"])]
    ax.set_yticks(yy, labels, fontsize=8.5)
    small = (t["Test samples"] < SMALL).to_numpy()
    for lab, sm in zip(ax.get_yticklabels(), small):
        if sm:
            lab.set_color(MUTED)
            lab.set_fontstyle("italic")
    if small.any():
        ax.axhspan(-0.5, small.sum() - 0.5, color=GRID, alpha=0.45, linewidth=0, zorder=0)
        ax.text(0.505, small.sum() - 0.6, f"Fewer than {SMALL} test windows: statistically unreliable",
                fontsize=7.5, color=MUTED, va="top", fontstyle="italic")
    ax.set_xlim(0.5, 1.01)
    ax.set_ylim(-0.6, len(t) - 0.4)
    ax.set_xlabel("Score on S_test (mean ± std over seeds)")
    ax.set_title("Per-attack F1 and recall, protocol-blind model", fontsize=10, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=2)
    fig.tight_layout()
    fig.savefig(res / "per_attack_f1_recall.png")
    plt.close(fig)

    (res / "run_log_phase9.json").write_text(json.dumps({
        "scores_main": args.scores, "scores_compare": args.compare,
        "seeds_main": len(files), "seeds_compare": len(cmp_files),
        "unit": "test windows", "unreliable_below": SMALL,
        "excluded_classes": {"DDoS_UDP": "fields shifted in source CSV (Phase 1/2)",
                             "MITM": "fields shifted in source CSV (Phase 1/2)"}}, indent=2))
    with pd.option_context("display.width", 260, "display.max_colwidth", 40):
        print(table.drop(columns=[c for c in table if c.startswith("_")]).to_string(index=False))


if __name__ == "__main__":
    main()
