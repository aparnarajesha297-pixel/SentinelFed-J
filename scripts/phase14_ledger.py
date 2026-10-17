"""Phase 14: hash-chained ledger vs a conventional database log.

    python scripts/phase14_ledger.py

Every Trust / Verify / Reject decision on the S_test windows (protocol-blind
model, seed 0, Phase 6 scores and thresholds) is written as one record:
window id, timestamp, decision, calibrated risk, both branch scores and the
model version. Three ways of keeping it:

  no ledger        decisions are computed and discarded
  database log     one SQLite row per decision, committed per record
  hash-chained     same SQLite table plus prev_hash and hash, where
                   hash = SHA-256(prev_hash || canonical JSON of the record)

Tamper test: one historical record is altered directly in each store; the
chain is then verified. The database log has nothing to verify against.
"""

import argparse
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sentinelfed.detector import decide
from sentinelfed.runlog import environment

GENESIS = "0" * 64


def records(scores_file, limit=None):
    z = np.load(scores_file)
    thr = json.loads(str(z["thresholds"]))
    dec = decide(z["p_sup"], z["d_norm"], float(z["tau_sup"]), float(z["tau_unsup"]))
    n = len(dec) if limit is None else min(limit, len(dec))
    base = 1_760_000_000.0
    return [{"window_id": int(z["window_id"][i]), "ts": round(base + i * 0.001, 3), "decision": str(dec[i]),
             "risk": round(float(z["risk"][i]), 6), "p_supervised": round(float(z["p_sup"][i]), 6),
             "d_norm": round(float(z["d_norm"][i]), 6), "model": "sentinelfed-j/protocol-blind/seed0"}
            for i in range(n)], len(thr)


def canonical(rec):
    return json.dumps(rec, sort_keys=True, separators=(",", ":"))


def link(prev, rec):
    return hashlib.sha256((prev + canonical(rec)).encode()).hexdigest()


def make_db(path, chained):
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    extra = ", prev_hash TEXT, hash TEXT" if chained else ""
    con.execute(f"CREATE TABLE log (seq INTEGER PRIMARY KEY, record TEXT{extra})")
    return con


def write(con, recs, chained):
    lat = np.empty(len(recs))
    prev = GENESIS
    t_all = time.perf_counter()
    for i, r in enumerate(recs):
        t = time.perf_counter()
        body = canonical(r)
        if chained:
            h = link(prev, r)
            con.execute("INSERT INTO log VALUES (?,?,?,?)", (i, body, prev, h))
            prev = h
        else:
            con.execute("INSERT INTO log VALUES (?,?)", (i, body))
        con.commit()
        lat[i] = time.perf_counter() - t
    return lat, time.perf_counter() - t_all


def verify(con):
    """Recompute the chain; return the first broken seq or None."""
    prev = GENESIS
    for seq, body, p, h in con.execute("SELECT seq, record, prev_hash, hash FROM log ORDER BY seq"):
        if p != prev or link(prev, json.loads(body)) != h:
            return seq
        prev = h
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="data/phase6_protocol_blind/test_scores_seed0.npz")
    ap.add_argument("--results", default="results")
    ap.add_argument("--store", default="data/phase14")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    res, store = Path(args.results), Path(args.store)
    store.mkdir(parents=True, exist_ok=True)
    recs, _ = records(args.scores, args.limit)
    n = len(recs)

    # no ledger: decisions are produced (serialised) and discarded
    t = time.perf_counter()
    lat0 = np.empty(n)
    for i, r in enumerate(recs):
        t1 = time.perf_counter()
        canonical(r)
        lat0[i] = time.perf_counter() - t1
    total0 = time.perf_counter() - t

    rows = [{"Logging method": "No ledger", "Latency / record (ms)": 1000 * lat0.mean(),
             "P99 latency (ms)": 1000 * np.percentile(lat0, 99), "Storage / record (bytes)": 0.0,
             "Records / s": n / total0, "Tamper detected?": "n/a"}]
    tamper = {}
    for name, chained in [("Conventional database log", False), ("Hash-chained ledger", True)]:
        path = store / ("ledger.sqlite" if chained else "dblog.sqlite")
        con = make_db(path, chained)
        lat, total = write(con, recs, chained)
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.commit()
        size = path.stat().st_size + sum(p.stat().st_size for p in path.parent.glob(path.name + "-*"))
        # tamper: flip the decision of one historical record directly in the store
        victim = n // 3
        body = json.loads(con.execute("SELECT record FROM log WHERE seq=?", (victim,)).fetchone()[0])
        original = body["decision"]
        body["decision"] = "Trust" if original != "Trust" else "Reject"
        con.execute("UPDATE log SET record=? WHERE seq=?", (canonical(body), victim))
        con.commit()
        if chained:
            t = time.perf_counter()
            broken = verify(con)
            vsec = time.perf_counter() - t
            detected = broken == victim
            tamper = {"tampered_seq": victim, "original_decision": original, "new_decision": body["decision"],
                      "chain_breaks_at": broken, "detected": detected, "verify_seconds_full_chain": vsec}
            verdict = f"Yes (chain breaks at record {broken}, the altered one)" if detected else "NO"
        else:
            verdict = "No (nothing to check the altered row against)"
        con.close()
        rows.append({"Logging method": name, "Latency / record (ms)": 1000 * lat.mean(),
                     "P99 latency (ms)": 1000 * np.percentile(lat, 99), "Storage / record (bytes)": size / n,
                     "Records / s": n / total, "Tamper detected?": verdict})

    t = pd.DataFrame(rows)
    t.to_csv(res / "ledger_results.csv", index=False)
    (res / "ledger_tamper_test.json").write_text(json.dumps(tamper, indent=2, default=str))
    (res / "run_log_phase14.json").write_text(json.dumps({
        **environment(), "records": n, "store": "SQLite (WAL), commit per record",
        "hash": "SHA-256(prev_hash || canonical JSON)", "tamper": tamper}, indent=2, default=str))
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print(t.round(4).to_string(index=False))
        print(json.dumps(tamper, indent=2, default=str))


if __name__ == "__main__":
    main()
