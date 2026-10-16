"""Write results/run_environment.json: the environment every run used and,
for each committed results file, the commit that last produced it.

    python scripts/provenance.py

This covers the guide's run-log fields for results produced before the
shared run-log helper (sentinelfed.runlog) existed: code version per file,
data version, hardware and software. Seeds and hyperparameters are in each
phase's own run_log_*.json; split cut points in results/split_cutpoints.csv.
"""

import json
import subprocess
from pathlib import Path

from sentinelfed.runlog import environment


def last_commit(path: Path) -> dict:
    out = subprocess.check_output(["git", "log", "-1", "--format=%H|%cI|%s", "--", str(path)],
                                  text=True).strip()
    if not out:
        return {"commit": "uncommitted"}
    h, date, subject = out.split("|", 2)
    return {"commit": h, "date": date, "subject": subject}


def main() -> None:
    files = sorted(p for p in Path("results").iterdir()
                   if p.is_file() and p.suffix in {".csv", ".json", ".png", ".txt"}
                   and p.name != "run_environment.json")
    env = environment()
    env["note"] = ("All runs used this container; code_version below is the commit at the time this "
                   "file was written. Per-file commits give the code that produced each result.")
    env["results"] = {p.name: last_commit(p) for p in files}
    Path("results/run_environment.json").write_text(json.dumps(env, indent=2))
    print(f"recorded {len(files)} results files")


if __name__ == "__main__":
    main()
