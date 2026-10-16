#!/usr/bin/env bash
# Phase 10 sweep. Safe to rerun: finished (study, setting, seed) runs are
# skipped, and each window/hop Phase 4 build is deleted once its runs are done.
set -euo pipefail
cd "$(dirname "$0")/.."
P=scripts/phase10_sensitivity.py
BASE_BUILD=$(python3 -c "import json;print(json.load(open('results/run_log_phase4.json'))['seconds'])")

done_count() {  # study setting -> number of finished seeds
  python3 - "$1" "$2" <<'EOF'
import sys, pandas as pd, os
p = "results/phase10_runs.csv"
if not os.path.exists(p): print(0); sys.exit()
r = pd.read_csv(p)
print(((r.study == sys.argv[1]) & (r.setting.astype(str) == sys.argv[2])).sum())
EOF
}

python3 $P --study baseline --build-seconds "$BASE_BUILD"
python3 $P --study history
python3 $P --study split
python3 $P --study imbalance
python3 $P --study classifier

sweep() {  # study flag values...
  local study=$1 flag=$2; shift 2
  for v in "$@"; do
    local prefix=${study:0:1}; [ "$study" = hop ] && prefix=k
    local setting="${prefix}${v}" dir="data/phase4_${prefix}${v}"
    if [ "$(done_count "$study" "$setting")" -ge 3 ]; then rm -rf "$dir"; continue; fi
    local secs=nan
    if [ ! -f "$dir/windows.pkl" ] || [ ! -f "$dir/build_seconds.txt" ]; then
      local t0=$(date +%s)
      python3 scripts/phase4_build.py "$flag" "$v" --out "$dir" --results "$dir/build_results" > "$dir.build.log" 2>&1
      echo $(( $(date +%s) - t0 )) > "$dir/build_seconds.txt"
    fi
    secs=$(cat "$dir/build_seconds.txt")
    python3 $P --study "$study" --setting "$setting" --phase4 "$dir" --build-seconds "$secs"
    rm -rf "$dir"
  done
}
sweep window --window 2 4 5 6
sweep hop --hops 2 3 4

python3 $P --report
echo "phase10 sweep complete"
