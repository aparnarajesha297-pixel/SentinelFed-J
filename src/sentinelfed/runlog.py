"""Run-log fields the guide asks for: code version, data version, hardware
and software versions. Seeds, split cut points and hyperparameters are added
by each script (cut points live in results/split_cutpoints.csv)."""

import json
import os
import platform
import subprocess
from pathlib import Path


def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def environment() -> dict:
    import numpy
    import sklearn
    import torch
    import torch_geometric

    phase3 = Path("results/run_log_phase3.json")
    data_hash = json.loads(phase3.read_text())["data"] if phase3.exists() else "unknown"
    return {
        "code_version": _git("rev-parse", "HEAD"),
        "code_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "data_version": {"raw_csv_sha256": "1d3ef6c7cc22784a528a117f1158f5cad750273679437e18a39c360ee6b79fd7",
                         "processed": data_hash},
        "split_cut_points": "results/split_cutpoints.csv",
        "hardware": {"machine": platform.machine(), "processor": platform.processor() or "unknown",
                     "cpu_count": os.cpu_count(), "gpu": torch.cuda.is_available()},
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "torch_geometric": torch_geometric.__version__, "sklearn": sklearn.__version__,
                     "numpy": numpy.__version__},
    }
