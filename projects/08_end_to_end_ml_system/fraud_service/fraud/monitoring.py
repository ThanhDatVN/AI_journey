"""Giám sát trôi dữ liệu so với mốc huấn luyện (lưu trong metadata.json).

    python -m fraud.monitoring --current transactions_this_week.csv [--model-dir models/current]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fraud.config import MODEL_DIR, MONITORED
from fraud.features import add_features

PSI_WARN, PSI_ALERT = 0.1, 0.25


def psi(profile: dict, current, eps: float = 1e-4) -> float:
    """PSI giữa dữ liệu hiện tại và mốc huấn luyện (biên bin + tỉ lệ tham chiếu lưu trong metadata)."""
    bins = np.concatenate([[-np.inf], np.asarray(profile["edges"], float), [np.inf]])
    ref = np.clip(np.asarray(profile["proportions"], float), eps, None)
    cur = np.clip(np.histogram(np.asarray(current, float), bins)[0] / len(current), eps, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def drift_report(reference_profile: dict, current: pd.DataFrame) -> pd.DataFrame:
    feats = add_features(current)
    rows = []
    for col in MONITORED:
        value = psi(reference_profile[col], feats[col])
        status = "ALERT" if value > PSI_ALERT else "WARN" if value > PSI_WARN else "ok"
        rows.append({"feature": col, "psi": round(value, 4), "status": status})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True)
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    args = parser.parse_args(argv)
    meta = json.loads((Path(args.model_dir) / "metadata.json").read_text(encoding="utf-8"))
    report = drift_report(meta["reference_profile"], pd.read_csv(args.current))
    print(report.to_string(index=False))
    return 1 if (report.status == "ALERT").any() else 0


if __name__ == "__main__":
    sys.exit(main())
