"""Huấn luyện + cổng chất lượng + đóng gói.

    python -m fraud.train [--data PATH] [--out models/current] [--min-pr-auc 0.06]

Quy trình: kiểm tra dữ liệu → train/validation/test → huấn luyện trên train, đánh giá validation (cổng chất lượng)
→ huấn luyện lại trên train + validation → đánh giá test một lần → lưu model.joblib + metadata.json.
Thoát với mã 1 nếu dữ liệu không hợp lệ hoặc không qua cổng.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from fraud import __version__
from fraud.config import DEFAULT_DATA, MODEL_DIR, MONITORED, TARGET
from fraud.data import load_transactions, split, validate
from fraud.features import add_features
from fraud.model import baseline_policies, build_model, decide, total_cost


def evaluate(model, df: pd.DataFrame) -> dict:
    p = model.predict_proba(df)[:, 1]
    costs = {name: total_cost(df, review) for name, review in baseline_policies(df).items()}
    model_review = decide(p, df.amount_vnd)
    return {
        "pr_auc": float(average_precision_score(df[TARGET], p)),
        "roc_auc": float(roc_auc_score(df[TARGET], p)),
        "brier": float(brier_score_loss(df[TARGET], p)),
        "mean_pred": float(p.mean()),
        "fraud_rate": float(df[TARGET].mean()),
        "review_rate": float(model_review.mean()),
        "cost_model_vnd": total_cost(df, model_review),
        "cost_baselines_vnd": costs,
    }


def reference_profile(df: pd.DataFrame) -> dict:
    """Mốc so sánh cho PSI (fraud.monitoring): biên các bin thập phân vị và tỉ lệ dữ liệu huấn luyện trong mỗi bin."""
    feats = add_features(df)
    profile = {}
    for col in MONITORED:
        inner = np.unique(np.quantile(feats[col], np.linspace(0, 1, 11))[1:-1])
        counts = np.histogram(feats[col], np.concatenate([[-np.inf], inner, [np.inf]]))[0]
        profile[col] = {"edges": inner.tolist(), "proportions": (counts / counts.sum()).tolist()}
    return profile


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(MODEL_DIR))
    parser.add_argument("--min-pr-auc", type=float, default=0.06)
    args = parser.parse_args(argv)

    df = load_transactions(args.data)
    issues = validate(df)
    if issues:
        print("❌ dữ liệu không hợp lệ:", issues)
        return 1
    train, valid, test = split(df)

    candidate = build_model().fit(train, train[TARGET])
    val = evaluate(candidate, valid)
    approve_all = val["cost_baselines_vnd"]["duyệt tất cả"]
    gate = {f"PR-AUC validation ≥ {args.min_pr_auc}": val["pr_auc"] >= args.min_pr_auc,
            "chi phí < 'duyệt tất cả'": val["cost_model_vnd"] < approve_all,
            "xác suất TB lệch < 50% so với tỉ lệ thật": abs(val["mean_pred"] / val["fraud_rate"] - 1) < 0.5}
    for name, ok in gate.items():
        print(f"{'✅' if ok else '❌'} {name}")
    if not all(gate.values()):
        print("❌ không qua cổng chất lượng — không đóng gói")
        return 1

    development = pd.concat([train, valid])
    final = build_model().fit(development, development[TARGET])
    test_metrics = evaluate(final, test)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out / "model.joblib")
    metadata = {
        "name": "fraud-scoring", "version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256((out / "model.joblib").read_bytes()).hexdigest(),
        "sklearn": sklearn.__version__, "python": sys.version.split()[0],
        "validation": val, "test": test_metrics, "reference_profile": reference_profile(development),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ đã đóng gói fraud-scoring {__version__} → {out}")
    print(f"   test: PR-AUC {test_metrics['pr_auc']:.3f} | chi phí {test_metrics['cost_model_vnd'] / 1e6:.1f} tr "
          f"(duyệt tất cả {test_metrics['cost_baselines_vnd']['duyệt tất cả'] / 1e6:.1f} tr) | kiểm duyệt {test_metrics['review_rate']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
