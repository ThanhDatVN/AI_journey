"""Cấu hình nghiệp vụ và đặc trưng — một nguồn sự thật cho huấn luyện, phục vụ và giám sát."""

from __future__ import annotations

import os
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("FRAUD_MODEL_DIR", SERVICE_ROOT / "models" / "current"))
DEFAULT_DATA = SERVICE_ROOT.parents[2] / "data" / "sample" / "fraud_transactions.csv"

REVIEW_COST_VND = 15_000          # chi phí một lần kiểm duyệt thủ công (nhân sự + trải nghiệm khách)
TARGET = "is_fraud"
RAW_COLUMNS = ["amount_vnd", "hour", "merchant_category", "distance_from_home_km", "is_foreign", "n_tx_last_24h", "card_age_days"]
NUMERIC = ["log_amount", "hour", "is_night", "distance_from_home_km", "n_tx_last_24h", "card_age_days"]
CATEGORICAL = ["merchant_category", "is_foreign"]
MERCHANT_CATEGORIES = ["grocery", "electronics", "travel", "fashion", "food", "online_games"]
MONITORED = ["log_amount", "hour", "distance_from_home_km", "n_tx_last_24h", "card_age_days"]
