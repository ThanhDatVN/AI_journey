"""API chấm điểm giao dịch.

    uvicorn fraud.api:app --port 8000          # MODEL_DIR mặc định: models/current (đổi bằng FRAUD_MODEL_DIR)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from fraud.config import MODEL_DIR, REVIEW_COST_VND
from fraud.model import decide

STATE: dict = {}


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount_vnd: float = Field(gt=0, le=5e9)
    hour: int = Field(ge=0, le=23)
    merchant_category: Literal["grocery", "electronics", "travel", "fashion", "food", "online_games"]
    distance_from_home_km: float = Field(ge=0, le=20_000)
    is_foreign: bool
    n_tx_last_24h: int = Field(ge=0, le=1_000)
    card_age_days: int = Field(ge=0, le=36_500)


class Score(BaseModel):
    fraud_probability: float
    expected_loss_vnd: float
    decision: Literal["approve", "review"]
    model_version: str


def load(model_dir: Path):
    """Nạp mô hình đã đóng gói; từ chối khởi động nếu file bị sửa hoặc phiên bản scikit-learn lệch lúc huấn luyện."""
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if hashlib.sha256((model_dir / "model.joblib").read_bytes()).hexdigest() != meta["model_sha256"]:
        raise RuntimeError("model.joblib không khớp hash — từ chối khởi động")
    if meta.get("sklearn") != sklearn.__version__:                     # pickle không đảm bảo tương thích giữa các phiên bản
        raise RuntimeError(f"mô hình huấn luyện bằng scikit-learn {meta.get('sklearn')} nhưng dịch vụ chạy {sklearn.__version__} — "
                           "huấn luyện lại, hoặc ghim đúng phiên bản bằng `make freeze` trước khi build image")
    return joblib.load(model_dir / "model.joblib"), meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["model"], STATE["meta"] = load(Path(os.environ.get("FRAUD_MODEL_DIR", MODEL_DIR)))
    yield
    STATE.clear()


app = FastAPI(title="Fraud scoring API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": STATE["meta"]["version"]}


@app.post("/v1/score", response_model=list[Score])
def score(transactions: list[Transaction]):
    if not 1 <= len(transactions) <= 1000:
        raise HTTPException(status_code=422, detail="cần 1–1000 giao dịch mỗi request")
    t0 = time.perf_counter()
    frame = pd.DataFrame([t.model_dump() for t in transactions])
    p = STATE["model"].predict_proba(frame)[:, 1]
    review = decide(p, frame.amount_vnd)
    version = STATE["meta"]["version"]
    STATE["last_latency_ms"] = (time.perf_counter() - t0) * 1000
    return [Score(fraud_probability=float(pi), expected_loss_vnd=float(pi * a), decision="review" if r else "approve", model_version=version)
            for pi, a, r in zip(p, frame.amount_vnd, review)]


@app.get("/v1/policy")
def policy():
    return {"rule": "review nếu fraud_probability × amount_vnd > review_cost", "review_cost_vnd": REVIEW_COST_VND}
