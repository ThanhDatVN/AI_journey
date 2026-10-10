# syntax=docker/dockerfile:1
# Build bằng `make docker-build`: huấn luyện → ghim phiên bản (requirements.txt) → build với đúng phiên bản Python.
# Thủ công (sau `make train` và `make freeze`):  docker build --build-arg PYTHON_VERSION=3.11 -t fraud-api:1.0.0 .
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 FRAUD_MODEL_DIR=/app/models/current
WORKDIR /app

# Phụ thuộc trước (cache layer), code và mô hình sau
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY fraud/ fraud/
COPY models/current/ models/current/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "fraud.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
