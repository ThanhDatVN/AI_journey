PYTHON ?= python
# Phiên bản Python của môi trường huấn luyện (vd 3.11) — image Docker dùng đúng phiên bản này.
PY_MINOR := $(shell $(PYTHON) -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
RUNTIME := numpy pandas scipy scikit-learn joblib fastapi uvicorn pydantic

.PHONY: train test serve monitor docker-build docker-run freeze

train:            ## huấn luyện + cổng chất lượng + đóng gói vào models/current
	$(PYTHON) -m fraud.train

test:             ## unit, dữ liệu, mô hình, API, đóng gói, giám sát
	$(PYTHON) -m pytest

serve:            ## chạy API tại http://127.0.0.1:8000 (tài liệu: /docs)
	$(PYTHON) -m uvicorn fraud.api:app --port 8000

monitor:          ## so dữ liệu mới với mốc huấn luyện: make monitor CURRENT=path/to/new.csv
	$(PYTHON) -m fraud.monitoring --current $(CURRENT)

freeze:           ## ghim phiên bản runtime ĐANG dùng để huấn luyện vào requirements.txt (đọc metadata, không cần pip)
	$(PYTHON) -c "import importlib.metadata as m, pathlib, sys; pathlib.Path('requirements.txt').write_text('# Phụ thuộc runtime — ghim đúng phiên bản dùng khi huấn luyện (sinh bởi make freeze).\n' + ''.join(p + '==' + m.version(p) + '\n' for p in sys.argv[1:]), encoding='utf-8')" $(RUNTIME)

docker-build: train freeze   ## mô hình + requirements.txt + phiên bản Python khớp nhau → image
	docker build --build-arg PYTHON_VERSION=$(PY_MINOR) -t fraud-api:1.0.0 .

docker-run:
	docker run --rm -p 8000:8000 fraud-api:1.0.0
