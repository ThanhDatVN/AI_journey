# Chương 25 · Time Series

> Dữ liệu theo thời gian phá vỡ giả định "mẫu độc lập" của hầu hết các chương trước: thứ tự quan trọng, tương lai không được rò vào quá khứ, và một holdout ngẫu nhiên là vô nghĩa. Chương đi từ phân rã và tính dừng, qua baseline và làm trơn hàm mũ, họ ARIMA, tới dự báo bằng feature + backtesting — mọi phương pháp cốt lõi đều cài đặt from scratch trên cùng một bộ dữ liệu doanh số. Không có trong khoá AIVN gốc — bổ sung mới theo template "mọi model".

## 🎯 Mục tiêu

- Thành phần xu hướng/mùa vụ/nhiễu; phân rã cổ điển và STL; ACF; tính dừng, sai phân, ADF; tương quan giả.
- Metric dự báo (MAE, RMSE, MAPE, **MASE**); baseline (naive, seasonal naive, drift); SES, Holt, Holt–Winters from scratch; khoảng dự báo.
- AR/MA, ACF/PACF, AR bằng OLS + AIC, dự báo đệ quy, SARIMA, **ARX** với khuyến mãi/lễ biết trước, Ljung–Box.
- Dự báo bằng feature (lịch, Fourier, sự kiện), **rò rỉ** và tính khả dụng của feature, vì sao CV ngẫu nhiên đánh lừa, cây không ngoại suy (hybrid), **rolling-origin backtest**, khoảng dự báo từ phần dư ngoài mẫu.

## 📋 Yêu cầu trước

[Chương 06](../06_linear_regression/README.md) (hồi quy, feature), [Chương 14](../14_model_evaluation/README.md) (metric hồi quy), [Chương 15](../15_cross_validation/README.md) (TimeSeriesSplit), [Chương 12](../12_gradient_boosting/README.md).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Thành phần chuỗi thời gian và tính dừng | 📖 Bài học | [View](01_components_and_stationarity.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/25_time_series/01_components_and_stationarity.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/25_time_series/01_components_and_stationarity.ipynb) |
| 02 | Bài 02 — Baseline và làm trơn hàm mũ | 📖 Bài học | [View](02_baselines_and_smoothing.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/25_time_series/02_baselines_and_smoothing.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/25_time_series/02_baselines_and_smoothing.ipynb) |
| 03 | Bài 03 — Họ ARIMA | 📖 Bài học | [View](03_arima_family.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/25_time_series/03_arima_family.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/25_time_series/03_arima_family.ipynb) |
| 04 | Bài 04 — Dự báo bằng feature và backtesting | 📖 Bài học | [View](04_feature_based_forecasting_and_backtesting.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/25_time_series/04_feature_based_forecasting_and_backtesting.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/25_time_series/04_feature_based_forecasting_and_backtesting.ipynb) |
| 90 | Bài tập — Chương 25: Time Series (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/25_time_series/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/25_time_series/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 25 · Time Series | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/25_time_series/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/25_time_series/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧰 Thư viện tuỳ chọn

`statsmodels` — `pip install -e ".[stats]"`: dùng để đối chiếu (STL, ADF, ACF/PACF, `AutoReg`, Ljung–Box); notebook tự bỏ qua nếu chưa cài. Đoạn `SARIMAX` ở bài 03 là **code tham khảo** (không chạy trong notebook) vì một số môi trường Windows chặn phần biên dịch `statsmodels.tsa.statespace`.

## 🧩 Code tham chiếu

Bài 02: `holt_winters` (SES/Holt/Holt–Winters dùng chung một hàm), `fit_smoothing`. Bài 03: `pacf`, `fit_ar`, `ljung_box`. Bài 04: `calendar_features`, `backtest`. Bài tập: `make_supervised` (feature direct không rò rỉ), ARMA(1,1) bằng CSS.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS (dự báo 28 ngày, MASE < 0.75).

## 🔗 Tiếp theo

[26 · Recommender Systems](../26_recommender_systems/README.md)
