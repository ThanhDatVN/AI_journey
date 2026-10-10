# Đánh giá mô hình — Bạn đã thực sự hiểu?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Vẽ ma trận nhầm lẫn; định nghĩa precision, recall, specificity, F1 từ nó. |
| 2 | Vì sao accuracy vô nghĩa với dữ liệu mất cân bằng? Ví dụ. |
| 3 | Khi nào ưu tiên precision, khi nào recall? Fβ dùng thế nào? |
| 4 | ROC-AUC đo gì? Diễn giải xác suất của AUC. |
| 5 | Khi nào PR-AUC phù hợp hơn ROC-AUC? |
| 6 | Ngưỡng 0.5 có gì đặc biệt? Chọn ngưỡng theo chi phí thế nào? |
| 7 | Macro, micro, weighted average khác nhau thế nào cho bài toán nhiều lớp? |
| 8 | MAE, MSE, RMSE, MAPE, R² — mỗi metric nhạy với điều gì? |
| 9 | Vì sao R² có thể âm? |
| 10 | Log-loss và Brier score đo gì mà AUC không đo? |
| 11 | Chọn metric cho: sàng lọc ung thư; lọc thư rác; dự báo nhu cầu kho; xếp hạng gợi ý. |
| 12 | Vì sao nên báo metric kèm độ bất định? Làm thế nào? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Hàng = thật, cột = dự đoán: TP, FN / FP, TN. Precision = TP/(TP+FP); recall = TP/(TP+FN); specificity = TN/(TN+FP); F1 = 2PR/(P+R).
2. Đoán "âm tính" cho mọi mẫu với 1% dương tính đã đạt 99% accuracy nhưng không bắt được ca nào.
3. Precision khi báo động giả đắt (lọc thư rác, đề xuất duyệt tự động); recall khi bỏ sót đắt (bệnh nặng, gian lận). Fβ với β > 1 nhấn mạnh recall.
4. Khả năng xếp hạng: xác suất một mẫu dương ngẫu nhiên được chấm điểm cao hơn một mẫu âm ngẫu nhiên; không phụ thuộc ngưỡng.
5. Khi lớp dương hiếm và quan tâm chất lượng của nhóm được cảnh báo — ROC có thể trông tốt vì FPR nhỏ trên số âm khổng lồ.
6. Chỉ tối ưu khi chi phí hai loại sai bằng nhau và xác suất hiệu chỉnh. Ngưỡng tối ưu: p* = C_FP/(C_FP + C_FN), hoặc quét ngưỡng trên validation để tối thiểu chi phí.
7. Macro: trung bình đều các lớp (lớp hiếm quan trọng như lớp lớn); micro: gộp TP/FP/FN toàn bộ (≈ accuracy); weighted: theo số mẫu mỗi lớp.
8. MAE: sai số tuyệt đối trung bình, bền ngoại lai; MSE/RMSE: phạt nặng sai số lớn; MAPE: tương đối, nổ khi y ≈ 0; R²: tỉ lệ phương sai giải thích so với dự đoán trung bình.
9. Khi mô hình tệ hơn việc luôn dự đoán trung bình (thường gặp trên tập test khác phân phối).
10. Chất lượng của **xác suất** (calibration + độ sắc nét); AUC chỉ đo thứ hạng.
11. Ung thư: recall (ở mức precision chấp nhận được), PR-AUC. Thư rác: precision cao (đừng mất thư quan trọng). Kho: MAE/WAPE + chi phí thiếu/thừa hàng. Gợi ý: NDCG@k, precision@k.
12. Một con số trên một tập test dao động theo mẫu; dùng bootstrap trên tập test hoặc CV lặp, báo khoảng tin cậy, so sánh mô hình theo cặp.

</details>
