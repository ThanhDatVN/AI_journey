# MLOps — Bạn đã thực sự "biết" vận hành mô hình?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Một hợp đồng API cho mô hình gồm những gì? Vì sao cần phiên bản trong URL? |
| 2 | Vì sao nạp mô hình khi khởi động? Liveness khác readiness thế nào? |
| 3 | Khi nào endpoint dùng `def`, khi nào `async def`? |
| 4 | Gom lô ảnh hưởng tới độ trễ và thông lượng thế nào? Mở rộng dịch vụ CPU-bound bằng cách nào? |
| 5 | Kể sáu loại kiểm thử cho hệ thống ML, mỗi loại một ví dụ. |
| 6 | Dockerfile tốt cho dịch vụ ML có những đặc điểm gì? |
| 7 | Một pipeline CI/CD cho ML gồm những bước nào? Continuous Training là gì? |
| 8 | Giám sát những tầng nào? Tầng nào có tín hiệu ngay, tầng nào trễ? |
| 9 | Trôi dữ liệu và trôi khái niệm khác nhau thế nào? Phát hiện mỗi loại ra sao? |
| 10 | PSI là gì, tính thế nào? Vì sao không dùng p-value làm thước đo trôi? |
| 11 | Bộ phân loại miền phát hiện được gì mà giám sát từng đặc trưng bỏ sót? |
| 12 | Nhãn trễ gây khó khăn gì? Cách giảm thiểu? |
| 13 | Sau trôi khái niệm, huấn luyện lại trên cửa sổ nào? |
| 14 | So sánh shadow, canary, blue-green, A/B test, bandit. |
| 15 | Vì sao phải gán người dùng bằng băm tất định có salt? |
| 16 | Tính cỡ mẫu A/B test thế nào? Vì sao "nhìn trộm" nguy hiểm? |
| 17 | Vòng phản hồi là gì? Ví dụ và cách phòng. |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Endpoint, phương thức, schema đầu vào/đầu ra có ràng buộc, mã lỗi, `model_version`, `request_id`. Phiên bản trong URL cho phép đổi schema mà không phá bên gọi cũ.
2. Nạp mỗi request tốn thời gian (mô hình lớn: hàng trăm ms đến giây); nạp khi khởi động kèm kiểm tra hash — lỗi thì không khởi động. Liveness: tiến trình còn sống; readiness: đã sẵn sàng nhận traffic (mô hình đã nạp).
3. `def` cho tác vụ CPU chặn (predict của scikit-learn) — chạy trong threadpool; `async def` cho I/O bất đồng bộ (gọi dịch vụ khác, database async). Code chặn trong `async def` khoá event loop.
4. Lô tăng thông lượng nhiều lần (chia chi phí cố định), tăng độ trễ của request đó. Mở rộng: nhiều worker, nhiều bản sao sau load balancer.
5. Unit (hàm đặc trưng), dữ liệu (schema, cột hằng, tỉ lệ nhãn), hiệu năng tối thiểu + tái lập metric, bất biến (cột không liên quan), theo hướng (tăng cuộc gọi hỗ trợ → rủi ro tăng), API/parity.
6. Image `-slim`, cùng phiên bản Python lúc train, ghim phiên bản thư viện, cài phụ thuộc trước khi copy code (cache), `.dockerignore`, user không phải root, healthcheck, cấu hình qua biến môi trường.
7. Lint → kiểm tra dữ liệu → huấn luyện với cổng chất lượng → test → build image (tag commit) → push registry → deploy staging → phê duyệt → production. Continuous Training: pipeline huấn luyện chạy định kỳ/theo kích hoạt trên dữ liệu mới, mô hình qua cổng thăng hạng.
8. Dịch vụ, đầu vào, dự đoán (ngay); hiệu năng, kinh doanh (trễ, cần nhãn).
9. Trôi dữ liệu: P(X) đổi — PSI/KS/bộ phân loại miền; mô hình có thể vẫn tốt. Trôi khái niệm: P(y|X) đổi — chỉ thấy qua nhãn, nhãn proxy, nhóm đối chứng, sự kiện kinh doanh.
10. Σ (c − r)·ln(c/r) trên các bin theo phân vị reference; < 0.1 ổn, 0.1–0.25 vừa, > 0.25 lớn. P-value phụ thuộc cỡ mẫu — với n lớn mọi thay đổi nhỏ đều "có ý nghĩa" → cảnh báo vô nghĩa.
11. Trôi trong quan hệ giữa các đặc trưng (phân phối đồng thời) khi từng phân phối biên không đổi.
12. Hiệu năng thật chỉ biết sau khi nhãn về → mô hình có thể sai nhiều tuần. Giảm thiểu: nhãn proxy sớm, nhóm đối chứng, theo dõi sự kiện kinh doanh, ước lượng hiệu năng không cần nhãn (có giới hạn).
13. Cửa sổ nằm trong giai đoạn mới (đủ dữ liệu), chọn bằng backtest; cân nhắc trọng số giảm dần; trộn dữ liệu cũ kéo mô hình về quan hệ cũ.
14. Shadow: kiểm tra kỹ thuật không rủi ro. Canary: phát hành dần, tự rollback. Blue-green: chuyển toàn bộ, rollback tức thì. A/B: đo tác động nhân quả lên metric kinh doanh. Bandit: tối ưu phần thưởng trong lúc học, suy luận kém hơn.
15. Cùng người dùng luôn vào cùng phiên bản ở mọi server/tiến trình (`hash()` của Python ngẫu nhiên theo tiến trình); salt riêng để các thí nghiệm độc lập; bucket lồng nhau khi tăng tỉ lệ.
16. n ≈ (z₁₋α/₂·√(2p̄(1−p̄)) + z₁₋β·√(p₁(1−p₁)+p₂(1−p₂)))²/δ² — tỉ lệ nghịch bình phương hiệu ứng. Nhìn trộm và dừng khi p < 0.05 làm dương tính giả tăng (≈ 25% với 20 lần nhìn); cố định cỡ mẫu hoặc dùng kiểm định tuần tự.
17. Hành động của mô hình thay đổi dữ liệu dùng để huấn luyện nó (giữ chân khách rủi ro cao → dữ liệu "nói" họ không rủi ro; gợi ý chỉ nhận click trên món đã hiển thị; tín dụng chỉ biết người được duyệt). Phòng: nhóm đối chứng/khám phá ngẫu nhiên, log hành động và xác suất, IPW, mô hình hoá tác động.

</details>
