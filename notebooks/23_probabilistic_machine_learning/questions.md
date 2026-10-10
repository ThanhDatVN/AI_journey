# Machine Learning xác suất — Bạn đã thực sự "biết" GMM, hồi quy Bayes, GP, HMM?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Ma trận hiệp phương sai quyết định hình dạng Gaussian thế nào? |
| 2 | Viết công thức Gaussian có điều kiện. Nó liên hệ với hồi quy tuyến tính và GP ra sao? |
| 3 | LDA, QDA, Gaussian NB khác nhau ở giả định nào? |
| 4 | GMM sinh dữ liệu thế nào? Vì sao cần EM? |
| 5 | E-step và M-step làm gì? Vì sao log-likelihood không giảm? EM có tìm cực đại toàn cục không? |
| 6 | Chọn số thành phần và kiểu hiệp phương sai thế nào? |
| 7 | Singularity trong GMM là gì? Xử lý thế nào? |
| 8 | K-Means liên hệ với GMM ra sao? |
| 9 | Hồi quy tuyến tính Bayes khác Ridge ở đâu? Giống ở đâu? |
| 10 | Phương sai dự đoán gồm những thành phần nào? Khi nào độ bất định đáng tin? |
| 11 | Evidence (marginal likelihood) là gì? Vì sao nó chọn được mô hình mà likelihood train thì không? |
| 12 | Gaussian Process là gì? Kernel đóng vai trò gì? |
| 13 | Siêu tham số GP chọn thế nào? Length scale ảnh hưởng gì? |
| 14 | Vì sao GP với RBF không ngoại suy được xu hướng? |
| 15 | Hạn chế tính toán của GP và cách vượt qua? |
| 16 | HMM gồm những thành phần nào? Ba bài toán và thuật toán tương ứng? |
| 17 | Viterbi khác posterior decoding thế nào? |
| 18 | Baum–Welch là gì? Những khó khăn khi học HMM? |
| 19 | Khi nào nên dùng mô hình xác suất thay vì model dự đoán điểm? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. $\Sigma = Q\Lambda Q^\top$: ellipse đồng mức có trục theo vector riêng, bán trục $\propto\sqrt{\lambda}$; hiệp phương sai ngoài đường chéo → ellipse nghiêng (tương quan).
2. $\boldsymbol\mu_{a|b} = \boldsymbol\mu_a + \Sigma_{ab}\Sigma_{bb}^{-1}(\mathbf x_b - \boldsymbol\mu_b)$, $\Sigma_{a|b} = \Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}$. Kỳ vọng có điều kiện tuyến tính = đường hồi quy; GP regression = áp dụng công thức này cho giá trị hàm tại điểm mới với điều kiện là các quan sát.
3. Gaussian NB: hiệp phương sai chéo riêng mỗi lớp; LDA: hiệp phương sai đầy đủ **chung** → ranh giới tuyến tính; QDA: đầy đủ **riêng** → bậc hai, nhiều tham số nhất.
4. Chọn $z \sim \text{Cat}(\boldsymbol\pi)$ rồi $\mathbf x \sim \mathcal N(\boldsymbol\mu_z, \Sigma_z)$. Log-likelihood là log của tổng → không có nghiệm đóng; nếu biết $z$ thì dễ → EM luân phiên đoán $z$ (mềm) và cập nhật tham số.
5. E-step: responsibility $r_{ik}$; M-step: MLE có trọng số. Jensen → ELBO; E-step làm cận chạm, M-step đẩy cận lên → không giảm. Chỉ cực đại địa phương → nhiều khởi tạo (`n_init`), khởi tạo K-Means.
6. BIC (hoặc AIC) trên lưới K × `covariance_type`; diag/tied khi nhiều chiều, ít dữ liệu.
7. Thành phần co về một điểm, phương sai → 0, likelihood → ∞. `reg_covar`, prior (MAP / Bayesian GMM), loại điểm trùng, khởi động lại thành phần.
8. Giới hạn của EM cho GMM với $\Sigma = \sigma^2 I$ chung, $\sigma \to 0$: gán cứng, tâm = trung bình cụm.
9. Ridge cho một điểm $\mathbf w$ = trung bình (= mode) hậu nghiệm với $\lambda = \alpha/\beta$. Bayes giữ cả phân phối → khoảng dự đoán, học tuần tự, chọn mô hình bằng evidence.
10. $\beta^{-1}$ (nhiễu, không giảm) + $\boldsymbol\phi^\top S_N\boldsymbol\phi$ (bất định tham số, giảm khi có dữ liệu). Đáng tin khi giả định model đúng (nhiễu Gaussian, hàm cơ sở phù hợp) — luôn kiểm tra độ phủ trên dữ liệu giữ lại.
11. $p(\mathbf y \mid \text{model}) = \int p(\mathbf y \mid \mathbf w)p(\mathbf w)\,d\mathbf w$ — trung bình trên prior. Model phức tạp "rải" xác suất cho quá nhiều bộ dữ liệu → mỗi bộ cụ thể có evidence thấp (Occam's razor); likelihood train dùng tham số tốt nhất nên luôn tăng.
12. Phân phối trên hàm: mọi tập hữu hạn giá trị hàm có phân phối Gaussian chung với hiệp phương sai $K_{ij} = k(x_i, x_j)$. Kernel mã hoá giả định: độ trơn, tuần hoàn, xu hướng.
13. Tối đa log marginal likelihood (khớp dữ liệu − phạt độ phức tạp), nhiều điểm khởi đầu. ℓ nhỏ: uốn lượn, bất định lớn giữa các điểm; ℓ lớn: gần đường thẳng, underfit.
14. Xa dữ liệu, $k(x^*, x_i) \to 0$ → trung bình về prior. Dùng kernel tuyến tính (DotProduct) cho xu hướng, ExpSineSquared cho mùa vụ; chọn bằng log-ML và backtest.
15. Cholesky $O(n^3)$, bộ nhớ $O(n^2)$. Sparse GP / inducing points, random Fourier features, GPU (GPyTorch), chia nhỏ dữ liệu.
16. $\boldsymbol\pi$ (ban đầu), $A$ (chuyển), $B$ (phát xạ). Đánh giá: forward; giải mã: Viterbi / forward–backward; học: Baum–Welch.
17. Viterbi: chuỗi trạng thái **đồng thời** khả dĩ nhất (max); posterior decoding: trạng thái khả dĩ nhất **từng thời điểm** + xác suất (có thể tạo chuỗi có chuyển trạng thái xác suất 0).
18. EM cho HMM: E-step forward–backward ($\gamma$, $\xi$), M-step đếm có trọng số. Khó khăn: cực trị địa phương, hoán vị nhãn, cần nhiều dữ liệu, chọn số trạng thái, underflow (tính trong log).
19. Khi cần độ bất định cho quyết định (cam kết, rủi ro, y tế), ít dữ liệu, học tuần tự, active learning / Bayesian optimization, hoặc cần mô hình sinh (mật độ, dữ liệu thiếu, biến ẩn diễn giải được).

</details>
