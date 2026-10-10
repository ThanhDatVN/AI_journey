# Reinforcement Learning — Bạn đã thực sự "biết" học tăng cường?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Học tăng cường khác học có giám sát ở những điểm nào? |
| 2 | Khám phá vs khai thác là gì? So sánh ε-greedy và UCB. |
| 3 | MDP gồm những thành phần nào? Tính Markov là gì? |
| 4 | Return, hàm giá trị V và Q là gì? Vai trò của γ? |
| 5 | Viết phương trình Bellman kỳ vọng và tối ưu. |
| 6 | Value iteration và policy iteration hoạt động thế nào? Cần giả định gì? |
| 7 | Monte Carlo và TD khác nhau thế nào? Bias–variance? |
| 8 | SARSA và Q-learning khác nhau thế nào? On-policy vs off-policy? |
| 9 | Vì sao Q-learning đánh giá quá cao? Double Q-learning sửa ra sao? |
| 10 | Vì sao Q-learning với mạng nơ-ron dễ bất ổn? DQN làm gì? |
| 11 | Định lý policy gradient nói gì? Vai trò của baseline? |
| 12 | So sánh value-based và policy-based. Actor-critic, PPO là gì? |
| 13 | Vì sao phải báo cáo deep RL trên nhiều seed? |
| 14 | Thuật toán di truyền gồm những bước nào? Vai trò của đột biến? |
| 15 | Neuroevolution / Evolution Strategies hoạt động thế nào? Khi nào nên dùng? |
| 16 | Đánh giá off-policy là gì? Vì sao cần log propensity? |
| 17 | Những rủi ro khi đưa RL vào thực tế? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Không có nhãn từng bước, chỉ phần thưởng (thường trễ); dữ liệu phụ thuộc hành động; phải khám phá; mục tiêu là return dài hạn.
2. Khai thác: chọn cái đang tốt nhất; khám phá: thử để biết thêm. ε-greedy khám phá ngẫu nhiên; UCB cộng thưởng cho độ không chắc chắn — khám phá có định hướng, ít lãng phí.
3. (S, A, P, R, γ); tương lai chỉ phụ thuộc trạng thái hiện tại, không phụ thuộc lịch sử.
4. G = tổng thưởng chiết khấu; V(s) = E[G | s]; Q(s, a) = E[G | s, a]; γ quyết định mức "nhìn xa"/kiên nhẫn.
5. V^π(s) = Σ_a π(a|s) Σ P[r + γV^π(s')]; V*(s) = max_a Σ P[r + γV*(s')].
6. VI lặp Bellman tối ưu rồi lấy tham lam; PI luân phiên đánh giá chính xác và cải thiện tham lam. Cần biết mô hình và duyệt được mọi trạng thái.
7. MC: return thật, không thiên lệch, phương sai cao, chờ hết lượt. TD: r + γV(s'), thiên lệch, phương sai thấp, học mỗi bước.
8. SARSA bootstrap bằng hành động sẽ làm (on-policy: giá trị chính sách đang chạy); Q-learning bằng max (off-policy: giá trị chính sách tối ưu).
9. Max trên ước lượng nhiễu luôn ≥ max của kỳ vọng. Double Q: một bảng chọn hành động, bảng kia đánh giá.
10. Deadly triad (xấp xỉ hàm + bootstrap + off-policy), mẫu tương quan, mục tiêu di động. DQN: experience replay, target network, Double DQN, Huber loss.
11. ∇J = E[Σ ∇log π(a|s)·G]; baseline giảm phương sai không làm lệch kỳ vọng (advantage = G − V(s)).
12. Value-based: học Q, off-policy, hiệu quả mẫu, khó với hành động liên tục. Policy-based: học π trực tiếp, xử lý hành động liên tục, phương sai cao. Actor-critic kết hợp; PPO giới hạn bước cập nhật chính sách để ổn định.
13. Kết quả dao động rất lớn theo seed và thời điểm; một seed có thể là may mắn; cần trung bình, khoảng tin cậy, checkpoint tốt nhất.
14. Đánh giá fitness → chọn lọc → lai ghép → đột biến (+ tinh hoa). Đột biến duy trì đa dạng: quá nhỏ → hội tụ sớm, quá lớn → mất thông tin di truyền.
15. Tối ưu tham số chính sách bằng quần thể/nhiễu ngẫu nhiên, fitness = return; ES ước lượng gradient bằng nhiễu đối xứng. Dùng khi không khả vi, thưởng thưa, song song hoá được.
16. Ước lượng giá trị chính sách mới từ log của chính sách cũ (IPS, SNIPS, DR); cần xác suất chính sách cũ đã chọn hành động (propensity) để cân lại.
17. Chi phí khám phá trên người/hệ thống thật, reward hacking, sai lệch mô phỏng–thực tế, ngoại suy khi offline, bất ổn theo seed, môi trường thay đổi; cần an toàn, giới hạn hành động, con người trong vòng lặp.

</details>
