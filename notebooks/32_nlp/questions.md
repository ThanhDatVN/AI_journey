# NLP — Bạn đã thực sự "biết" xử lý văn bản?

Trả lời bằng lời của bạn, không nhìn tài liệu. Mở gợi ý chỉ sau khi đã tự trả lời. Ôn lại sau 1 ngày, 1 tuần, 1 tháng.

| # | Câu hỏi |
|---|---|
| 1 | Vì sao phải chuẩn hoá Unicode với tiếng Việt? NFC và NFD khác nhau thế nào? |
| 2 | Bỏ dấu có lợi và hại gì? Khi nào nên dùng? |
| 3 | Tách âm tiết và tách từ khác nhau thế nào? Các cách tách từ tiếng Việt? |
| 4 | Khi nào bỏ stopword có hại? |
| 5 | Mô tả BPE. Kích thước từ vựng ảnh hưởng gì tới độ dài chuỗi và OOV? |
| 6 | Định luật Zipf nói gì? Hệ quả cho mô hình ngôn ngữ? |
| 7 | Viết công thức TF-IDF. Vì sao chuẩn hoá L2? |
| 8 | Vì sao mô hình tuyến tính trên TF-IDF là baseline mạnh? Diễn giải nó thế nào? |
| 9 | Vì sao BoW không xử lý được phủ định? n-gram giúp đến đâu? |
| 10 | Phân tích lỗi phân loại văn bản: nên xem gì? |
| 11 | Giả thuyết phân bố là gì? PPMI là gì? |
| 12 | Mô tả word2vec skip-gram với negative sampling. |
| 13 | Đánh giá embedding thế nào? |
| 14 | Khi nào embedding giúp phân loại hơn TF-IDF? |
| 15 | Giới hạn của embedding tĩnh? |
| 16 | BM25 khác TF-IDF thế nào? Recall@k, MRR là gì? |
| 17 | Lexical, dense, hybrid retrieval: khi nào dùng cái nào? |
| 18 | RAG là gì? Vì sao chất lượng retrieval quan trọng? |

<details>
<summary>💡 Gợi ý trả lời</summary>

1. Ký tự có dấu có hai cách mã hoá (dựng sẵn NFC / tổ hợp NFD) và hai kiểu bỏ dấu (hoà/hòa) → cùng từ thành chuỗi khác nhau; chuẩn hoá NFC + thống nhất kiểu dấu khi đọc dữ liệu.
2. Lợi: khớp với truy vấn gõ không dấu (tìm kiếm cải thiện rất mạnh). Hại: nhiều từ trùng nhau (má/mà/mả → ma). Dùng cho so khớp/tìm kiếm, áp dụng cả hai phía; giữ bản có dấu cho mô hình hiểu nghĩa.
3. Tiếng Việt viết cách theo âm tiết; từ có thể nhiều âm tiết. Khớp cực đại (tham lam, sai khi mơ hồ), CRF/mạng nơ-ron (VnCoreNLP, underthesea), hoặc bỏ qua và dùng subword.
4. Khi từ trong danh sách mang nghĩa cho bài toán: "không", "chưa", "rất" với cảm xúc; mô hình hiện đại thường không cần bỏ stopword.
5. Gộp dần cặp ký hiệu phổ biến nhất; tokenize bằng áp dụng phép gộp theo thứ tự. Từ vựng lớn → chuỗi ngắn, mỗi token hiếm hơn; nhỏ → chuỗi dài. Không có OOV.
6. Tần suất ∝ 1/hạng^s: vài từ chiếm phần lớn văn bản, rất nhiều từ hiếm → luôn có từ chưa gặp, cần smoothing/subword.
7. tf × idf, idf = ln((1+N)/(1+df)) + 1; L2 để văn bản dài/ngắn so sánh được (cosine = tích vô hướng).
8. Văn bản nhiều chiều, thưa, gần tách tuyến tính; train nhanh, ít overfit với regularization. Hệ số mỗi lớp cho biết từ nào đẩy về lớp đó.
9. Mất thứ tự; "không" đảo nghĩa từ sau là tương tác — tuyến tính trên unigram không biểu diễn được. Bigram bắt phủ định liền kề; phủ định xa, mỉa mai cần mô hình chuỗi.
10. Ma trận nhầm lẫn, đọc các mẫu sai: nhãn sai, đoạn lạc đề, từ ghép bị tách, từ chưa gặp — sửa dữ liệu/biểu diễn trước khi đổi mô hình.
11. Từ có ngữ cảnh giống nhau có nghĩa giống nhau. PMI = log P(w,c)/(P(w)P(c)); PPMI cắt âm về 0 — đo đồng xuất hiện "hơn ngẫu nhiên".
12. Từ trung tâm dự đoán từ ngữ cảnh; thay softmax bằng phân biệt cặp thật với k cặp ngẫu nhiên (theo tần suất^0.75); ngầm phân tích ma trận PMI dịch chuyển.
13. Nội tại: láng giềng, bộ đồng nghĩa/tương đồng, phép loại suy; ngoại tại: hiệu quả trên tác vụ đích. Cosine chỉ so được trong cùng không gian.
14. Khi ít nhãn và có nhiều văn bản không nhãn: embedding mang thông tin đồng nghĩa mà vài văn bản có nhãn không có. Khi đủ nhãn, TF-IDF + n-gram thường ngang hoặc tốt hơn.
15. Một vector mỗi từ (đa nghĩa bị trộn), không có vector cho từ mới (fastText dùng n-gram ký tự), mang thiên lệch dữ liệu; embedding ngữ cảnh (BERT) khắc phục đa nghĩa.
16. BM25: tf bão hoà (k1), chuẩn hoá độ dài (b). Recall@k: tỉ lệ truy vấn có tài liệu đúng trong top-k; MRR: trung bình 1/hạng.
17. Lexical cho mã, tên riêng, thuật ngữ chính xác; dense cho diễn đạt tự do, đồng nghĩa; hybrid + re-ranking trong hệ thống thật, trọng số chọn trên validation.
18. Truy xuất đoạn liên quan rồi đưa vào prompt cho LLM; nếu retrieval không tìm được thông tin đúng, LLM không thể trả lời đúng (hoặc bịa) — đánh giá retrieval riêng.

</details>
