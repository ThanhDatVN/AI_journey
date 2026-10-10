# Chương 31 · Transformers

> "Attention is all you need." Chương xây Transformer từ từng mảnh: attention như tra cứu mềm (NumPy), multi-head attention và mã hoá vị trí (sinusoidal, learned, RoPE), khối Transformer, một mini GPT sinh review tiếng Việt, rồi ba họ kiến trúc — encoder–decoder, BERT, GPT — và ViT cho ảnh. Mỗi bước đều có thí nghiệm cho thấy **vì sao** thiết kế như vậy, và khi nào Transformer **không** thắng.

## 🎯 Mục tiêu

- Scaled dot-product attention từ đầu; vì sao chia $\sqrt{d_k}$ (đo entropy, gradient); padding/causal mask; tính hoán vị-đẳng biến; chi phí $O(n^2)$.
- Multi-head attention (khớp `nn.MultiheadAttention`); mã hoá vị trí sinusoidal/learned/RoPE và tính chất vị trí tương đối; khối pre-LN; không có vị trí thì không đảo/sắp xếp được.
- Mini GPT: causal attention, weight tying, công thức train (AdamW, warmup + cosine), kiểm tra nhân quả, nhiệt độ/top-k/top-p, KV cache; GPT vs LSTM (ngang nhau trên review, GPT vượt hẳn ở bài sao chép trong ngữ cảnh).
- Encoder–decoder với cross-attention; BERT: MLM, pretrain → fine-tune khi ít nhãn (và so với baseline TF-IDF); ViT vs CNN theo lượng dữ liệu; dùng mô hình tiền huấn luyện (Hugging Face).

## 📋 Yêu cầu trước

[Chương 28](../28_neural_networks/README.md) (huấn luyện mạng), [Chương 30](../30_sequence_models/README.md) (seq2seq, attention); [Chương 29](../29_cnn/README.md) để so sánh với ViT.

## 🧰 Thư viện

PyTorch — `pip install -e ".[dl]"`. Bài 02–04 huấn luyện nhiều mô hình nhỏ: tự dùng GPU nếu có, vẫn chạy được trên CPU (chậm hơn đáng kể). Phần Hugging Face ở bài 04 là tùy chọn (`RUN_HF = False`, cần `pip install transformers` và mạng).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Attention from scratch | 📖 Bài học | [View](01_attention_from_scratch.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/31_transformers/01_attention_from_scratch.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/31_transformers/01_attention_from_scratch.ipynb) |
| 02 | Bài 02 — Multi-head attention, positional encoding và khối Transformer | 📖 Bài học | [View](02_multi_head_and_positional_encoding.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/31_transformers/02_multi_head_and_positional_encoding.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/31_transformers/02_multi_head_and_positional_encoding.ipynb) |
| 03 | Bài 03 — Mini GPT: Transformer decoder-only từ đầu | 📖 Bài học | [View](03_mini_gpt.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/31_transformers/03_mini_gpt.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/31_transformers/03_mini_gpt.ipynb) |
| 04 | Bài 04 — Encoder–decoder, BERT, GPT, ViT | 📖 Bài học | [View](04_encoder_decoder_bert_gpt_vit.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/31_transformers/04_encoder_decoder_bert_gpt_vit.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/31_transformers/04_encoder_decoder_bert_gpt_vit.ipynb) |
| 90 | Bài tập — Chương 31: Transformers (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/31_transformers/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/31_transformers/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 31 · Transformers | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/31_transformers/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/31_transformers/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

Bài 01: `attention` (NumPy, có mask), `RecallAttention`. Bài 02: `MultiHeadAttention`, `sinusoidal_encoding`, `apply_rope`, `TransformerBlock`. Bài 03: `CausalSelfAttention` (có KV cache), `GPT`, `filter_logits`, `generate`. Bài 04: `TransformerSeq2Seq`, `mask_tokens`, `MiniBERT`, `ViT`. Bài tập: `make_attention_mask`, `mha_forward`, induction head, `chunked_attention` (online softmax).

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS (Transformer sắp xếp dãy số, khớp cả chuỗi ≥ 0.9).

## 🔗 Tiếp theo

[32 · NLP](../32_nlp/README.md)
