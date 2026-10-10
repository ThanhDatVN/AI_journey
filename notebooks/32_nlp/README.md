# Chương 32 · NLP — Xử lý ngôn ngữ tự nhiên

> Từ chuỗi ký tự đến ý nghĩa: chuẩn hoá và tách từ tiếng Việt (Unicode NFC/NFD, không dấu, từ ghép, BPE), Bag-of-Words/TF-IDF và các mô hình tuyến tính mạnh bất ngờ, word embedding học từ văn bản không nhãn, rồi tìm kiếm ngữ nghĩa — nền tảng của search và RAG. Chương dùng ngữ liệu tin tức tổng hợp có từ đồng nghĩa và một bộ FAQ với truy vấn viết tay.

## 🎯 Mục tiêu

- Tiền xử lý tiếng Việt đúng cách: chuẩn hoá Unicode, kiểu bỏ dấu, bỏ dấu khi so khớp, tách từ (khớp cực đại và giới hạn), stopword; BPE from scratch; định luật Zipf.
- BoW, TF-IDF (khớp `TfidfVectorizer`), Naive Bayes/LogReg/SVM, n-gram; diễn giải hệ số, phân tích lỗi; giới hạn phủ định; đường cong số nhãn.
- Embedding: PPMI + SVD, word2vec skip-gram negative sampling (PyTorch); đánh giá bằng láng giềng và bộ đồng nghĩa; embedding cho phân loại ít nhãn; đa nghĩa.
- Retrieval: Jaccard/cosine, Recall@k/MRR, BM25 from scratch, bỏ dấu cho truy vấn không dấu, lexical vs dense vs hybrid (chọn trọng số trên validation), sentence embedding/ANN/RAG.

## 📋 Yêu cầu trước

[Chương 21](../21_dimensionality_reduction/README.md) (SVD), [Chương 27](../27_deep_learning_foundations/README.md) (PyTorch); [Chương 31](../31_transformers/README.md) giúp kết nối với BERT/sentence embedding.

## 🧰 Thư viện

scikit-learn; bài 03 dùng PyTorch (`pip install -e ".[dl]"`) để train word2vec (vài giây trên CPU). Phần sentence-transformers ở bài 04 là tùy chọn (`RUN_ST = False`).

## 📚 Nội dung

<!-- LESSONS:START -->
| # | Nội dung | Loại | GitHub | nbviewer | Colab |
|:---:|:---|:---:|:---:|:---:|:---:|
| 01 | Bài 01 — Tiền xử lý văn bản và tokenization | 📖 Bài học | [View](01_text_preprocessing_and_tokenization.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/32_nlp/01_text_preprocessing_and_tokenization.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/32_nlp/01_text_preprocessing_and_tokenization.ipynb) |
| 02 | Bài 02 — Bag-of-Words, TF-IDF và phân loại văn bản | 📖 Bài học | [View](02_bow_tfidf_classification.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/32_nlp/02_bow_tfidf_classification.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/32_nlp/02_bow_tfidf_classification.ipynb) |
| 03 | Bài 03 — Word embeddings | 📖 Bài học | [View](03_word_embeddings.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/32_nlp/03_word_embeddings.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/32_nlp/03_word_embeddings.ipynb) |
| 04 | Bài 04 — Độ tương đồng ngữ nghĩa và tìm kiếm (retrieval) | 📖 Bài học | [View](04_semantic_similarity_and_retrieval.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/32_nlp/04_semantic_similarity_and_retrieval.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/32_nlp/04_semantic_similarity_and_retrieval.ipynb) |
| 90 | Bài tập — Chương 32: NLP (L0 → L7) | 📝 Bài tập | [View](90_exercises.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/32_nlp/90_exercises.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/32_nlp/90_exercises.ipynb) |
| 99 | ✅ Checkpoint — 32 · NLP | ✅ Checkpoint | [View](99_checkpoint.ipynb) | [View](https://nbviewer.org/github/ThanhDatVN/AI_journey/blob/main/notebooks/32_nlp/99_checkpoint.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThanhDatVN/AI_journey/blob/main/notebooks/32_nlp/99_checkpoint.ipynb) |
<!-- LESSONS:END -->

## ❓ Bộ câu hỏi "biết model thật sự"

[questions.md](questions.md)

## 🧩 Code tham chiếu

Bài 01: `strip_accents`, `clean`, `segment` (khớp cực đại), `train_bpe` / `bpe_tokenize`. Bài 02: `tfidf_from_scratch`. Bài 03: PPMI-SVD, `SkipGramNS`. Bài 04: `BM25`, `evaluate` (Recall@k, MRR), hybrid z-score. Bài tập: `normalize_vi`, `MyMultinomialNB`, mở rộng truy vấn, NMF topic modeling. Dữ liệu: `make_news_vi`, `NEWS_SYNONYMS`, `NEWS_LEXICON` (`ml_course.datasets`); `data/sample/faq_vi.csv`, `faq_queries_vi.csv`.

## ✅ Checkpoint

`99_checkpoint.ipynb` — Theory ≥ 80%, Math ≥ 80%, coding PASS, phân tích (rubric) PASS, mini project PASS (phân loại chủ đề tin, macro-F1 ≥ 0.95).

## 🔗 Tiếp theo

[33 · Computer Vision](../33_computer_vision/README.md)
