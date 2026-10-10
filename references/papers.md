# 📄 Paper

Những paper gốc được nhắc trong khoá học, theo chủ đề. Đọc sau khi đã học chương tương ứng — mục tiêu là hiểu **ý tưởng chính** và **thí nghiệm**, không cần hiểu mọi chứng minh.

## Nền tảng, đánh giá, thực hành

| Paper | Ý chính | Ch. |
|---|---|---|
| Wolpert (1996) — *The Lack of A Priori Distinctions Between Learning Algorithms* | No Free Lunch | 05 |
| Kaufman et al. (2012) — *Leakage in Data Mining: Formulation, Detection, and Avoidance* | phân loại các dạng rò rỉ | 05 |
| Bergstra & Bengio (2012) — *Random Search for Hyper-Parameter Optimization* | random search hơn grid search | 16 |
| Li et al. (2018) — *Hyperband* | successive halving có nguyên tắc | 16 |
| Akiba et al. (2019) — *Optuna* | tối ưu siêu tham số define-by-run | 16 |
| Cawley & Talbot (2010) — *On Over-fitting in Model Selection…* | lạc quan khi chọn mô hình, nested CV | 15, 39 |
| Chawla et al. (2002) — *SMOTE* | tăng mẫu lớp hiếm tổng hợp | 19 |
| Elkan (2001) — *The Foundations of Cost-Sensitive Learning* | ngưỡng theo chi phí | 19 |
| Nadeau & Bengio (2003) — *Inference for the Generalization Error* | t-test hiệu chỉnh cho CV lặp | 39 |

## Mô hình kinh điển

| Paper | Ý chính | Ch. |
|---|---|---|
| Tibshirani (1996) — *Regression Shrinkage and Selection via the Lasso* | Lasso | 06 |
| Breiman (1996) — *Bagging Predictors*; Breiman (2001) — *Random Forests* | bagging, rừng ngẫu nhiên | 11 |
| Freund & Schapire (1997) — *A Decision-Theoretic Generalization of On-Line Learning…* | AdaBoost | 12 |
| Friedman (2001) — *Greedy Function Approximation: A Gradient Boosting Machine* | gradient boosting | 12 |
| Chen & Guestrin (2016) — *XGBoost*; Ke et al. (2017) — *LightGBM*; Prokhorenkova et al. (2018) — *CatBoost* | boosting hiện đại | 12 |
| Cortes & Vapnik (1995) — *Support-Vector Networks* | SVM | 13 |
| Wolpert (1992) — *Stacked Generalization* | stacking | 24 |
| Ester et al. (1996) — *DBSCAN* | phân cụm theo mật độ | 20 |
| Liu, Ting, Zhou (2008) — *Isolation Forest* | phát hiện bất thường bằng cô lập | 22 |
| van der Maaten & Hinton (2008) — *t-SNE*; McInnes et al. (2018) — *UMAP* | giảm chiều trực quan hoá | 21 |
| Koren, Bell, Volinsky (2009) — *Matrix Factorization Techniques for Recommender Systems* | MF cho gợi ý | 26 |
| Cremonesi et al. (2010) — *Performance of Recommender Algorithms on Top-N Recommendation Tasks* | PureSVD, đánh giá top-N | 26 |

## Deep learning

| Paper | Ý chính | Ch. |
|---|---|---|
| Rumelhart, Hinton, Williams (1986) — *Learning Representations by Back-propagating Errors* | backpropagation | 27 |
| Glorot & Bengio (2010); He et al. (2015) — *Delving Deep into Rectifiers* | khởi tạo Xavier, He | 28 |
| Ioffe & Szegedy (2015) — *Batch Normalization*; Ba et al. (2016) — *Layer Normalization* | chuẩn hoá | 28, 31 |
| Kingma & Ba (2015) — *Adam*; Loshchilov & Hutter (2019) — *Decoupled Weight Decay (AdamW)* | optimizer | 28 |
| Srivastava et al. (2014) — *Dropout* | điều chuẩn | 28 |
| LeCun et al. (1998); Krizhevsky et al. (2012) — *AlexNet*; He et al. (2016) — *ResNet* | CNN | 29 |
| Hochreiter & Schmidhuber (1997) — *LSTM*; Cho et al. (2014) — *GRU* | mô hình chuỗi | 30 |
| Bahdanau et al. (2015) — *Neural Machine Translation by Jointly Learning to Align and Translate* | attention | 30 |
| Vaswani et al. (2017) — *Attention Is All You Need* | Transformer | 31 |
| Devlin et al. (2019) — *BERT*; Radford et al. (2018, 2019) — *GPT, GPT-2*; Dosovitskiy et al. (2021) — *ViT* | mô hình tiền huấn luyện | 31 |
| Mikolov et al. (2013) — *word2vec*; Joulin et al. (2016) — *fastText* | embedding từ | 32 |
| Nguyen & Nguyen (2020) — *PhoBERT* | BERT tiếng Việt | 32 |
| Kingma & Welling (2014) — *Auto-Encoding Variational Bayes* | VAE | 34 |
| Chen et al. (2020) — *SimCLR* | học tương phản | 34 |
| Geirhos et al. (2020) — *Shortcut Learning in Deep Neural Networks* | học đường tắt | 33, project 07 |
| Mnih et al. (2015) — *Human-level control through deep RL (DQN)*; Williams (1992) — *REINFORCE* | RL sâu | 35 |
| Salimans et al. (2017) — *Evolution Strategies as a Scalable Alternative to RL* | neuroevolution | 35 |

## Đáng tin cậy

| Paper | Ý chính | Ch. |
|---|---|---|
| Ribeiro et al. (2016) — *"Why Should I Trust You?" (LIME)*; Lundberg & Lee (2017) — *SHAP* | giải thích cục bộ | 36 |
| Rudin (2019) — *Stop Explaining Black Box Models for High Stakes Decisions…* | ưu tiên mô hình diễn giải được | 36 |
| Kleinberg et al. (2016) — *Inherent Trade-Offs in the Fair Determination of Risk Scores* | các tiêu chí công bằng xung đột | 36 |
| Guo et al. (2017) — *On Calibration of Modern Neural Networks* | calibration, temperature scaling | 37 |
| Niculescu-Mizil & Caruana (2005) — *Predicting Good Probabilities with Supervised Learning* | Platt vs isotonic | 37 |
| Romano et al. (2019) — *Conformalized Quantile Regression* | CQR | 37 |
| Lakshminarayanan et al. (2017) — *Deep Ensembles* | bất định trong DL | 37 |
| Rosenbaum & Rubin (1983) — *The Central Role of the Propensity Score…* | điểm xu hướng | 38 |
| Chernozhukov et al. (2018) — *Double/Debiased Machine Learning* | DML | 38 |
| Künzel et al. (2019) — *Metalearners for Estimating Heterogeneous Treatment Effects* | S/T/X-learner | 38 |
| Deng et al. (2013) — *CUPED* | giảm phương sai A/B test | 38 |

## Hệ thống và MLOps

| Paper | Ý chính | Ch. |
|---|---|---|
| Sculley et al. (2015) — *Hidden Technical Debt in Machine Learning Systems* | nợ kỹ thuật của hệ thống ML | 39–40 |
| Breck et al. (2017) — *The ML Test Score* | rubric sẵn sàng production | 39–41 |
| Mitchell et al. (2019) — *Model Cards for Model Reporting*; Gebru et al. (2021) — *Datasheets for Datasets* | tài liệu mô hình, dữ liệu | 39 |
| Ribeiro et al. (2020) — *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList* | kiểm thử hành vi | 40 |
| Rabanser et al. (2019) — *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift* | phát hiện dịch chuyển | 40 |
| Gama et al. (2014) — *A Survey on Concept Drift Adaptation* | trôi khái niệm | 40 |
| Johari et al. (2017) — *Peeking at A/B Tests* | nhìn trộm và kiểm định tuần tự | 40 |
| Lakkaraju et al. (2017) — *The Selective Labels Problem* | nhãn chọn lọc, vòng phản hồi | 40 |
