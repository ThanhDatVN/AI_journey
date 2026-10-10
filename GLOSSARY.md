# 📖 Thuật ngữ Anh – Việt

Tài liệu ML chủ yếu bằng tiếng Anh. Bảng này giúp bạn nối thuật ngữ tiếng Anh với cách gọi trong khoá học. Sắp theo bảng chữ cái tiếng Anh; cột cuối là chương giới thiệu chính. Dùng `Ctrl + F` để tra.

| English | Tiếng Việt | Giải thích ngắn | Ch. |
|---|---|---|---|
| A/B test | thí nghiệm A/B | chia ngẫu nhiên người dùng vào hai phiên bản để đo tác động nhân quả | 38, 40 |
| accuracy | độ chính xác | tỉ lệ dự đoán đúng — vô dụng khi lớp mất cân bằng | 14 |
| activation function | hàm kích hoạt | phi tuyến sau lớp tuyến tính: ReLU, sigmoid, tanh, GELU | 28 |
| AdaBoost | AdaBoost | boosting tăng trọng số mẫu bị phân loại sai | 12 |
| Adam | Adam | optimizer kết hợp momentum và tỉ lệ học thích nghi | 28 |
| adversarial validation | kiểm định đối kháng | phân loại train vs test để phát hiện dịch chuyển phân phối | 05, 40 |
| aleatoric uncertainty | bất định ngẫu nhiên | nhiễu vốn có của dữ liệu, không giảm khi thêm dữ liệu | 37 |
| anomaly detection | phát hiện bất thường | tìm điểm khác biệt với phần lớn dữ liệu | 22 |
| attention | cơ chế chú ý | trung bình có trọng số của value theo độ tương đồng query–key | 30, 31 |
| AUC (ROC-AUC) | diện tích dưới đường ROC | xác suất một mẫu dương được xếp trên một mẫu âm ngẫu nhiên | 14 |
| autoencoder | bộ tự mã hoá | mạng nén rồi tái tạo đầu vào qua lớp cổ chai | 34 |
| backpropagation | lan truyền ngược | tính gradient bằng quy tắc dây chuyền từ loss về tham số | 27 |
| bagging | bagging | huấn luyện nhiều mô hình trên mẫu bootstrap rồi lấy trung bình | 11 |
| baseline | mô hình cơ sở | cách làm đơn giản nhất để so sánh (tỉ lệ nền, luật nghiệp vụ) | 05, 14 |
| batch inference | suy luận theo lô | chấm điểm cả bảng theo lịch; rẻ hơn nhiều bậc mỗi dự đoán so với online | 39 |
| batch normalization | chuẩn hoá theo batch | chuẩn hoá kích hoạt theo thống kê của batch | 28 |
| batch / mini-batch | lô / lô nhỏ | tập mẫu dùng cho một bước cập nhật gradient | 06 |
| Bayes' theorem | định lý Bayes | P(A\|B) = P(B\|A)P(A)/P(B) | 02, 09 |
| bias (statistical) | độ chệch | sai lệch có hệ thống của ước lượng/mô hình | 05 |
| bias (parameter) | hệ số chặn / bias | tham số cộng thêm, b trong wx + b | 06 |
| bias–variance tradeoff | đánh đổi độ chệch – phương sai | mô hình đơn giản chệch nhiều, phức tạp phương sai lớn | 05 |
| bootstrap | bootstrap | lấy mẫu có hoàn lại để ước lượng độ bất định | 02, 11 |
| boosting | boosting | cộng dần các mô hình yếu, mỗi mô hình sửa lỗi của tổng trước đó | 12 |
| calibration | hiệu chỉnh xác suất | xác suất dự đoán khớp tần suất thực tế | 37 |
| canary deployment | triển khai canary | chuyển dần một phần nhỏ traffic sang bản mới | 40 |
| categorical feature | đặc trưng phân loại | biến nhận giá trị trong tập danh mục | 04 |
| causal inference | suy luận nhân quả | ước lượng tác động của can thiệp, không chỉ tương quan | 38 |
| centroid | tâm cụm | trung bình các điểm trong cụm | 20 |
| checkpoint | checkpoint | (khoá học) bài kiểm tra cuối chương; (DL) bản lưu mô hình | — |
| CIFAR-10 | CIFAR-10 | 60 000 ảnh màu 32×32 thuộc 10 lớp — bộ chuẩn để so sánh mô hình thị giác | 07, 29 |
| class imbalance | mất cân bằng lớp | một lớp hiếm hơn nhiều so với lớp khác | 19 |
| classification | phân loại | dự đoán nhãn rời rạc | 05 |
| clustering | phân cụm | nhóm dữ liệu không nhãn theo độ tương đồng | 20 |
| CNN (convolutional neural network) | mạng nơ-ron tích chập | mạng dùng bộ lọc chia sẻ trọng số trên ảnh | 29 |
| computational graph | đồ thị tính toán | phép tính tách thành các nút; nền tảng của autograd | 27 |
| concept drift | trôi khái niệm | quan hệ P(y\|X) thay đổi theo thời gian | 40 |
| confounder | biến gây nhiễu | biến ảnh hưởng cả điều trị và kết quả | 38 |
| conformal prediction | dự đoán phù hợp (conformal) | khoảng/tập dự đoán có bảo đảm độ phủ | 37 |
| confusion matrix | ma trận nhầm lẫn | bảng TP, FP, FN, TN | 14 |
| contrastive learning | học tương phản | kéo gần biểu diễn của các góc nhìn cùng mẫu, đẩy xa mẫu khác | 34 |
| convex | lồi | hàm có một cực tiểu toàn cục — tối ưu dễ | 02 |
| cost-sensitive learning | học nhạy chi phí | quyết định theo chi phí của từng loại sai | 19 |
| covariance | hiệp phương sai | mức hai biến cùng biến thiên | 02 |
| covariate shift | dịch chuyển hiệp biến | P(X) đổi, P(y\|X) giữ nguyên | 05, 40 |
| cross-entropy | entropy chéo | loss cho phân loại xác suất | 07, 27 |
| cross-validation (CV) | kiểm định chéo | chia dữ liệu thành k phần, luân phiên làm validation | 15 |
| curse of dimensionality | lời nguyền chiều cao | khoảng cách mất ý nghĩa khi số chiều lớn | 05, 08 |
| data augmentation | tăng cường dữ liệu | biến đổi giữ nhãn để tạo thêm dữ liệu huấn luyện | 33 |
| data leakage | rò rỉ dữ liệu | thông tin không có lúc dự đoán lọt vào huấn luyện | 05 |
| decision boundary | biên quyết định | ranh giới giữa các vùng dự đoán khác nhau | 07 |
| decision tree | cây quyết định | mô hình chia dữ liệu bằng chuỗi câu hỏi ngưỡng | 10 |
| deep learning | học sâu | mạng nơ-ron nhiều lớp học biểu diễn | 27 |
| dimensionality reduction | giảm chiều | biểu diễn dữ liệu bằng ít chiều hơn | 21 |
| distribution shift | dịch chuyển phân phối | dữ liệu triển khai khác dữ liệu huấn luyện | 05 |
| divisive clustering | phân cụm tách dần | phân cụm phân cấp từ trên xuống (vd bisecting K-Means) | 20 |
| dropout | dropout | tắt ngẫu nhiên nơ-ron khi huấn luyện để điều chuẩn | 28 |
| DVC | DVC | quản lý phiên bản dữ liệu song song với git bằng file hash nhỏ | 39 |
| early stopping | dừng sớm | dừng huấn luyện khi validation không cải thiện | 12, 28 |
| embedding | vector nhúng | biểu diễn vector đặc của đối tượng rời rạc (từ, người dùng) | 32 |
| ensemble | tổ hợp mô hình | kết hợp nhiều mô hình | 24 |
| epistemic uncertainty | bất định nhận thức | bất định do thiếu dữ liệu/kiến thức — giảm khi thêm dữ liệu | 37 |
| epoch | epoch | một lượt đi qua toàn bộ dữ liệu huấn luyện | 27 |
| error analysis | phân tích lỗi | xem mô hình sai ở đâu, vì sao | 05 |
| exception | ngoại lệ (lỗi khi chạy) | TypeError, ValueError, KeyError…; xử lý bằng try/except đúng loại | 01 |
| expected calibration error (ECE) | sai số hiệu chỉnh kỳ vọng | trung bình chênh lệch xác suất – tần suất theo bin | 37 |
| explainability (XAI) | khả năng giải thích | hiểu vì sao mô hình đưa ra dự đoán | 36 |
| exploding gradient | gradient bùng nổ | gradient tăng vọt qua nhiều lớp/bước | 30 |
| F1 score | điểm F1 | trung bình điều hoà của precision và recall | 14 |
| Fashion-MNIST | Fashion-MNIST | 70 000 ảnh xám 28×28 của 10 loại quần áo, giày dép, túi | 07, 27, 29 |
| feature | đặc trưng | biến đầu vào của mô hình | 04 |
| feature engineering | kỹ thuật đặc trưng | tạo đặc trưng mới từ dữ liệu | 17 |
| feature importance | độ quan trọng đặc trưng | mức đóng góp của đặc trưng vào dự đoán | 11, 36 |
| feature selection | chọn đặc trưng | giữ lại tập đặc trưng hữu ích | 18 |
| feature store | kho đặc trưng | hệ thống lưu đặc trưng đúng thời điểm cho train và serve | 39 |
| feedback loop | vòng phản hồi | hành động của mô hình thay đổi dữ liệu huấn luyện tương lai | 40 |
| fine-tuning | tinh chỉnh | huấn luyện tiếp mô hình tiền huấn luyện trên dữ liệu mới | 33 |
| gradient | gradient | vector đạo hàm riêng, chỉ hướng tăng nhanh nhất | 02 |
| gradient boosting | gradient boosting | boosting khớp gradient âm của loss | 12 |
| gradient descent | hạ gradient | cập nhật tham số ngược hướng gradient | 02, 06 |
| GradientTape | GradientTape | ghi lại phép toán để TensorFlow tính gradient | 27 |
| ground truth | nhãn thật | giá trị đúng dùng để đánh giá | 05 |
| hyperparameter | siêu tham số | tham số chọn trước khi huấn luyện (độ sâu cây, learning rate) | 16 |
| imputation | điền giá trị thiếu | thay giá trị thiếu bằng ước lượng | 04 |
| inference | suy luận / dự đoán | dùng mô hình đã huấn luyện để dự đoán | 39 |
| interpretability | khả năng diễn giải | mô hình tự nó dễ hiểu (tuyến tính, cây nông) | 36 |
| IPW (inverse propensity weighting) | trọng số nghịch đảo xu hướng | cân mẫu theo 1/P(điều trị) để giả lập thí nghiệm | 38 |
| k-fold | k-fold | CV chia dữ liệu thành k phần | 15 |
| Keras | Keras | API cấp cao của TensorFlow: Sequential, Functional, compile/fit; Keras 3 chạy nhiều backend | 27 |
| kernel trick | thủ thuật kernel | tính tích vô hướng trong không gian cao chiều gián tiếp | 13 |
| k-means | k-means | phân cụm gán điểm vào tâm gần nhất | 20 |
| kNN (k-nearest neighbors) | k láng giềng gần nhất | dự đoán theo k điểm gần nhất | 08 |
| label | nhãn | giá trị mục tiêu cần dự đoán | 05 |
| label delay | độ trễ của nhãn | thời gian từ dự đoán đến khi biết kết quả thật | 40, 41 |
| Lasso (L1) | Lasso (L1) | điều chuẩn tổng trị tuyệt đối — đưa hệ số về 0 | 06 |
| latent variable | biến ẩn | biến không quan sát được trong mô hình | 23, 34 |
| layer normalization | chuẩn hoá theo lớp | chuẩn hoá trong từng mẫu (Transformer) | 31 |
| learning curve | đường cong học | hiệu năng theo lượng dữ liệu/huấn luyện | 05 |
| learning rate | tốc độ học | độ lớn bước cập nhật gradient | 02 |
| likelihood | hàm hợp lý | xác suất dữ liệu cho trước tham số | 02 |
| linear regression | hồi quy tuyến tính | dự đoán giá trị liên tục bằng tổ hợp tuyến tính | 06 |
| logistic regression | hồi quy logistic | phân loại bằng sigmoid của tổ hợp tuyến tính | 07 |
| logit | logit | log của tỉ số odds; đầu ra trước sigmoid/softmax | 07 |
| loss function | hàm mất mát | đo mức sai của dự đoán để tối ưu | 05 |
| LSTM | LSTM | RNN có cổng, nhớ được phụ thuộc dài | 30 |
| MAE / MSE / RMSE | sai số tuyệt đối / bình phương trung bình | metric hồi quy | 14 |
| MAP (maximum a posteriori) | cực đại hậu nghiệm | MLE cộng thêm prior — tương đương điều chuẩn | 02 |
| map-reduce | map-reduce | tính từng phần độc lập (map) rồi gộp (reduce) — nền tảng của song song dữ liệu | 06, 39 |
| matrix factorization | phân rã ma trận | xấp xỉ ma trận đánh giá bằng tích hai ma trận nhỏ | 26 |
| MLE (maximum likelihood estimation) | ước lượng hợp lý cực đại | chọn tham số làm dữ liệu quan sát có xác suất lớn nhất | 02 |
| MLOps | MLOps | thực hành vận hành mô hình: CI/CD, giám sát, huấn luyện lại | 40 |
| model card | thẻ mô hình | tài liệu mục đích, dữ liệu, hiệu năng, hạn chế của mô hình | 39 |
| model registry | kho mô hình | lưu phiên bản mô hình, alias champion, rollback | 39 |
| momentum | quán tính | tích luỹ gradient quá khứ để tăng tốc | 28 |
| multiclass SVM loss | hàm mất mát SVM đa lớp | phạt khi điểm lớp sai không thấp hơn lớp đúng ít nhất một lề Δ | 07, 13 |
| multicollinearity | đa cộng tuyến | đặc trưng tương quan mạnh với nhau | 06 |
| Naive Bayes | Naive Bayes | phân loại Bayes giả định đặc trưng độc lập có điều kiện | 09 |
| nested cross-validation | CV lồng nhau | vòng trong chọn siêu tham số, vòng ngoài đánh giá | 15 |
| neural forecasting | dự báo bằng mạng nơ-ron | mô hình chuỗi thời gian dùng mạng nơ-ron (DeepAR, N-BEATS, TFT); mạnh khi có nhiều chuỗi | 25 |
| neural network | mạng nơ-ron | hàm hợp của các lớp tuyến tính và phi tuyến | 27 |
| normalization / standardization | chuẩn hoá | đưa đặc trưng về cùng thang (min-max, z-score) | 04 |
| one-cycle schedule | lịch learning rate one-cycle | tăng rồi giảm learning rate trong một chu kỳ huấn luyện | 29 |
| one-hot encoding | mã hoá one-hot | biến danh mục thành vector nhị phân | 04 |
| online inference | suy luận trực tuyến | dự đoán theo từng yêu cầu, cần độ trễ thấp | 39, 40 |
| online learning | học trực tuyến | cập nhật mô hình theo từng lô dữ liệu mới (partial_fit) thay vì train lại | 06 |
| out-of-fold (OOF) | ngoài fold | dự đoán cho mẫu bởi mô hình không học trên mẫu đó | 24 |
| outlier | ngoại lai | giá trị khác thường so với phần lớn dữ liệu | 04 |
| overfitting | quá khớp | học thuộc nhiễu của train, kém trên dữ liệu mới | 05 |
| parameter | tham số | giá trị mô hình học được (trọng số) | 05 |
| PCA | phân tích thành phần chính | chiếu dữ liệu lên các hướng phương sai lớn nhất | 21 |
| permutation importance | độ quan trọng hoán vị | hiệu năng giảm bao nhiêu khi xáo trộn một đặc trưng | 36 |
| pipeline | pipeline | chuỗi bước tiền xử lý + mô hình fit cùng nhau | 04 |
| point-in-time correctness | đúng thời điểm | đặc trưng chỉ dùng dữ liệu có trước thời điểm dự đoán | 39 |
| policy | chính sách | (RL) quy tắc chọn hành động theo trạng thái | 35 |
| positional encoding | mã hoá vị trí | thêm thông tin thứ tự vào Transformer | 31 |
| posterior / prior | hậu nghiệm / tiên nghiệm | phân phối tham số sau / trước khi thấy dữ liệu | 02, 23 |
| precision / recall | độ chính xác dương / độ phủ | TP/(TP+FP) / TP/(TP+FN) | 14 |
| propensity score | điểm xu hướng | xác suất được điều trị cho trước đặc trưng | 38 |
| PSI (population stability index) | chỉ số ổn định quần thể | đo độ trôi phân phối giữa hai tập | 40 |
| Q-learning | Q-learning | học giá trị hành động off-policy bằng TD | 35 |
| quantile regression | hồi quy phân vị | dự đoán phân vị thay vì trung bình | 37 |
| random forest | rừng ngẫu nhiên | bagging cây + chọn ngẫu nhiên đặc trưng mỗi lần chia | 11 |
| recommender system | hệ gợi ý | gợi ý món phù hợp cho người dùng | 26 |
| regularization | điều chuẩn | phạt độ phức tạp để giảm overfitting | 06 |
| reinforcement learning | học tăng cường | học hành động qua phần thưởng từ môi trường | 35 |
| reproducibility | khả năng tái lập | người khác chạy lại ra cùng kết quả | 39 |
| residual | phần dư | y − ŷ | 06 |
| residual connection | kết nối tắt | y = x + F(x) — giúp huấn luyện mạng sâu | 29 |
| Ridge (L2) | Ridge (L2) | điều chuẩn tổng bình phương hệ số | 06 |
| RNN | mạng nơ-ron hồi quy | mạng xử lý chuỗi với trạng thái ẩn | 30 |
| ROC curve | đường ROC | TPR theo FPR khi đổi ngưỡng | 14 |
| sample | mẫu | một quan sát (một dòng dữ liệu) | 05 |
| self-attention | tự chú ý | attention trong cùng một chuỗi | 31 |
| self-supervised learning | học tự giám sát | tự tạo nhãn từ dữ liệu (che, dự đoán phần khác) | 34 |
| SGD (stochastic gradient descent) | hạ gradient ngẫu nhiên | cập nhật theo từng mẫu/lô nhỏ | 06 |
| shadow deployment | triển khai bóng | bản mới chạy song song, không ảnh hưởng người dùng | 40 |
| SHAP | SHAP | giá trị Shapley giải thích đóng góp của từng đặc trưng | 36 |
| shortcut learning | học đường tắt | mô hình dựa vào tín hiệu giả tương quan với nhãn | 33, project 07 |
| sigmoid | hàm sigmoid | σ(z) = 1/(1 + e⁻ᶻ), ánh xạ về (0, 1) | 07 |
| silhouette score | hệ số silhouette | đo cụm chặt và tách biệt | 20 |
| softmax | softmax | biến vector logit thành phân phối xác suất | 07 |
| stacking | stacking | mô hình tầng 2 học trên dự đoán OOF của mô hình tầng 1 | 24 |
| stationarity | tính dừng | thống kê của chuỗi không đổi theo thời gian | 25 |
| stratified split | chia phân tầng | giữ tỉ lệ lớp khi chia dữ liệu | 15 |
| supervised learning | học có giám sát | học từ dữ liệu có nhãn | 05 |
| support vector machine (SVM) | máy vector hỗ trợ | phân loại với lề cực đại | 13 |
| target encoding | mã hoá theo mục tiêu | thay danh mục bằng trung bình nhãn (phải làm OOF) | 17 |
| temporal split | chia theo thời gian | train quá khứ, test tương lai | 15 |
| TensorFlow | TensorFlow | framework deep learning của Google: tf.Tensor, tf.Variable, tf.function, tf.data | 27 |
| test set | tập kiểm tra | dữ liệu đánh giá cuối cùng, chạm một lần | 05 |
| threshold | ngưỡng | giá trị xác suất để quyết định nhãn | 14 |
| tokenization | tách token | chia văn bản thành đơn vị (từ, âm tiết, subword) | 32 |
| traceback | traceback | chuỗi lời gọi hàm dẫn tới lỗi — đọc từ dưới lên | 01 |
| training–serving skew | lệch huấn luyện – phục vụ | tiền xử lý lúc phục vụ khác lúc huấn luyện | 39 |
| transfer learning | học chuyển giao | dùng mô hình học trên dữ liệu lớn cho bài toán mới | 33 |
| Transformer | Transformer | kiến trúc dựa hoàn toàn vào attention | 31 |
| t-SNE / UMAP | t-SNE / UMAP | giảm chiều phi tuyến để trực quan hoá | 21 |
| underfitting | chưa khớp | mô hình quá đơn giản, kém cả trên train | 05 |
| unsupervised learning | học không giám sát | tìm cấu trúc trong dữ liệu không nhãn | 05, 20 |
| uplift modeling | mô hình hoá tác động tăng thêm | dự đoán ai thay đổi hành vi nhờ can thiệp | 38 |
| validation set | tập kiểm định | dữ liệu chọn mô hình/siêu tham số | 05 |
| vanishing gradient | gradient tiêu biến | gradient nhỏ dần qua nhiều lớp/bước | 28, 30 |
| VAE (variational autoencoder) | autoencoder biến phân | autoencoder sinh dữ liệu với không gian ẩn xác suất | 34 |
| variance | phương sai | (thống kê) độ phân tán; (mô hình) độ nhạy với dữ liệu huấn luyện | 02, 05 |
| weight decay | suy giảm trọng số | điều chuẩn L2 trong optimizer | 28 |
| winner's curse | lời nguyền người thắng | điểm của cấu hình tốt nhất trong nhiều cấu hình bị lạc quan | 39 |
