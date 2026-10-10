"""ml_course — thư viện hỗ trợ cho khoá "Machine Learning: Zero → Advanced".

Các module:
    course          cấu trúc khoá học (phase, chapter, thứ tự học)
    datasets        bộ dữ liệu tổng hợp có "ground truth" để học (không cần tải mạng)
    real_datasets   bộ dữ liệu ảnh thật tải một lần, cache trong data/raw (Fashion-MNIST, CIFAR-10)
    preprocessing   scaler/encoder/split viết from scratch
    metrics         metric viết from scratch (đối chiếu với sklearn trong tests/)
    validation      K-Fold, Stratified, TimeSeriesSplit, bootstrap from scratch
    visualization   các hàm vẽ dùng lại nhiều lần (decision boundary, loss surface, ...)
    utils           seed, timer, numerical gradient, kiểm tra shape
    checks          hàm check_* in ✅/❌ dùng trong bài tập
    checkpoint      engine chấm quiz + gate PASS/REVIEW
    progress        lưu & hiển thị tiến độ học
    algorithms      phiên bản tham chiếu của các thuật toán from scratch
"""

__version__ = "1.0.0"
