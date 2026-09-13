# BÀI TẬP MÔ HÌNH HỌC MÁY CƠ BẢN

**Đề tài:** Ensemble Learning (Random Forest Classifier)  
**Sinh viên thực hiện:** Lê Tiến Tiệp  
**Mã sinh viên:** 10123314  
**STT:** 36  
**Lớp:** 12523W.2  

---

## 1. Cấu Trúc Dự Án

```text
.
├── app/
│   ├── preprocessing.py   # Tải dữ liệu Iris, chia tập Train/Test và chuẩn hóa dữ liệu
│   └── model.py           # Xây dựng mô hình Random Forest Classifier (Ensemble Learning)
└── training/
    └── train.py           # Huấn luyện mô hình, đánh giá độ chính xác và thử nghiệm dự đoán

## 2. Cài Đặt Thư Viện

Trước khi chạy chương trình, hãy đảm bảo bạn đã cài đặt các thư viện cần thiết bằng lệnh:

```bash
pip install scikit-learn pandas numpy

## 3. Hướng Dẫn Chạy Chương Trình

Mở cửa sổ Terminal / Command Prompt tại thư mục gốc của dự án và chạy câu lệnh sau:

```bash
python training/train.py

## 4. Kết Quả Thực Nghiệm

- **Bộ dữ liệu:** Iris Dataset (150 mẫu, 4 thuộc tính đặc trưng)
- **Thuật toán:** Random Forest Classifier (Số lượng cây `n_estimators = 100`)
- **Kết quả đánh giá:** Mô hình đạt độ chính xác (Accuracy) **100%** trên tập test, các chỉ số Precision, Recall và F1-Score đạt mức tối ưu (1.0).
