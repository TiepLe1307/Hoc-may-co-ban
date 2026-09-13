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
