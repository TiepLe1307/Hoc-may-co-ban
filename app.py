# =========================================================
# BÀI TẬP MÔ HÌNH HỌC MÁY CƠ BẢN
# ĐỀ TÀI: ENSEMBLE LEARNING (RANDOM FOREST CLASSIFIER)
# SV: LÊ TIẾN TIỆP - MSV: 10123314 - STT: 36
# =========================================================

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier  # Mô hình Ensemble Learning
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Tải bộ dữ liệu hoa Iris
iris = load_iris()
X = iris.data
y = iris.target

# 2. Chia tập dữ liệu 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Huấn luyện mô hình Ensemble Learning (Random Forest)
# Random Forest kết hợp nhiều Cây quyết định (Decision Trees) lại với nhau
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 5. Đánh giá độ chính xác
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(
    f"=== ĐỘ CHÍNH XÁC CỦA MÔ HÌNH ENSEMBLE LEARNING (RANDOM FOREST): {accuracy * 100:.2f}% ==="
)
print("\nBáo cáo chi tiết:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 6. Thử nghiệm dự đoán
sample_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
sample_scaled = scaler.transform(sample_flower)
prediction = rf_model.predict(sample_scaled)

print(
    f"\n-> Kết quả dự đoán bằng Ensemble Learning: {iris.target_names[prediction[0]]}"
)
