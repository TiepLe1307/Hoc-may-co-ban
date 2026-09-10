# =========================================================
# BÀI TẬP MÔ HÌNH HỌC MÁY CƠ BẢN: PHÂN LOẠI HOA IRIS
# =========================================================

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Tải bộ dữ liệu hoa Iris
iris = load_iris()
X = iris.data  # 4 đặc trưng: chiều dài/rộng đài hoa và cánh hoa
y = iris.target  # Nhãn loài hoa (0, 1, 2)

# 2. Chia tập dữ liệu: 80% để huấn luyện (train), 20% để kiểm thử (test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Chuẩn hóa dữ liệu (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Khởi tạo và huấn luyện mô hình Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Đánh giá độ chính xác của mô hình
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"=== ĐỘ CHÍNH XÁC CỦA MÔ HÌNH: {accuracy * 100:.2f}% ===")
print("\nBáo cáo chi tiết:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))


# 6. Thử nghiệm dự đoán một bông hoa mới
# Giả sử bông hoa có kích thước: [sepal_length, sepal_width, petal_length, petal_width]
sample_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
sample_scaled = scaler.transform(sample_flower)
prediction = model.predict(sample_scaled)

print(
    f"\n-> Kết quả dự đoán bông hoa mẫu: {iris.target_names[prediction[0]]}"
)
