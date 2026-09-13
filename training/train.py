import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from app.preprocessing import load_and_preprocess_data
from app.model import build_random_forest_model

def main():
    # 1. Load và tiền xử lý dữ liệu
    X_train, X_test, y_train, y_test, scaler, target_names = load_and_preprocess_data()

    # 2. Huấn luyện mô hình Random Forest
    model = build_random_forest_model()
    model.fit(X_train, y_train)

    # 3. Đánh giá độ chính xác
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"=== ĐỘ CHÍNH XÁC CỦA MÔ HÌNH ENSEMBLE LEARNING (RANDOM FOREST): {accuracy * 100:.2f}% ===")
    print("\nBáo cáo chi tiết:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 4. Thử nghiệm dự đoán 1 mẫu hoa mới
    sample_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
    sample_scaled = scaler.transform(sample_flower)
    prediction = model.predict(sample_scaled)
    print(f"\n-> Kết quả dự đoán loài hoa: {target_names[prediction[0]]}")

if __name__ == "__main__":
    main()
