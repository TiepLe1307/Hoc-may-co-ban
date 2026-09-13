from sklearn.ensemble import RandomForestClassifier

def build_random_forest_model():
    # Khởi tạo mô hình Random Forest (Ensemble Learning) với 100 cây quyết định
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model
