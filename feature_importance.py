# feature_importance.py
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def run_feature_importance(csv_path, scenario_name):
    print(f"\n===== {scenario_name} =====")

    # CSV 불러오기
    df = pd.read_csv(csv_path)

    # y = win 컬럼, X = 나머지
    if "win" not in df.columns:
        raise ValueError("CSV에 'win' 컬럼이 없습니다. 타겟(y) 컬럼이 필요합니다.")

    X = df.drop(columns=["win"])
    y = df["win"]

    # ❗ 문자열(object) 컬럼 제거
    obj_cols = X.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0:
        print(f"제거된 문자열 컬럼들: {list(obj_cols)}")
        X = X.drop(columns=obj_cols)

    # 데이터 분할
    X_train, X_test, y_train = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # XGBoost 분류기 학습
    model = XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    model.fit(X_train, y_train)

    # 피처 중요도 출력
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]

    print("\n--- 상위 30개 피처 중요도 ---")
    for idx in sorted_idx[:30]:
        print(f"{X.columns[idx]}: {importances[idx]:.4f}")

# --------------------------------------------------
# 실행부
# --------------------------------------------------
if __name__ == "__main__":
    csv_path = r"C:\Users\권도혁\Desktop\시나리오1,2\renamed_data.csv"
    run_feature_importance(csv_path, "시나리오1")
    run_feature_importance(csv_path, "시나리오2")
