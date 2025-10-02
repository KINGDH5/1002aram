# viz_feature_importance.py (시나리오1 피처 중요도 시각화 - 경로 수정)
import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def plot_feature_importance(csv_path: str, top_n: int = 20):
    df = pd.read_csv(csv_path)
    if "feature" not in df.columns or "importance" not in df.columns:
        raise ValueError("CSV에 feature/importance 컬럼이 필요합니다.")

    df = df.sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    plt.barh(df["feature"], df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top Feature Importances")
    plt.gca().invert_yaxis()

    # 저장
    out_path = os.path.join(".", "models", "feature_importance.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path

def main():
    st.title("Feature Importance 시각화")

    csv_path = st.text_input("피처 중요도 CSV 경로 입력", value="./models/feature_importance.csv")
    if not os.path.exists(csv_path):
        st.error("CSV 파일을 찾을 수 없습니다.")
        st.stop()

    out_path = plot_feature_importance(csv_path)
    st.image(out_path, caption="Feature Importance (Top 20)", use_container_width=True)

if __name__ == "__main__":
    main()
