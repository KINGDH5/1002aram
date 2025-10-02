# -*- coding: utf-8 -*-
"""
Scenario1: 모델 학습 + 피처 중요도 저장 (CSV/PNG)
"""
import os, pickle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ml import read_csv_safe, train_models

BASE_DIR = Path(".")
CSV_PATH = BASE_DIR / "renamed_data.csv"
OUT_DIR  = BASE_DIR / "models"; OUT_DIR.mkdir(exist_ok=True)
PKL_PATH = OUT_DIR / "scenario1_model.pkl"

# 1) 데이터 & 모델 학습
df = read_csv_safe(CSV_PATH)
bundle = train_models(df)  # dict 반환 가정
synergy, champ, stat = bundle.get("synergy_model"), bundle.get("champ_model"), bundle.get("stat_model")

# 2) 피처 중요도 저장 함수
def save_fi(name, model, feats):
    if not hasattr(model, "feature_importances_"): return
    imp = model.feature_importances_
    df = pd.DataFrame({"feature": feats, "importance": imp}).sort_values("importance", ascending=False)
    df.to_csv(OUT_DIR/f"feature_importance_{name}.csv", index=False, encoding="utf-8-sig")
    df.head(30).plot(kind="barh", x="feature", y="importance", legend=False)
    plt.gca().invert_yaxis(); plt.tight_layout()
    plt.savefig(OUT_DIR/f"feature_importance_{name}.png", dpi=200); plt.close()

# 3) 각 모델 중요도 저장
save_fi("synergy", synergy, bundle.get("mlb").classes_)
save_fi("champ", champ, bundle.get("champ_profile").columns)
save_fi("stat_tag", stat, bundle.get("feature_cols"))

# 4) 번들 저장
with open(PKL_PATH, "wb") as f: pickle.dump(bundle, f)
print(f"[DONE] 모델 & 중요도 저장 완료 → {OUT_DIR.resolve()}")
