# -*- coding: utf-8 -*-
"""
시나리오1: 모델 학습 후 피처 중요도 CSV/PNG 저장 + 콘솔 요약 출력
- 입력: renamed_data.csv (동일 폴더)
- 출력: models/feature_importance_{synergy|champ|stat_tag}.csv / .png
- 추가: models/scenario1_model.pkl (모델/스키마 번들 저장)
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ---- Matplotlib: GUI 없이 파일만 저장 (Qt 오류 방지) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import pickle

# 프로젝트 유틸/모델 (사용자 제공 ml.py)
from ml import read_csv_safe, train_models

# ----------------------------------
# 경로/파일
# ----------------------------------
BASE_DIR = Path(".")
CSV_PATH = BASE_DIR / "renamed_data.csv"
OUT_DIR  = BASE_DIR / "models"
OUT_DIR.mkdir(exist_ok=True)

PKL_PATH = OUT_DIR / "scenario1_model.pkl"

# ----------------------------------
# 데이터 로드
# ----------------------------------
assert CSV_PATH.exists(), f"CSV가 없습니다: {CSV_PATH.resolve()}"
df = read_csv_safe(CSV_PATH) if 'read_csv_safe' in globals() else pd.read_csv(CSV_PATH)

# ----------------------------------
# 모델 학습 (ml.py의 파이프라인 재사용)
# 기대 반환(튜플): (synergy_model, champ_model, mlb, champ_profile,
#                  stat_model, scaler, feature_cols, vectorizer, df_ref, champ_cols)
# 혹은 dict 형태일 수도 있어 안전 처리
# ----------------------------------
ret = train_models(df)

def _as_dict(ret_obj):
    if isinstance(ret_obj, dict):
        return ret_obj
    # tuple → dict로 맵핑 시도 (미정인 경우 대비)
    keys = [
        "synergy_model","champ_model","mlb","champ_profile",
        "stat_model","scaler","feature_cols","vectorizer","df_ref","champ_cols"
    ]
    try:
        return {k: v for k, v in zip(keys, ret_obj)}
    except Exception:
        return {}

bundle = _as_dict(ret)

synergy_model = bundle.get("synergy_model", None)
champ_model   = bundle.get("champ_model",   None)
mlb           = bundle.get("mlb",           None)
champ_profile = bundle.get("champ_profile", None)
stat_model    = bundle.get("stat_model",    None)
feature_cols  = bundle.get("feature_cols",  None)

# ----------------------------------
# 유틸: 중요도 표/그림 저장
# ----------------------------------
def to_df(features, importances, top_n=50):
    if features is None or importances is None:
        return pd.DataFrame(columns=["feature","importance"]), pd.DataFrame(columns=["feature","importance"])
    s = pd.DataFrame({"feature": list(features), "importance": list(importances)})
    s = s.sort_values("importance", ascending=False)
    return s, s.head(top_n)

def save_csv_png(name, df_full, df_top, title):
    csv_path = OUT_DIR / f"feature_importance_{name}.csv"
    png_path = OUT_DIR / f"feature_importance_{name}.png"
    df_full.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 10))
    df_top.sort_values("importance", ascending=True).plot(
        kind="barh", x="feature", y="importance", legend=False
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"[OK] {name}: CSV={csv_path.name}, PNG={png_path.name}")

def _safe_feature_names_from_model(model, n_features):
    # XGBoost / LightGBM 등에서 feature_names를 가져오되, 실패 시 f0.. 생성
    names = None
    try:
        # xgboost.XGB* 계열
        booster = getattr(model, "get_booster", lambda: None)()
        if booster is not None:
            names = booster.feature_names
    except Exception:
        names = None
    if names is None:
        try:
            names = getattr(model, "feature_name_", None)  # LightGBM
        except Exception:
            names = None
    if not names:
        names = [f"f{i}" for i in range(n_features)]
    return names

# ----------------------------------
# 1) Synergy 모델 중요도
# ----------------------------------
if synergy_model is not None:
    imp = getattr(synergy_model, "feature_importances_", None)
    # 기본 피처: MultiLabelBinarizer 클래스(챔피언 원-핫)
    if mlb is not None and hasattr(mlb, "classes_"):
        feats = list(mlb.classes_)
    else:
        n = 0 if imp is None else len(imp)
        feats = _safe_feature_names_from_model(synergy_model, n)
    synergy_df, synergy_top = to_df(feats, imp)
    save_csv_png("synergy", synergy_df, synergy_top, "Scenario1 Feature Importance - Synergy (Top 50)")
else:
    print("[WARN] synergy_model 을 찾지 못했습니다. (ml.py 반환 확인 필요)")

# ----------------------------------
# 2) Champ-wise 모델 중요도 (챔피언 프로파일 수치 피처)
# ----------------------------------
if champ_model is not None:
    imp = getattr(champ_model, "feature_importances_", None)
    if champ_profile is not None and isinstance(champ_profile, pd.DataFrame):
        feats = [c for c in champ_profile.columns if c.lower() not in ("champion","label","target","y")]
    else:
        n = 0 if imp is None else len(imp)
        feats = _safe_feature_names_from_model(champ_model, n)
    champ_df, champ_top = to_df(feats, imp)
    save_csv_png("champ", champ_df, champ_top, "Scenario1 Feature Importance - Champ-wise (Top 50)")
else:
    print("[WARN] champ_model 을 찾지 못했습니다. (ml.py 반환 확인 필요)")

# ----------------------------------
# 3) Stat/Tag 모델 중요도
# ----------------------------------
if stat_model is not None:
    imp = getattr(stat_model, "feature_importances_", None)
    feats = feature_cols if feature_cols is not None else _safe_feature_names_from_model(stat_model, 0 if imp is None else len(imp))
    stat_df, stat_top = to_df(feats, imp)
    save_csv_png("stat_tag", stat_df, stat_top, "Scenario1 Feature Importance - Stat/Tag (Top 50)")
else:
    print("[WARN] stat_model 을 찾지 못했습니다. (ml.py 반환 확인 필요)")

# ----------------------------------
# 콘솔 요약 출력
# ----------------------------------
def _print_head(df, title):
    print(f"\n[{title} Top10]")
    if df is None or df.empty:
        print("(없음)")
    else:
        print(df.head(10).to_string(index=False))

try:
    _print_head(synergy_top, "Synergy")
except NameError:
    pass
try:
    _print_head(champ_top, "Champ-wise")
except NameError:
    pass
try:
    _print_head(stat_top, "Stat/Tag")
except NameError:
    pass

# ----------------------------------
# 번들 저장(나중에 재사용)
# ----------------------------------
save_bundle = {
    "synergy_model": synergy_model,
    "champ_model":   champ_model,
    "stat_model":    stat_model,
    "mlb":           mlb,
    "champ_profile": champ_profile,
    "feature_cols":  feature_cols,
}
with open(PKL_PATH, "wb") as f:
    pickle.dump(save_bundle, f)
print(f"\n[DONE] 모델 번들 저장: {PKL_PATH.resolve()}")
