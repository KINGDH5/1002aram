# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")  # GUI 없이 저장
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

IN_DIR  = Path("models")
OUT_DIR = Path("models"); OUT_DIR.mkdir(exist_ok=True)

# Windows 한글 폰트(맑은 고딕) — 필요 없으면 이 두 줄 주석 처리
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False  # 음수 깨짐 방지

def plot_importance(csv_name: str, title: str, top: int = 20):
    csv_path = IN_DIR / csv_name
    assert csv_path.exists(), f"파일 없음: {csv_path}"

    df = pd.read_csv(csv_path)
    # 안전 정렬
    df = df.sort_values("importance", ascending=False).head(top)

    # 그리기(가로 막대)
    plt.figure(figsize=(10, 8))
    ax = df.sort_values("importance").plot(
        kind="barh", x="feature", y="importance", legend=False
    )
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    # 값 라벨(막대 끝에 중요도 숫자 표기)
    for p in ax.patches:
        w = p.get_width()
        ax.text(w, p.get_y() + p.get_height()/2, f"{w:.3f}",
                va="center", ha="left")

    plt.tight_layout()
    out_png = OUT_DIR / (csv_name.replace(".csv", f"_top{top}.png"))
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"[OK] {out_png.name}")

if __name__ == "__main__":
    # 필요에 따라 top 개수 조절
    plot_importance("feature_importance_champ.csv",    "Champ-wise Feature Importance (Top 20)", top=20)
    plot_importance("feature_importance_synergy.csv",  "Synergy Feature Importance (Top 20)",    top=20)
    plot_importance("feature_importance_stat_tag.csv", "Stat/Tag Feature Importance (Top 20)",   top=20)
