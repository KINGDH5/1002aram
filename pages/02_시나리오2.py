# -*- coding: utf-8 -*-
# Scenario2 Page: 룬/역할 감지 → 아이템 추천 (chdir/runpy 없이 동작)

import sys, os, base64, json, io
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image

# =========================
# 0) 안전 임포트 설정
# =========================
ROOT = Path(__file__).resolve().parents[1]        # 레포 루트
CANDIDATES = [ROOT, ROOT / "시나리오2"]           # 모듈이 있을 법한 경로

for p in CANDIDATES:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

missing = []
if not (ROOT / "item_recommender.py").exists() and not (ROOT / "시나리오2" / "item_recommender.py").exists():
    missing.append("item_recommender.py")
if not (ROOT / "rune_champion.py").exists() and not (ROOT / "시나리오2" / "rune_champion.py").exists():
    missing.append("rune_champion.py")

if missing:
    st.set_page_config(page_title="Scenario2 | Missing files", layout="wide")
    st.error("필수 파일이 레포에 없습니다: " + ", ".join(missing))
    st.caption(f"검색한 경로: {ROOT} / {ROOT/'시나리오2'}")
    try:
        st.write("루트 목록:", sorted([p.name for p in ROOT.iterdir()]))
        if (ROOT / "시나리오2").exists():
            st.write("시나리오2 목록:", sorted([p.name for p in (ROOT / '시나리오2').iterdir()]))
    except Exception:
        pass
    st.stop()

# 임포트 (경로 주입 후)
import item_recommender as ir
import rune_champion as rc

# APP_BASE 환경변수(있으면 모듈에서 사용하도록)
os.environ.setdefault("APP_BASE", str(ROOT))

# =========================
# 1) Vertex Endpoint (S2)
# =========================
from google.cloud import aiplatform
from google.oauth2 import service_account

@st.cache_resource
def get_endpoint_s2():
    s2 = st.secrets["SCENARIO2"]
    cred_info = json.loads(base64.b64decode(s2["GOOGLE_APPLICATION_CREDENTIALS_B64"]))
    creds = service_account.Credentials.from_service_account_info(cred_info)
    aiplatform.init(
        project=s2["PROJECT_ID"],
        location=s2["REGION"].strip().lower(),
        credentials=creds,
        api_endpoint=f"{s2['REGION'].strip().lower()}-aiplatform.googleapis.com",
    )
    return aiplatform.Endpoint(s2["ENDPOINT_ID"])

# =========================
# 2) UI
# =========================
st.set_page_config(page_title="Scenario2 | 룬/역할 감지 · 아이템 추천", layout="wide")
st.title("⭐ Scenario2 — 룬/역할 감지 → 아이템 추천")

# --- 2.1 스크린샷 업로드 & 감지 ---
st.subheader("1) 스크린샷 업로드")
uploaded = st.file_uploader("픽 화면 스크린샷 (png/jpg)", type=["png","jpg","jpeg"])
threshold = st.slider("신뢰도(%)", 50, 95, 70, 1)

detected_current, detected_bench, overlay_img = [], [], None
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="업로드 이미지", use_container_width=True)

    with st.spinner("룬/역할 감지 중..."):
        ep = get_endpoint_s2()
        # rc.predict_image(endpoint, image, threshold=...) 시그니처 가정
        cur, bench, overlay = rc.predict_image(ep, img, threshold=threshold)

    detected_current = [c for c in (cur or []) if c][:5]
    detected_bench   = [b for b in (bench or []) if b][:10]
    overlay_img = overlay
    if overlay_img is not None:
        st.image(overlay_img, caption="탐지 영역", use_container_width=True)

# --- 2.2 내 챔피언/상대 조합 ---
st.subheader("2) 내 챔피언 선택 & 상대 조합")
# 전체 챔프 목록 헬퍼가 있으면 활용 (없으면 감지값 기반)
all_champs = None
if hasattr(ir, "list_all_champs"):
    try:
        all_champs = ir.list_all_champs()
    except Exception:
        all_champs = None

fallback_pool = sorted(set(detected_current + detected_bench))
options = all_champs if all_champs else (fallback_pool if fallback_pool else [])
if not options:
    st.info("먼저 스크린샷을 업로드하면 감지된 챔피언으로 목록이 채워집니다.")

my_champ = st.selectbox("내 챔피언", options=options) if options else None
enemy_pool = st.multiselect(
    "상대 챔피언(최대 10명, 편집 가능)",
    options=all_champs if all_champs else options,
    default=detected_bench if detected_bench else []
)

# =========================
# 3) 아이템 추천
# =========================
st.subheader("3) 아이템 추천")

def try_recommend(champ, enemies):
    """
    item_recommender의 공개 함수명이 레포마다 다를 수 있어
    아래 후보 이름들을 순차적으로 시도한다.
    반환 형식도 dict / list / tuple 모두 허용.
    """
    if not champ or not enemies:
        return None

    candidates = [
        "recommend_items",
        "recommend",
        "get_recommendations",
        "recommend_for",           # 혹시 다른 프로젝트에서 썼다면
    ]
    for fn_name in candidates:
        if hasattr(ir, fn_name):
            fn = getattr(ir, fn_name)
            try:
                return fn(champ, enemies)
            except TypeError:
                # 인자 시그니처 불일치 → 다음 후보 시도
                continue
            except Exception as e:
                st.warning(f"{fn_name} 실행 중 오류: {e}")
                continue
    return None

if st.button("아이템 추천 실행", type="primary", disabled=not (my_champ and enemy_pool)):
    with st.spinner("추천 계산 중..."):
        result = try_recommend(my_champ, enemy_pool)
        items, reason = None, None

        if isinstance(result, dict):
            items  = result.get("items") or result.get("recommendations") or result.get("result")
            reason = result.get("reason") or result.get("explanation")
        elif isinstance(result, (list, tuple)):
            items = list(result)
        else:
            items = result

    if items:
        st.success("추천 아이템")
        st.dataframe(pd.DataFrame({"추천 아이템": items}), use_container_width=True)
        if reason:
            st.caption(reason)
    else:
        st.info("추천 결과가 비어있습니다. `item_recommender.py`의 공개 함수명/시그니처를 알려주면 정확히 맞춰 드릴게요.")

st.caption(f"PYTHONPATH search: {', '.join([str(p) for p in CANDIDATES])}")
