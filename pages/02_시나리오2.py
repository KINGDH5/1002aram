# -*- coding: utf-8 -*-
# Scenario2 Page: chdir / runpy / sys.path hack 없이 바로 렌더링
import io, os, base64, json
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image

# ---- 프로젝트 루트 추정 & 환경변수 (item_recommender가 참조할 수 있게) ----
APP_BASE = str(Path(__file__).resolve().parents[1])  # 레포 루트
os.environ["APP_BASE"] = APP_BASE  # item_recommender.py가 참조하도록(있다면)

# ---- 외부 모듈 임포트 ----
# 이 두 파일은 레포 루트에 있다고 가정 (이미 올려준 구조 기준)
import item_recommender as ir
import rune_champion as rc

# ---- Vertex Endpoint 초기화(시나리오2 시크릿 사용) ----
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

st.set_page_config(page_title="Scenario2 | 룬/역할 감지 · 아이템 추천", layout="wide")
st.title("⭐ Scenario2 — 룬/역할 감지 → 아이템 추천")

# =========================
# 1) 스크린샷 업로드 & 감지
# =========================
st.subheader("1) 스크린샷 업로드")
uploaded = st.file_uploader("픽 화면 스크린샷 (png/jpg)", type=["png","jpg","jpeg"])
threshold = st.slider("신뢰도(%)", 50, 95, 70, 1)

detected_current, detected_bench = [], []
overlay_img = None

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="업로드 이미지", use_container_width=True)

    with st.spinner("룬/역할 감지 중..."):
        # rune_champion 모듈에 predict_image가 있다면 그대로 사용
        if hasattr(rc, "predict_image") and hasattr(rc, "init_vertex"):
            ep = get_endpoint_s2()
            cur, bench, overlay = rc.predict_image(ep, img, threshold=threshold)
        else:
            # 혹시 함수명이 다르면, image.py와 동일 시그니처를 가정하고 사용
            ep = get_endpoint_s2()
            cur, bench, overlay = rc.predict_image(ep, img, threshold=threshold)  # 같은 이름 가정

    detected_current = [c for c in (cur or []) if c][:5]
    detected_bench   = [b for b in (bench or []) if b][:10]
    overlay_img = overlay

    if overlay_img is not None:
        st.image(overlay_img, caption="탐지 영역", use_container_width=True)

# =========================
# 2) 조합 요약 & 내 챔피언 선택
# =========================
st.subheader("2) 내 챔피언 선택")
# item_recommender에서 전체 챔피언 목록을 노출하는 헬퍼가 있으면 사용
all_champs = None
if hasattr(ir, "list_all_champs"):
    try:
        all_champs = ir.list_all_champs()  # 존재한다면 사용
    except Exception:
        pass

# fallback: 감지된 챔피언 합치기
fallback_pool = sorted(set(detected_current + detected_bench))
options = all_champs if all_champs else fallback_pool
my_champ = st.selectbox("내 챔피언", options=options if options else ["(먼저 스크린샷을 올리세요)"])

st.subheader("상대 조합(감지값 기반, 편집 가능)")
enemy_pool = st.multiselect(
    "상대 챔피언(최대 10명)",
    options=all_champs if all_champs else fallback_pool,
    default=detected_bench
)

# =========================
# 3) 아이템 추천
# =========================
st.subheader("3) 아이템 추천")
recommendations = None
explanation = None

def try_recommend():
    """
    item_recommender의 함수명이 레포마다 다를 수 있으니, 여러 후보를 순차적으로 시도.
    - recommend_items(champion, enemy_list)
    - recommend(champion, enemy_list)
    - get_recommendations(champion, enemy_list)
    필요 시 상황키 등을 내부에서 자동 생성한다고 가정.
    """
    # 가장 흔한 이름들 시도
    for fn_name in ["recommend_items", "recommend", "get_recommendations"]:
        if hasattr(ir, fn_name):
            fn = getattr(ir, fn_name)
            try:
                return fn(my_champ, enemy_pool)
            except TypeError:
                # (champ, enemies) 외 시그니처면 다음 후보
                continue
            except Exception as e:
                st.warning(f"{fn_name} 실행 중 오류: {e}")
                continue
    return None

if st.button("아이템 추천 실행", type="primary"):
    with st.spinner("추천 계산 중..."):
        result = try_recommend()
        # result 반환 형태가 다양할 수 있으므로 몇 가지 패턴 처리
        if isinstance(result, dict):
            recommendations = result.get("items") or result.get("recommendations") or result.get("result")
            explanation     = result.get("reason") or result.get("explanation")
        elif isinstance(result, (list, tuple)):
            recommendations = result
        else:
            recommendations = result

    if recommendations:
        st.success("추천 아이템")
        df = pd.DataFrame({"추천 아이템": recommendations})
        st.dataframe(df, use_container_width=True)
        if explanation:
            st.caption(explanation)
    else:
        st.info("추천 결과가 비어있습니다. 함수 시그니처가 다른 경우일 수 있어요. `item_recommender.py`의 공개 함수명을 알려주면 정확히 맞춰 드릴게요.")

st.caption(f"APP_BASE = {APP_BASE} (env에 주입됨)")
