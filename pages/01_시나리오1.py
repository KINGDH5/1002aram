# -*- coding: utf-8 -*-
# Scenario1 Page: runpy 없이 페이지 내에서 바로 렌더링
import io, os, re, tempfile, requests
import pandas as pd
import streamlit as st
from PIL import Image

from ml import read_csv_safe, train_models, get_team_winrate, list_all_champs
from image import init_vertex, predict_image  # (시나리오1용 image.py: secrets B64 인증 적용되어 있어야 함)

st.set_page_config(page_title="Scenario1 | ARAM 픽 최적화", layout="wide")
st.title("⭐ Scenario1 — ARAM 픽 최적화")

# ---------------------------
# Drive CSV 로더 (requests 버전)
# ---------------------------
@st.cache_data(show_spinner=False)
def load_csv_from_drive_link(drive_url: str) -> pd.DataFrame:
    m = re.search(r"/d/([a-zA-Z0-9_-]{20,})", drive_url) or \
        re.search(r"[?&]id=([a-zA-Z0-9_-]{20,})", drive_url)
    if not m:
        raise ValueError("구글드라이브 링크 형식이 아닙니다. (/d/<ID>/ 또는 ?id=<ID>)")
    file_id = m.group(1)
    direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(direct_url, allow_redirects=True, timeout=60)
    if r.status_code != 200 or not r.content:
        raise RuntimeError("드라이브 다운로드 실패(권한/링크 확인). 파일 공유를 '링크 소지자 보기'로 설정하세요.")
    return pd.read_csv(io.BytesIO(r.content))

# ---------------------------
# 데이터 로딩
# ---------------------------
st.sidebar.header("데이터")
src = st.sidebar.radio("CSV 소스", ["Drive (기본)", "파일 업로드"], horizontal=True)

df = None
if src == "Drive (기본)":
    try:
        drive_url = st.secrets["DRIVE"]["RENAMED_CSV_URL"]
    except KeyError:
        st.error("secrets.toml의 [DRIVE].RENAMED_CSV_URL이 없습니다.")
        st.stop()
    with st.spinner("Drive에서 CSV 불러오는 중..."):
        df = load_csv_from_drive_link(drive_url)
else:
    up = st.sidebar.file_uploader("CSV 업로드", type=["csv"])
    if up:
        df = read_csv_safe(up)

if df is None:
    st.info("CSV를 선택하거나 업로드하세요.")
    st.stop()

st.dataframe(df.head(3), use_container_width=True)

# ---------------------------
# 모델 학습/보관
# ---------------------------
if "models" not in st.session_state:
    st.session_state.models = None

if st.button("학습 시작 / 다시 학습", type="primary"):
    with st.spinner("학습 중..."):
        st.session_state.models = train_models(df)

models = st.session_state.models
if not models:
    st.stop()

all_champs = list_all_champs(models)

# ---------------------------
# 스크린샷 감지(옵션)
# ---------------------------
st.sidebar.subheader("스크린샷 감지 (옵션)")
use_vertex = st.sidebar.checkbox("사용", value=False)
threshold = st.sidebar.slider("신뢰도(%)", 50, 95, 70, 1)

@st.cache_resource
def get_endpoint():
    proj = st.secrets["SCENARIO1"]["PROJECT_ID"]
    reg  = st.secrets["SCENARIO1"]["REGION"]
    epid = st.secrets["SCENARIO1"]["ENDPOINT_ID"]
    return init_vertex(proj, reg, epid)

detected_current, detected_bench = [], []
if use_vertex:
    uploaded = st.file_uploader("픽 화면 스크린샷 (png/jpg)", type=["png","jpg","jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="업로드 이미지", use_container_width=True)
        with st.spinner("감지 중..."):
            endpoint = get_endpoint()
            cur, bench, overlay = predict_image(endpoint, image, threshold=threshold)
        st.image(overlay, caption="탐지 영역", use_container_width=True)
        detected_current = cur[:5]
        detected_bench = [b for b in bench if b][:10]

# ---------------------------
# 우리 팀 선택 & 승률
# ---------------------------
default_team = (detected_current if len(detected_current)==5 else all_champs[:5])
my_team = st.multiselect("우리 팀 (5명)", options=all_champs, default=default_team, max_selections=5)
if len(my_team) != 5:
    st.warning("5명을 선택하세요.")
    st.stop()

wr = get_team_winrate(my_team, models)
st.markdown(f"### 현재 픽 승률: **{wr*100:.2f}%**")

# ---------------------------
# 교체 추천
# ---------------------------
pool = st.multiselect("교체 후보", options=[c for c in all_champs if c not in my_team],
                      default=[c for c in detected_bench if c not in my_team])

target = st.selectbox("교체할 내 챔피언", options=my_team)
rows=[]; best=None; best_inc=0.0
for cand in pool:
    new_team = [cand if x==target else x for x in my_team]
    w = get_team_winrate(new_team, models)
    inc = w - wr
    rows.append({"교체 챔피언": cand, "새 승률(%)": round(w*100,2), "Δ(%)": round(inc*100,2)})
    if inc > best_inc:
        best, best_inc = (target,cand,w), inc

if rows:
    st.dataframe(pd.DataFrame(rows).sort_values("새 승률(%)", ascending=False), use_container_width=True)
if best:
    st.success(f"🔷 {best[0]} → {best[1]} 교체 시 **{best[2]*100:.2f}%**")
else:
    st.info("교체 후보를 선택하면 추천이 표시됩니다.")
