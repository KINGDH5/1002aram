# app.py (시나리오1 개선판 - Drive CSV + Secrets 인증)
import os, re, tempfile
import gdown
import streamlit as st
import pandas as pd
from PIL import Image

from ml import read_csv_safe, train_models, get_team_winrate, list_all_champs
from image import init_vertex, predict_image

# === Drive CSV 로더 ===
@st.cache_data(show_spinner=False)
def load_csv_from_drive_link(drive_url: str) -> pd.DataFrame:
    m = re.search(r"/d/([a-zA-Z0-9_-]{20,})", drive_url)
    if not m:
        m = re.search(r"[?&]id=([a-zA-Z0-9_-]{20,})", drive_url)
    if not m:
        raise ValueError("유효한 구글 드라이브 링크 아님")

    file_id = m.group(1)
    direct_url = f"https://drive.google.com/uc?id={file_id}"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp_path = tmp.name
    tmp.close()

    gdown.download(direct_url, tmp_path, quiet=True)
    df = read_csv_safe(tmp_path)
    os.remove(tmp_path)
    return df

# === Secrets 불러오기 ===
drive_url = st.secrets["DRIVE"]["RENAMED_CSV_URL"]

PROJECT_ID  = st.secrets["SCENARIO1"]["PROJECT_ID"]
REGION      = st.secrets["SCENARIO1"]["REGION"]
ENDPOINT_ID = st.secrets["SCENARIO1"]["ENDPOINT_ID"]

# === Streamlit UI ===
st.set_page_config(page_title="ARAM 픽 최적화", layout="wide")
st.title("⭐ ARAM 픽 최적화 (Scenario1)")

st.sidebar.header("데이터")
mode = st.sidebar.radio("CSV 소스 선택", ["Drive (기본)", "파일 업로드"], horizontal=True)
df = None

if mode == "Drive (기본)":
    with st.spinner("Drive에서 CSV 불러오는 중..."):
        df = load_csv_from_drive_link(drive_url)
else:
    up = st.sidebar.file_uploader("CSV 업로드", type=["csv"])
    if up:
        df = read_csv_safe(up)

if df is None:
    st.info("CSV를 선택하세요.")
    st.stop()

st.dataframe(df.head(3), use_container_width=True)

# 2) 모델 학습
if "models" not in st.session_state:
    st.session_state.models = None

if st.button("학습 시작 / 다시 학습", type="primary"):
    with st.spinner("학습 중..."):
        st.session_state.models = train_models(df)

if not st.session_state.models:
    st.stop()

models = st.session_state.models
all_champs = list_all_champs(models)

# 3) 스크린샷 감지 옵션
st.sidebar.subheader("스크린샷 감지 (옵션)")
use_vertex = st.sidebar.checkbox("사용", value=False)
threshold = st.sidebar.slider("신뢰도(%)", 50, 95, 70, 1)

@st.cache_resource
def get_endpoint():
    return init_vertex(PROJECT_ID, REGION, ENDPOINT_ID)

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
        detected_bench = bench[:10]

# 4) 우리 팀 선택
default_team = (detected_current if len(detected_current)==5 else all_champs[:5])
my_team = st.multiselect("우리 팀 (5명)", options=all_champs, default=default_team, max_selections=5)

if len(my_team) != 5:
    st.warning("5명을 선택하세요.")
    st.stop()

wr = get_team_winrate(my_team, models)
st.markdown(f"### 현재 픽 승률: **{wr*100:.2f}%**")

# 5) 교체 추천
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
