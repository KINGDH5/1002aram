# -*- coding: utf-8 -*-
"""
rune_champion.py
- Streamlit Cloud 호환: 로컬 JSON 키 파일 경로 사용 금지
- 시크릿(.streamlit/secrets.toml)의 [SCENARIO2] 값을 사용해 메모리로 인증
- Vertex AI Endpoint로 룬/역할(또는 챔피언) 분류
- predict_image(endpoint, image, threshold=...) -> (current5, bench10, overlay)
"""

from __future__ import annotations
import base64
import io
import json
import time
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
import streamlit as st

from google.cloud import aiplatform
from google.oauth2 import service_account

# (옵션) OCR이 필요하면 주석 해제
try:
    from google.cloud import vision  # noqa: F401
    _HAS_VISION = True
except Exception:
    _HAS_VISION = False


# =========================
# 기본 좌표(1920x1080 기준)
# =========================
BASE_W, BASE_H = 1920, 1080

# 블루팀(현재 픽) 5칸
BLUE = [
    (83, 157, 175, 248),
    (83, 277, 175, 368),
    (83, 397, 175, 488),
    (83, 517, 175, 608),
    (83, 637, 175, 728),
]

# 레드팀(벤치/상대 후보) 10칸
RED = [
    (529, 16, 602, 89), (617, 16, 690, 89), (705, 16, 778, 89), (793, 16, 866, 89), (881, 16, 954, 89),
    (969, 16, 1042, 89), (1057, 16, 1130, 89), (1145, 16, 1218, 89), (1233, 16, 1306, 89), (1321, 16, 1394, 89)
]


# =========================
# Secrets → Credentials
# =========================
def _get_credentials_and_cfg():
    """
    [SCENARIO2] 섹션에서 프로젝트/리전/엔드포인트/SA B64를 읽어 인증 객체와 설정을 반환
    """
    s2 = st.secrets["SCENARIO2"]
    cred_b64 = s2["GOOGLE_APPLICATION_CREDENTIALS_B64"]
    info = json.loads(base64.b64decode(cred_b64))
    creds = service_account.Credentials.from_service_account_info(info)
    return creds, {
        "project": s2["PROJECT_ID"],
        "region": s2["REGION"].strip().lower(),
        "endpoint_id": s2["ENDPOINT_ID"],
    }


@st.cache_resource
def init_vertex_from_secrets():
    """
    secrets 기반 Vertex AI Endpoint 초기화 (캐시)
    """
    creds, cfg = _get_credentials_and_cfg()
    aiplatform.init(
        project=cfg["project"],
        location=cfg["region"],
        credentials=creds,
        api_endpoint=f"{cfg['region']}-aiplatform.googleapis.com",
    )
    return aiplatform.Endpoint(cfg["endpoint_id"])


@st.cache_resource
def init_vision_from_secrets():
    """
    (옵션) Google Vision OCR 클라이언트 초기화
    """
    if not _HAS_VISION:
        return None
    from google.cloud import vision  # local import
    creds, _ = _get_credentials_and_cfg()
    return vision.ImageAnnotatorClient(credentials=creds)


# =========================
# 이미지 유틸
# =========================
def _scale_coords(img_w: int, img_h: int, boxes: List[Tuple[int, int, int, int]],
                  dx: int = 0, dy: int = 0, sx: float = 1.0, sy: float = 1.0) -> List[Tuple[int, int, int, int]]:
    rx, ry = (img_w / BASE_W) * sx, (img_h / BASE_H) * sy
    scaled = []
    for (l, t, r, b) in boxes:
        scaled.append((
            int(l * rx) + dx,
            int(t * ry) + dy,
            int(r * rx) + dx,
            int(b * ry) + dy
        ))
    return scaled


def _crop_tiles(image: Image.Image, dx: int = 0, dy: int = 0,
                sx: float = 1.0, sy: float = 1.0) -> Tuple[list[bytes], list[Tuple[int,int,int,int]], list[Tuple[int,int,int,int]]]:
    """
    이미지에서 BLUE 5 + RED 10 총 15개 영역을 잘라 128x128로 리사이즈 후 JPEG 바이트로 반환
    """
    w, h = image.size
    b = _scale_coords(w, h, BLUE, dx, dy, sx, sy)
    r = _scale_coords(w, h, RED,  dx, dy, sx, sy)

    tiles: list[bytes] = []
    for (l, t, rr, bb) in b + r:
        crop = image.crop((l, t, rr, bb)).convert("RGB").resize((128, 128), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=50)
        tiles.append(buf.getvalue())
    return tiles, b, r


def _draw_overlay(image: Image.Image, blue_boxes, red_boxes) -> Image.Image:
    im = image.copy()
    dr = ImageDraw.Draw(im)
    for x in blue_boxes:
        dr.rectangle(x, outline=(0, 128, 255), width=3)   # blue
    for x in red_boxes:
        dr.rectangle(x, outline=(255, 64, 64), width=3)   # red
    return im


# =========================
# Inference
# =========================
def _predict_one(endpoint, img_bytes: bytes, retries: int = 3, delay: float = 0.4) -> Tuple[Optional[str], float]:
    """
    Vertex AI 엔드포인트에 단일 이미지(바이트) 분류 요청
    반환: (displayName, confidence(0~100))
    """
    inst = {"content": base64.b64encode(img_bytes).decode("utf-8")}
    last = None
    for _ in range(retries):
        try:
            resp = endpoint.predict(instances=[inst])
            # predictions: {"displayNames": [...], "confidences": [...] } 형태 가정
            pred = getattr(resp, "predictions", None)
            if not pred:
                return None, 0.0
            p0 = pred[0]
            names = p0.get("displayNames", []) if isinstance(p0, dict) else []
            confs = p0.get("confidences", [])  if isinstance(p0, dict) else []
            if not names or not confs:
                return None, 0.0
            i = int(np.argmax(confs))
            return names[i], float(confs[i]) * 100.0
        except Exception as e:
            last = e
            time.sleep(delay)
    # 재시도 후 실패 → 마지막 예외를 올리기보다 (None,0) 으로 무해하게 처리
    return None, 0.0


def predict_image(endpoint,
                  image: Image.Image,
                  threshold: float = 70.0,
                  dx: int = 0, dy: int = 0,
                  scale_w: float = 1.0, scale_h: float = 1.0
                  ) -> Tuple[list[str], list[Optional[str]], Image.Image]:
    """
    시나리오2 전용: 15칸(현재5 + 벤치10) 분류 → (current, bench, overlay) 반환
    - endpoint 가 None이면 secrets로 내부 초기화
    - threshold(%) 미만이면 None 처리
    """
    if endpoint is None:
        endpoint = init_vertex_from_secrets()

    tiles, b, r = _crop_tiles(image, dx=dx, dy=dy, sx=scale_w, sy=scale_h)

    names_conf = []
    for t in tiles:
        n, c = _predict_one(endpoint, t)
        if n and c >= float(threshold):
            names_conf.append((n, c))
        else:
            names_conf.append((None, 0.0))

    # 앞 5개: 현재 픽, 뒤 10개: 벤치
    cur_raw = names_conf[:5]
    ben_raw = names_conf[5:]

    # 필터/클린업 규칙(예: 특정 클래스 무시 등) 필요하면 여기서 처리
    current = [n for (n, _) in cur_raw if n]
    bench = []
    for (n, _) in ben_raw:
        if not n:
            bench.append(None)
        else:
            # 클래스명 교정 예시: 'Hwei' 오탐 제거
            if n.strip().lower() in {"hwei", "흐웨이"}:
                bench.append(None)
            else:
                bench.append(n)

    overlay = _draw_overlay(image, b, r)
    return current, bench, overlay


# =========================
# (선택) OCR 보조 기능
# =========================
def ocr_champion_names(image: Image.Image) -> list[str]:
    """
    Google Vision OCR로 화면에서 챔피언 한글명을 추출(선택적)
    - secrets 필요
    """
    if not _HAS_VISION:
        st.warning("google-cloud-vision 패키지가 설치되어 있지 않아 OCR을 사용할 수 없습니다.")
        return []

    client = init_vision_from_secrets()
    if client is None:
        st.warning("Vision 클라이언트를 초기화할 수 없습니다.")
        return []

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    content = buf.getvalue()
    from google.cloud import vision  # local import
    img = vision.Image(content=content)
    resp = client.text_detection(image=img)
    if resp.error.message:
        st.warning(f"OCR 오류: {resp.error.message}")
        return []
    texts = [t.description.strip() for t in (resp.text_annotations or []) if t.description]
    # 간단 후처리: 줄바꿈/공백 분리
    tokens = []
    for t in texts:
        tokens.extend([x for x in t.replace("\n", " ").split(" ") if x])
    # 필요하면 교정 사전 적용 가능
    return list(dict.fromkeys(tokens))  # 중복 제거 순서 보존
