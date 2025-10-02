# image.py (시나리오1 - Vertex 인증 secrets 적용)
import os, io, base64, time, json
from typing import Tuple, Optional
from PIL import Image, ImageDraw
import numpy as np
from google.cloud import aiplatform
from google.oauth2 import service_account
import streamlit as st

BASE_W, BASE_H = 1920, 1080

# 블루팀 박스
BLUE = [
    (83, 157, 175, 248), (83, 277, 175, 368), (83, 397, 175, 488),
    (83, 517, 175, 608), (83, 637, 175, 728),
]

# 레드팀 박스
RED  = [
    (529,16,602,89),(617,16,690,89),(705,16,778,89),(793,16,866,89),(881,16,954,89),
    (969,16,1042,89),(1057,16,1130,89),(1145,16,1218,89),(1233,16,1306,89),(1321,16,1394,89)
]

def _scale_coords(w,h, base, dx=0,dy=0, sx=1.0,sy=1.0):
    rx, ry = (w/BASE_W)*sx, (h/BASE_H)*sy
    return [(int(l*rx)+dx,int(t*ry)+dy,int(r*rx)+dx,int(b*ry)+dy) for (l,t,r,b) in base]

def _crop(image, dx=0,dy=0, sx=1.0,sy=1.0):
    w,h = image.size
    b = _scale_coords(w,h,BLUE,dx,dy,sx,sy)
    r = _scale_coords(w,h,RED, dx,dy,sx,sy)
    tiles=[]
    for (l,t,rr,bb) in b+r:
        im = image.crop((l,t,rr,bb)).convert("RGB").resize((128,128), Image.Resampling.LANCZOS)
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=50)
        tiles.append(buf.getvalue())
    return tiles,b,r

def draw_overlay(img, b, r):
    im = img.copy(); dr = ImageDraw.Draw(im)
    for x in b: dr.rectangle(x, outline=(0,128,255), width=3)
    for x in r: dr.rectangle(x, outline=(255,64,64), width=3)
    return im

# ---- 엔드포인트 캐시 ----
_ENDPOINT_CACHE = {}

def init_vertex(project_id: str, region: str, endpoint_id: str):
    # secrets에서 B64 키 불러오기
    cred_b64 = st.secrets["SCENARIO1"]["GOOGLE_APPLICATION_CREDENTIALS_B64"]
    cred_info = json.loads(base64.b64decode(cred_b64))
    creds = service_account.Credentials.from_service_account_info(cred_info)

    key = (project_id, region.strip().lower(), endpoint_id)
    if key in _ENDPOINT_CACHE:
        return _ENDPOINT_CACHE[key]

    aiplatform.init(
        project=project_id,
        location=region.strip().lower(),
        credentials=creds,
        api_endpoint=f"{region.strip().lower()}-aiplatform.googleapis.com",
    )
    ep = aiplatform.Endpoint(endpoint_id)
    _ENDPOINT_CACHE[key] = ep
    return ep

def _predict_one(endpoint, img_bytes, retries=3, delay=0.5) -> Tuple[Optional[str], float]:
    inst = {"content": base64.b64encode(img_bytes).decode("utf-8")}
    last = None
    for _ in range(retries):
        try:
            resp = endpoint.predict(instances=[inst])
            if not getattr(resp, "predictions", None):
                return (None, 0.0)
            pred = resp.predictions[0]
            names = pred.get("displayNames", []); confs = pred.get("confidences", [])
            if not names or not confs:
                return (None, 0.0)
            i = int(np.argmax(confs))
            return (names[i], float(confs[i]) * 100.0)
        except Exception as e:
            last = e
            time.sleep(delay)
    raise last

def predict_image(endpoint, image, threshold=70.0, dx=0,dy=0, scale_w=1.0,scale_h=1.0):
    tiles, b, r = _crop(image, dx,dy, scale_w,scale_h)
    named=[]
    for t in tiles:
        n,c = _predict_one(endpoint, t)
        named.append((n if (n and c>=threshold) else None, c if c>=threshold else 0.0))

    current = [n for (n,_) in named[:5] if n]
    bench = []
    for (n,_) in named[5:]:
        if not n: bench.append(None)
        elif n.strip().lower() in ["hwei","흐웨이"]: bench.append(None)
        else: bench.append(n)

    overlay = draw_overlay(image, b, r)
    return current, bench, overlay
