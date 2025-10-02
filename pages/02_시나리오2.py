# pages/02_시나리오2.py
import os, sys, runpy

# === 새 위치 반영 ===
BASE2 = r"C:/Users/권도혁/Desktop/시나리오1,2/시나리오2"
APP2  = os.path.join(BASE2, "app.py")

# 1) 이전에 로드된 동일 모듈(루트의 item_recommender 등) 캐시 제거
for mod in ["item_recommender", "rune_champion"]:
    if mod in sys.modules:
        sys.modules.pop(mod, None)

# 2) 시나리오2 폴더를 임포트 우선순위 최상단에
if BASE2 in sys.path:
    sys.path.remove(BASE2)
sys.path.insert(0, BASE2)

# 3) CWD를 시나리오2로 바꾸고 실행(상대경로 보호)
_prev = os.getcwd()
os.chdir(BASE2)
try:
    runpy.run_path(APP2, run_name="__main__")
finally:
    os.chdir(_prev)
