# item_recommender.py
# -*- coding: utf-8 -*-
import os
import pandas as pd
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# ===============================
# 경로 설정 (새 위치 + 폴백)
# ===============================
# 1) 우선순위: 사용자가 옮긴 절대 경로
ABS_BASE = r"C:\Users\권도혁\Desktop\시나리오1,2\시나리오2"
# 2) 폴백: 이 파일이 있는 폴더
REL_BASE = os.path.dirname(__file__)  # (kdh 폴더 기준)

def _resolve(candidate_path, fallback_path):
    """candidate가 존재하면 그걸 쓰고, 없으면 fallback 사용"""
    return candidate_path if os.path.exists(candidate_path) else fallback_path

MODEL_PATH = _resolve(os.path.join(ABS_BASE, "lgbm_model_tuned.joblib"),
                      os.path.join(REL_BASE, "lgbm_model_tuned.joblib"))
BUILD_JSON = _resolve(os.path.join(ABS_BASE, "템트리_converted_fixed.json"),
                      os.path.join(REL_BASE, "템트리_converted_fixed.json"))
CC_CSV     = _resolve(os.path.join(ABS_BASE, "champ_job_cc.csv"),
                      os.path.join(REL_BASE, "champ_job_cc.csv"))

# ===============================
# 초기화 함수
# ===============================
def initialize_recommender():
    try:
        global model, build_data, trained_features, cc_df, champion_to_roles_map
        model = joblib.load(MODEL_PATH)
        with open(BUILD_JSON, 'r', encoding='utf-8') as f:
            build_data = json.load(f)
        trained_features = model.feature_names_in_
        cc_df = pd.read_csv(CC_CSV, encoding='utf-8')
        # 빌드 JSON 구조: {챔피언: {역할군: {상황키: [아이템리스트]}}}
        champion_to_roles_map = {champ: list(roles.keys()) for champ, roles in build_data.items()}
        return True
    except Exception as e:
        print(f"초기화 실패: {e}")
        return False

# ===============================
# 상황 판단
# ===============================
def determine_situation(enemy_team):
    num_ad = num_ap = num_cc = num_tanks = 0
    for champ_name, rune_name, role_name in enemy_team:
        if "AD" in role_name: num_ad += 1
        if "AP" in role_name: num_ap += 1
        if "탱커" in role_name: num_tanks += 1
        cc_row = cc_df[cc_df['name'] == champ_name]
        if not cc_row.empty:
            try:
                num_cc += int(cc_row.iloc[0]['CCcount'])
            except Exception:
                pass
    damage_type_cond = "상대AP" if num_ap >= 3 else "상대AD"
    cc_cond = "CC많음" if num_cc >= 3 else "CC적음"
    tank_cond = "탱커많음" if num_tanks >= 2 else "탱커적음"
    return [f"{damage_type_cond}+{cc_cond}", f"{tank_cond}+{cc_cond}"]

# ===============================
# 추천 함수
# ===============================
def get_all_build_recommendations(my_champion, enemy_team):
    my_roles = champion_to_roles_map.get(my_champion, [])
    possible_situations = determine_situation(enemy_team)
    recommendations_by_role = []

    for role in my_roles:
        expert_build = None
        # 1순위: 상황키 정확 매칭
        for situation_key in possible_situations:
            build = build_data.get(my_champion, {}).get(role, {}).get(situation_key)
            if build:
                expert_build = build
                break
        # 2순위: 서포터일 때 CC 기준 대체 매칭
        if not expert_build and role == '서포터':
            cc_part = "CC많음" if "CC많음" in possible_situations[0] else "CC적음"
            for k, build in build_data.get(my_champion, {}).get(role, {}).items():
                if cc_part in k:
                    expert_build = build
                    break
        if not expert_build:
            continue

        # 모델 입력 벡터 생성
        input_data = pd.DataFrame(columns=trained_features); input_data.loc[0] = 0
        champ_col = f'championName_{my_champion}'
        if champ_col in input_data.columns: input_data.loc[0, champ_col] = 1
        team_col = f'team_role_{role}'
        if team_col in input_data.columns: input_data.loc[0, team_col] = 1

        # 상대 역할군 카운트 반영
        enemy_role_counts = {}
        for _, _, role_name in enemy_team:
            enemy_role_counts[role_name] = enemy_role_counts.get(role_name, 0) + 1
        for r, count in enemy_role_counts.items():
            col = f'enemy_role_{r}'
            if col in input_data.columns: input_data.loc[0, col] = count

        # 아이템 원-핫
        for item in expert_build:
            col = f'item_{item}'
            if col in input_data.columns: input_data.loc[0, col] = 1

        win_prob = model.predict_proba(input_data)[0][1]
        recommendations_by_role.append({"role": role, "build": expert_build, "win_prob": win_prob})

    return sorted(recommendations_by_role, key=lambda x: x['win_prob'], reverse=True)

def get_cc_df():
    return cc_df
