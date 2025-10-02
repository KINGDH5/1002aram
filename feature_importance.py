# feature_importance.py (시나리오1 피처 중요도 계산 - 경로 수정)
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

def compute_feature_importance(models, df):
    """
    XGBoost 기반 피처 중요도 계산
    """
    # y = 승리 여부, X = 피처들
    X = df.drop(columns=["win"], errors="ignore")
    y = df["win"] if "win" in df.columns else None
    if y is None:
        raise ValueError("CSV에 'win' 컬럼이 필요합니다.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "seed": 42
    }
    model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, "test")])

    importance = model.get_score(importance_type="weight")
    fi = pd.DataFrame(list(importance.items()), columns=["feature", "importance"])
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    return fi
