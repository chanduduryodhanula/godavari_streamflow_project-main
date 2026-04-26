from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def train_random_forest(X_train, y_train) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, X_val, y_val) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.2,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model
