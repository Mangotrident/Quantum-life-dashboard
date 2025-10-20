import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

def build_pipeline():
    categorical = ["topology"]
    numeric = ["J","gamma","sigma"]
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical),
         ("num", "passthrough", numeric)]
    )
    model = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
    )
    pipe = Pipeline([("pre", pre), ("model", model)])
    return pipe

def split_fit(pipe, df, test_size=0.2, seed=42):
    X = df[["topology","J","gamma","sigma"]]
    y = df["QLS"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=df["topology"])
    pipe.fit(X_tr, y_tr)
    return pipe, (X_te, y_te)
