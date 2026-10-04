import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

df = pd.read_csv("data/ATL_features.csv", index_col="hour", parse_dates=True)

feature_cols = []
for c in df.columns:
    if c != "future_ifr":
        feature_cols.append(c)

def best_threshold(y_true, probs):
    best_t = 0.5
    best_f1 = 0
    for i in range(5, 95, 5):
        t = i / 100.0
        preds = (probs >= t).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
            best_t = t
    return best_t

scores = {"Persistence": [], "Logistic regression": [], "XGBoost": []}

test_years = [2021, 2022, 2023, 2024, 2025]

for test_year in test_years:
    val_year = test_year - 1

    train = df[df.index.year < val_year]
    val = df[df.index.year == val_year]
    test = df[df.index.year == test_year]

    X_train = train[feature_cols]
    y_train = train["future_ifr"]
    X_val = val[feature_cols]
    y_val = val["future_ifr"]
    X_test = test[feature_cols]
    y_test = test["future_ifr"]

    # Persistence baseline
    f1_persist = f1_score(y_test, X_test["now_ifr"])
    scores["Persistence"].append(f1_persist)

    # Logistic regression
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_s, y_train)
    t = best_threshold(y_val, logreg.predict_proba(X_val_s)[:, 1])
    preds = (logreg.predict_proba(X_test_s)[:, 1] >= t).astype(int)
    f1_lr = f1_score(y_test, preds)
    scores["Logistic regression"].append(f1_lr)

    # XGBoost
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss")
    xgb.fit(X_train, y_train)
    t = best_threshold(y_val, xgb.predict_proba(X_val)[:, 1])
    preds = (xgb.predict_proba(X_test)[:, 1] >= t).astype(int)
    f1_xgb = f1_score(y_test, preds)
    scores["XGBoost"].append(f1_xgb)

    print("Test year", test_year, "| train rows", len(train),
          "| persistence", round(f1_persist, 3),
          "| logreg", round(f1_lr, 3),
          "| xgboost", round(f1_xgb, 3))

print("")
print("Average F1 across folds (and spread):")
summary = {}
for name in scores:
    avg = np.mean(scores[name])
    spread = np.std(scores[name])
    print(name, ":", round(avg, 3), "+/-", round(spread, 3))
    rounded = []
    for x in scores[name]:
        rounded.append(round(x, 3))
    summary[name] = {"mean_f1": round(avg, 3), "std_f1": round(spread, 3),
                     "fold_f1": rounded}

if not os.path.exists("results"):
    os.makedirs("results")
out_file = open("results/cv_metrics.json", "w")
json.dump(summary, out_file, indent=2)
out_file.close()