import os
import json
import numpy as np
import pandas as pd
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

def run_fold(test_year, params):
    val_year = test_year - 1
    train = df[df.index.year < val_year]
    val = df[df.index.year == val_year]
    test = df[df.index.year == test_year]

    model = XGBClassifier(n_estimators=params["n_estimators"],
                          max_depth=params["max_depth"],
                          learning_rate=params["learning_rate"],
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", random_state=42)
    model.fit(train[feature_cols], train["future_ifr"])

    val_probs = model.predict_proba(val[feature_cols])[:, 1]
    t = best_threshold(val["future_ifr"], val_probs)
    test_probs = model.predict_proba(test[feature_cols])[:, 1]
    preds = (test_probs >= t).astype(int)
    return f1_score(test["future_ifr"], preds)

tune_years = [2021, 2022, 2023]
holdout_years = [2024, 2025]

# Step 1: try every combination, scoring on the tuning years only
all_results = []
best_params = None
best_score = 0

for depth in [3, 4, 6]:
    for lr in [0.03, 0.1]:
        for n_trees in [200, 400]:
            params = {"max_depth": depth, "learning_rate": lr, "n_estimators": n_trees}
            fold_scores = []
            for year in tune_years:
                fold_scores.append(run_fold(year, params))
            avg = np.mean(fold_scores)
            print(params, "-> avg F1 on tuning years:", round(avg, 4))
            all_results.append({"params": params, "tuning_f1": round(avg, 4)})
            if avg > best_score:
                best_score = avg
                best_params = params

print("")
print("Best settings:", best_params, "tuning F1:", round(best_score, 4))

# Step 2: compare default vs tuned on the holdout years
default_params = {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300}
print("")
print("Holdout check (years never used to pick settings):")
holdout = {}
for name, params in [("default", default_params), ("tuned", best_params)]:
    scores = []
    for year in holdout_years:
        scores.append(run_fold(year, params))
    print(name, "per-year F1:", round(scores[0], 3), round(scores[1], 3),
          "| average:", round(np.mean(scores), 3))
    holdout[name] = {"params": params, "avg_f1": round(np.mean(scores), 4)}

if not os.path.exists("results"):
    os.makedirs("results")
out_file = open("results/tuning.json", "w")
json.dump({"grid": all_results, "best_params": best_params, "holdout": holdout}, out_file, indent=2)
out_file.close()