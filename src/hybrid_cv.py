import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

RANK = {"VFR": 0, "MVFR": 1, "IFR": 2, "LIFR": 3}

df = pd.read_csv("data/ATL_features.csv", index_col="hour", parse_dates=True)
taf = pd.read_csv("data/ATL_taf_baseline.csv", index_col="hour", parse_dates=True)

in_file = open("results/feature_trim.json", "r")
local_cols = json.load(in_file)["kept_features"]
in_file.close()

# Turn the TAF category forecasts into numbers 0-3
taf = taf[taf["has_taf"] == True].copy()
taf_cols = ["taf_base_rank", "taf_tempo_rank", "taf_all_rank"]
taf["taf_base_rank"] = taf["taf_base"].map(RANK)
taf["taf_tempo_rank"] = taf["taf_tempo"].map(RANK)
taf["taf_all_rank"] = taf["taf_all"].map(RANK)

# Keep only hours that have both local features and a TAF
df = df.join(taf[taf_cols], how="inner")
hybrid_cols = local_cols + taf_cols
print("Rows:", len(df))

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

def run_fold(test_year, cols):
    val_year = test_year - 1
    train = df[df.index.year < val_year]
    val = df[df.index.year == val_year]
    test = df[df.index.year == test_year]

    model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", random_state=42)
    model.fit(train[cols], train["future_ifr"])

    val_probs = model.predict_proba(val[cols])[:, 1]
    t = best_threshold(val["future_ifr"], val_probs)
    test_probs = model.predict_proba(test[cols])[:, 1]
    preds = (test_probs >= t).astype(int)
    return f1_score(test["future_ifr"], preds)

years = [2021, 2022, 2023, 2024, 2025]
scores = {"TAF base": [], "Local XGBoost": [], "Hybrid XGBoost": []}

print("Year | TAF base | Local XGBoost | Hybrid XGBoost")
for year in years:
    test = df[df.index.year == year]
    taf_ifr = (test["taf_base_rank"] >= 2).astype(int)
    taf_f1 = f1_score(test["future_ifr"], taf_ifr)
    local_f1 = run_fold(year, local_cols)
    hybrid_f1 = run_fold(year, hybrid_cols)
    scores["TAF base"].append(taf_f1)
    scores["Local XGBoost"].append(local_f1)
    scores["Hybrid XGBoost"].append(hybrid_f1)
    print(year, "|", round(taf_f1, 3), "|", round(local_f1, 3), "|", round(hybrid_f1, 3))

print("")
summary = {}
for name in scores:
    avg = np.mean(scores[name])
    spread = np.std(scores[name])
    print(name, "average F1:", round(avg, 3), "+/-", round(spread, 3))
    rounded = []
    for x in scores[name]:
        rounded.append(round(x, 3))
    summary[name] = {"mean_f1": round(avg, 3), "std_f1": round(spread, 3), "fold_f1": rounded}

if not os.path.exists("results"):
    os.makedirs("results")
out_file = open("results/hybrid_cv.json", "w")
json.dump(summary, out_file, indent=2)
out_file.close()