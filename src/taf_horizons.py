import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from taf_tools import RANK, load_tafs, forecasts_at

STATION = "ATL"
HORIZONS = [1, 2, 3, 4, 5, 6]
YEARS = [2021, 2022, 2023, 2024, 2025]
TAF_COLS = ["taf_base_rank", "taf_tempo_rank", "taf_all_rank"]

all_tafs, issue_times = load_tafs(STATION, 2019, 2025)
print("TAFs loaded:", len(all_tafs))

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

def run_fold(df, test_year, cols):
    val_year = test_year - 1
    train = df[df.index.year < val_year]
    val = df[df.index.year == val_year]
    test = df[df.index.year == test_year]
    model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", random_state=42)
    model.fit(train[cols], train["future_ifr"])
    t = best_threshold(val["future_ifr"], model.predict_proba(val[cols])[:, 1])
    preds = (model.predict_proba(test[cols])[:, 1] >= t).astype(int)
    return f1_score(test["future_ifr"], preds)

def add_taf_columns(df, horizon):
    base_list = []
    tempo_list = []
    all_list = []
    for hour in df.index:
        decision_time = hour + pd.Timedelta(minutes=59)
        target_time = decision_time + pd.Timedelta(hours=horizon)
        answer = forecasts_at(all_tafs, issue_times, decision_time, target_time)
        if answer is None:
            base_list.append(np.nan)
            tempo_list.append(np.nan)
            all_list.append(np.nan)
        else:
            base_list.append(RANK[answer[0]])
            tempo_list.append(RANK[answer[1]])
            all_list.append(RANK[answer[2]])
    df = df.copy()
    df["taf_base_rank"] = base_list
    df["taf_tempo_rank"] = tempo_list
    df["taf_all_rank"] = all_list
    return df.dropna(subset=TAF_COLS)

methods = ["Persistence", "TAF base", "TAF + TEMPO", "Local XGBoost", "Hybrid XGBoost"]
averages = {}
for m in methods:
    averages[m] = []

for horizon in HORIZONS:
    print("")
    print("=== Horizon", horizon, "hours ===")
    filename = "data/" + STATION + "_h" + str(horizon) + "_features.csv"
    df = pd.read_csv(filename, index_col="hour", parse_dates=True)
    df = add_taf_columns(df, horizon)

    local_cols = []
    for c in df.columns:
        if c != "future_ifr" and c not in TAF_COLS:
            local_cols.append(c)
    hybrid_cols = local_cols + TAF_COLS

    scores = {}
    for m in methods:
        scores[m] = []

    for year in YEARS:
        test = df[df.index.year == year]
        y = test["future_ifr"]
        scores["Persistence"].append(f1_score(y, test["now_ifr"]))
        scores["TAF base"].append(f1_score(y, (test["taf_base_rank"] >= 2).astype(int)))
        scores["TAF + TEMPO"].append(f1_score(y, (test["taf_tempo_rank"] >= 2).astype(int)))
        scores["Local XGBoost"].append(run_fold(df, year, local_cols))
        scores["Hybrid XGBoost"].append(run_fold(df, year, hybrid_cols))

    for m in methods:
        avg = np.mean(scores[m])
        averages[m].append(avg)
        per_year = []
        for x in scores[m]:
            per_year.append(round(x, 3))
        print(m, "| average", round(avg, 3), "| by year", per_year)

print("")
print("Average F1 by horizon:")
print("Horizon | Persistence | TAF base | TAF+TEMPO | Local XGB | Hybrid XGB")
for i in range(len(HORIZONS)):
    print(HORIZONS[i], "|",
          round(averages["Persistence"][i], 3), "|",
          round(averages["TAF base"][i], 3), "|",
          round(averages["TAF + TEMPO"][i], 3), "|",
          round(averages["Local XGBoost"][i], 3), "|",
          round(averages["Hybrid XGBoost"][i], 3))

if not os.path.exists("results"):
    os.makedirs("results")
saved = {}
for m in methods:
    rounded = []
    for x in averages[m]:
        rounded.append(round(x, 4))
    saved[m] = rounded
out_file = open("results/taf_horizons_" + STATION + ".json", "w")
json.dump({"horizons": HORIZONS, "avg_f1": saved}, out_file, indent=2)
out_file.close()

plt.figure(figsize=(8, 5.5))
for m in methods:
    plt.plot(HORIZONS, averages[m], marker="o", label=m)
plt.xlabel("Forecast horizon (hours)")
plt.ylabel("Average F1 on IFR (test years 2021-2025)")
plt.title(STATION + ": model vs TAF by horizon")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/taf_horizons_" + STATION + ".png", dpi=150)
plt.close()
print("Saved results/taf_horizons_" + STATION + ".png")