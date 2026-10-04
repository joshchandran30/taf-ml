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

def make_model():
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         eval_metric="logloss", random_state=42)

def top_features(train, n_features):
    model = make_model()
    model.fit(train[feature_cols], train["future_ifr"])
    imp = pd.DataFrame({"feature": feature_cols,
                        "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    return imp["feature"].tolist()[:n_features]

def run_fold(test_year, n_features):
    val_year = test_year - 1
    train = df[df.index.year < val_year]
    val = df[df.index.year == val_year]
    test = df[df.index.year == test_year]

    cols = top_features(train, n_features)

    model = make_model()
    model.fit(train[cols], train["future_ifr"])
    val_probs = model.predict_proba(val[cols])[:, 1]
    t = best_threshold(val["future_ifr"], val_probs)
    test_probs = model.predict_proba(test[cols])[:, 1]
    preds = (test_probs >= t).astype(int)
    return f1_score(test["future_ifr"], preds)

sizes = [6, 10, 15, 20, 28]
tune_years = [2021, 2022, 2023]
holdout_years = [2024, 2025]

# Step 1: score each feature count on the tuning years
tune_scores = {}
for n in sizes:
    fold_scores = []
    for year in tune_years:
        fold_scores.append(run_fold(year, n))
    tune_scores[n] = np.mean(fold_scores)
    print("Top", n, "features -> avg F1 on tuning years:", round(tune_scores[n], 4))

# Step 2: pick the smallest size within 0.003 of the best score
best_overall = max(tune_scores.values())
chosen = 28
for n in sizes:
    if tune_scores[n] >= best_overall - 0.003:
        chosen = n
        break
print("")
print("Chosen number of features:", chosen)

# Step 3: holdout check, all features vs chosen
print("")
print("Holdout check (2024 and 2025):")
holdout = {}
for n in [28, chosen]:
    scores = []
    for year in holdout_years:
        scores.append(run_fold(year, n))
    print(n, "features -> per-year F1:", round(scores[0], 3), round(scores[1], 3),
          "| average:", round(np.mean(scores), 3))
    holdout[str(n)] = round(np.mean(scores), 4)

# Step 4: show which features were kept (ranked using data before 2024)
final_train = df[df.index.year < 2024]
kept = top_features(final_train, chosen)
print("")
print("Kept features:", kept)

tuning_f1 = {}
for k in tune_scores:
    tuning_f1[str(k)] = round(tune_scores[k], 4)

if not os.path.exists("results"):
    os.makedirs("results")
out_file = open("results/feature_trim.json", "w")
json.dump({"tuning_f1": tuning_f1, "chosen": chosen,
           "holdout": holdout, "kept_features": kept},
          out_file, indent=2)
out_file.close()