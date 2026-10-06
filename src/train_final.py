import os
import json
import pandas as pd
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

STATIONS = ["ATL", "ORD", "SEA", "BOS", "DFW", "LAX"]
HORIZONS = [1, 2, 3, 4, 5, 6]

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

if not os.path.exists("models"):
    os.makedirs("models")

print("Station | Horizon | threshold | val F1 (2024) | test F1 (2025) | persistence F1 (2025)")
for station in STATIONS:
    for horizon in HORIZONS:
        filename = "data/" + station + "_h" + str(horizon) + "_features.csv"
        df = pd.read_csv(filename, index_col="hour", parse_dates=True)
        feature_cols = []
        for c in df.columns:
            if c != "future_ifr":
                feature_cols.append(c)

        train = df[df.index.year < 2024]
        val = df[df.index.year == 2024]
        test = df[df.index.year == 2025]

        model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", random_state=42)
        model.fit(train[feature_cols], train["future_ifr"])

        val_probs = model.predict_proba(val[feature_cols])[:, 1]
        threshold = best_threshold(val["future_ifr"], val_probs)
        val_f1 = f1_score(val["future_ifr"], (val_probs >= threshold).astype(int))

        test_probs = model.predict_proba(test[feature_cols])[:, 1]
        test_f1 = f1_score(test["future_ifr"], (test_probs >= threshold).astype(int))
        persistence_f1 = f1_score(test["future_ifr"], test["now_ifr"].astype(int))

        base = "models/" + station + "_h" + str(horizon)
        model.save_model(base + ".json")
        info = {
            "station": station,
            "horizon": horizon,
            "threshold": threshold,
            "features": feature_cols,
            "val_f1_2024": round(val_f1, 4),
            "test_f1_2025": round(test_f1, 4),
            "persistence_f1_2025": round(persistence_f1, 4),
        }
        out_file = open(base + "_info.json", "w")
        json.dump(info, out_file, indent=2)
        out_file.close()

        print(station, "|", horizon, "|", threshold, "|", round(val_f1, 3), "|",
              round(test_f1, 3), "|", round(persistence_f1, 3))