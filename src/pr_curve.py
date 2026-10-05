import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from xgboost import XGBClassifier

df = pd.read_csv("data/ATL_features.csv", index_col="hour", parse_dates=True)
taf = pd.read_csv("data/ATL_taf_baseline.csv", index_col="hour", parse_dates=True)

in_file = open("results/feature_trim.json", "r")
cols = json.load(in_file)["kept_features"]
in_file.close()

# Train on 2019-2023, evaluate on 2024-2025 hours that have a TAF forecast
train = df[df.index.year < 2024]
test = df[df.index.year >= 2024]
taf = taf[taf["has_taf"] == True]
common = test.index.intersection(taf.index)
test = test.loc[common]
taf = taf.loc[common]

model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      eval_metric="logloss", random_state=42)
model.fit(train[cols], train["future_ifr"])
probs = model.predict_proba(test[cols])[:, 1]
y = test["future_ifr"]

precision, recall, thresholds = precision_recall_curve(y, probs)

def precision_at_recall(target_recall):
    best = 0
    for i in range(len(recall)):
        if recall[i] >= target_recall and precision[i] > best:
            best = precision[i]
    return best

def recall_at_precision(target_precision):
    best = 0
    for i in range(len(precision)):
        if precision[i] >= target_precision and recall[i] > best:
            best = recall[i]
    return best

# The fixed points: persistence and the three TAF variants
points = {
    "Persistence": test["now_ifr"],
    "TAF base": taf["taf_base_ifr"],
    "TAF + TEMPO": taf["taf_tempo_ifr"],
    "TAF + TEMPO + PROB30": taf["taf_all_ifr"],
}

print("Method | its precision | its recall | XGBoost precision at that recall | XGBoost recall at that precision")
plt.figure(figsize=(7, 6))
plt.plot(recall, precision, label="XGBoost (all cutoffs)")

for name in points:
    p = precision_score(y, points[name])
    r = recall_score(y, points[name])
    xgb_p = precision_at_recall(r)
    xgb_r = recall_at_precision(p)
    print(name, "|", round(p, 3), "|", round(r, 3), "|",
          round(xgb_p, 3), "|", round(xgb_r, 3))
    plt.scatter([r], [p], s=60, label=name)

plt.xlabel("Recall (share of real IFR hours caught)")
plt.ylabel("Precision (share of IFR calls that were right)")
plt.title("Predicting IFR 3 hours ahead at ATL, 2024-2025")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/pr_curve.png", dpi=150)
plt.close()
print("Saved results/pr_curve.png")