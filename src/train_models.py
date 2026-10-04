import os
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier

df = pd.read_csv("data/ATL_features.csv", index_col="hour", parse_dates=True)

feature_cols = []
for c in df.columns:
    if c != "future_ifr":
        feature_cols.append(c)

# Split by time: train on the past, validate, then test on the future
train = df[df.index < "2023-01-01"]
val = df[(df.index >= "2023-01-01") & (df.index < "2024-01-01")]
test = df[df.index >= "2024-01-01"]
print("Train rows:", len(train), "Val rows:", len(val), "Test rows:", len(test))

X_train = train[feature_cols]
y_train = train["future_ifr"]
X_val = val[feature_cols]
y_val = val["future_ifr"]
X_test = test[feature_cols]
y_test = test["future_ifr"]

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

results = {}

def report(name, y_true, preds):
    p = precision_score(y_true, preds)
    r = recall_score(y_true, preds)
    f = f1_score(y_true, preds)
    print(name)
    print("  precision:", round(p, 3))
    print("  recall:   ", round(r, 3))
    print("  F1:       ", round(f, 3))
    results[name] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}

# Baseline: persistence on the test period
report("Persistence", y_test, X_test["now_ifr"])

# Model 1: logistic regression (needs scaled features)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_s, y_train)
t = best_threshold(y_val, logreg.predict_proba(X_val_s)[:, 1])
test_probs = logreg.predict_proba(X_test_s)[:, 1]
report("Logistic regression (threshold " + str(t) + ")", y_test, (test_probs >= t).astype(int))

# Model 2: XGBoost
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss")
xgb.fit(X_train, y_train)
t = best_threshold(y_val, xgb.predict_proba(X_val)[:, 1])
test_probs = xgb.predict_proba(X_test)[:, 1]
report("XGBoost (threshold " + str(t) + ")", y_test, (test_probs >= t).astype(int))

# Which features matter most to XGBoost?
importance = pd.DataFrame({"feature": feature_cols, "importance": xgb.feature_importances_})
importance = importance.sort_values("importance", ascending=False)
print(importance.head(10))

if not os.path.exists("results"):
    os.makedirs("results")
out_file = open("results/metrics.json", "w")
json.dump(results, out_file, indent=2)
out_file.close()