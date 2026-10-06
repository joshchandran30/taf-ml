import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import copy
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import torch
import torch.nn as nn

torch.set_num_threads(1)

SEQ_LEN = 12
HIDDEN = 64
EPOCHS = 15
PATIENCE = 3
BATCH = 256
HORIZONS = [3, 6]
YEARS = [2021, 2022, 2023, 2024, 2025]

station = "ATL"
if len(sys.argv) > 1:
    station = sys.argv[1]

class IfrLSTM(nn.Module):
    # Reads a sequence of hourly rows and outputs one score for "IFR at the horizon"
    def __init__(self, n_features, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        last = self.drop(h_n[-1])
        return self.out(last).squeeze(1)

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

def make_windows(df, feature_cols, seq_len):
    # Build one window of seq_len consecutive hours for every row that has them
    X = df[feature_cols].values.astype("float32")
    for name in ["ceiling", "ceiling_min_6h"]:
        col = feature_cols.index(name)
        X[:, col] = np.minimum(X[:, col], 12000)
    y = df["future_ifr"].values
    hours = np.array((df.index - df.index[0]).total_seconds() / 3600.0)

    windows = []
    targets = []
    times = []
    for i in range(seq_len - 1, len(df)):
        if hours[i] - hours[i - seq_len + 1] == seq_len - 1:
            windows.append(X[i - seq_len + 1:i + 1])
            targets.append(y[i])
            times.append(df.index[i])
    return np.array(windows), np.array(targets), pd.DatetimeIndex(times)

def train_lstm(X_train, y_train, X_val, y_val, n_features):
    torch.manual_seed(42)
    model = IfrLSTM(n_features, HIDDEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    best_loss = 1000000.0
    best_state = None
    bad_epochs = 0
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(X_train_t))
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH]
            optimizer.zero_grad()
            loss = loss_fn(model(X_train_t[idx]), y_train_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs = bad_epochs + 1
            if bad_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model

def predict_probs(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()

def run_fold(test_year, W, y, years, now_idx):
    val_year = test_year - 1
    train_mask = years < val_year
    val_mask = years == val_year
    test_mask = years == test_year

    W_train = W[train_mask]
    W_val = W[val_mask]
    W_test = W[test_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]

    # Persistence: the "now" flag at the last hour of each window
    f1_persist = f1_score(y_test, W_test[:, -1, now_idx].astype(int))

    # XGBoost on the same rows, using the last hour's features
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="logloss", random_state=42)
    xgb.fit(W_train[:, -1, :], y_train)
    t = best_threshold(y_val, xgb.predict_proba(W_val[:, -1, :])[:, 1])
    preds = (xgb.predict_proba(W_test[:, -1, :])[:, 1] >= t).astype(int)
    f1_xgb = f1_score(y_test, preds)

    # LSTM: scale using training statistics only
    mean = W_train[:, -1, :].mean(axis=0)
    std = W_train[:, -1, :].std(axis=0)
    std[std == 0] = 1.0
    model = train_lstm((W_train - mean) / std, y_train,
                       (W_val - mean) / std, y_val, W.shape[2])
    t = best_threshold(y_val, predict_probs(model, (W_val - mean) / std))
    preds = (predict_probs(model, (W_test - mean) / std) >= t).astype(int)
    f1_lstm = f1_score(y_test, preds)

    return f1_persist, f1_xgb, f1_lstm

for horizon in HORIZONS:
    filename = "data/" + station + "_h" + str(horizon) + "_features.csv"
    df = pd.read_csv(filename, index_col="hour", parse_dates=True)
    feature_cols = []
    for c in df.columns:
        if c != "future_ifr":
            feature_cols.append(c)

    W, y, times = make_windows(df, feature_cols, SEQ_LEN)
    years = np.array(times.year)
    now_idx = feature_cols.index("now_ifr")
    print("")
    print("=== " + station + " horizon " + str(horizon) + " | windows:", len(W), "===")

    scores = {"Persistence": [], "XGBoost": [], "LSTM": []}
    for year in YEARS:
        f1_persist, f1_xgb, f1_lstm = run_fold(year, W, y, years, now_idx)
        scores["Persistence"].append(f1_persist)
        scores["XGBoost"].append(f1_xgb)
        scores["LSTM"].append(f1_lstm)
        print("Test year", year, "| persistence", round(f1_persist, 3),
              "| xgboost", round(f1_xgb, 3), "| lstm", round(f1_lstm, 3))

    summary = {}
    for name in scores:
        avg = np.mean(scores[name])
        print(name, "average F1:", round(avg, 3), "+/-", round(np.std(scores[name]), 3))
        rounded = []
        for x in scores[name]:
            rounded.append(round(x, 4))
        summary[name] = {"mean_f1": round(avg, 4), "fold_f1": rounded}

    if not os.path.exists("results"):
        os.makedirs("results")
    out_name = "results/lstm_" + station + "_h" + str(horizon) + ".json"
    out_file = open(out_name, "w")
    json.dump(summary, out_file, indent=2)
    out_file.close()
    print("Saved " + out_name)