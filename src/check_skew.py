import io
import sys
import json
import datetime
import numpy as np
import pandas as pd
import requests
from xgboost import XGBClassifier
from pipeline import URL, DATA_COLUMNS, FEATURE_COLS
from live_data import fetch_live_observations, features_from_observations

station = "ATL"
if len(sys.argv) > 1:
    station = sys.argv[1]
DAYS = 7

def download_recent_iem(station, days):
    end = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=days + 1)
    params = {
        "station": station,
        "data": DATA_COLUMNS,
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "missing": "M",
        "trace": "T",
        "report_type": [3, 4],
    }
    response = requests.get(URL, params=params, timeout=120)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text), na_values=["M"])
    df["valid"] = pd.to_datetime(df["valid"])
    df = df.sort_values("valid")
    df = df.drop_duplicates(subset="valid")
    return df.reset_index(drop=True)

iem_obs = download_recent_iem(station, DAYS)
awc_obs = fetch_live_observations(station, DAYS * 24)
print("IEM observations:", len(iem_obs), "| AWC observations:", len(awc_obs))

iem_features = features_from_observations(iem_obs)
awc_features = features_from_observations(awc_obs)

# Compare only hours where both sources have complete features
common = iem_features.index.intersection(awc_features.index)
iem_f = iem_features.loc[common, FEATURE_COLS]
awc_f = awc_features.loc[common, FEATURE_COLS]
both_ok = iem_f.notnull().all(axis=1) & awc_f.notnull().all(axis=1)
iem_f = iem_f[both_ok]
awc_f = awc_f[both_ok]
print("Hours compared:", len(iem_f))

print("")
print("Feature | typical size (IEM) | mean abs diff | max abs diff")
for col in FEATURE_COLS:
    diff = (iem_f[col] - awc_f[col]).abs()
    typical = iem_f[col].abs().mean()
    print(col, "|", round(typical, 3), "|", round(diff.mean(), 4), "|", round(diff.max(), 4))

print("")
print("Model predictions from each source:")
print("Horizon | mean abs prob diff | max abs prob diff | IFR flag agreement | IFR flags (IEM) | IFR flags (AWC)")
for horizon in [1, 2, 3, 4, 5, 6]:
    base = "models/" + station + "_h" + str(horizon)
    in_file = open(base + "_info.json", "r")
    info = json.load(in_file)
    in_file.close()

    model = XGBClassifier()
    model.load_model(base + ".json")
    cols = info["features"]

    p_iem = model.predict_proba(iem_f[cols])[:, 1]
    p_awc = model.predict_proba(awc_f[cols])[:, 1]
    diff = np.abs(p_iem - p_awc)
    flags_iem = (p_iem >= info["threshold"]).astype(int)
    flags_awc = (p_awc >= info["threshold"]).astype(int)
    agree = (flags_iem == flags_awc).mean()
    print(horizon, "|", round(diff.mean(), 4), "|", round(diff.max(), 3), "|",
          round(agree, 4), "|", flags_iem.sum(), "|", flags_awc.sum())