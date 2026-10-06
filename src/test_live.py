import sys
import pandas as pd
from live_data import fetch_live_observations, build_live_features

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 110)

station = "ATL"
if len(sys.argv) > 1:
    station = sys.argv[1]

obs = fetch_live_observations(station, 24)
print("Observations fetched:", len(obs))
print(obs[["valid", "tmpf", "dwpf", "relh", "drct", "sknt", "gust", "alti", "mslp",
           "vsby", "skyc1", "skyl1", "wxcodes"]].tail(6))
print("")
print(obs[["valid", "raw"]].tail(3))

features, message = build_live_features(station, 24)
print("")
print("Status:", message)
if features is not None:
    print("Features for hour:", features.index[0])
    print(features.T)