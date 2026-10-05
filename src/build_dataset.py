import sys
from pipeline import download_station, build_dataset

FIRST_YEAR = 2019
LAST_YEAR = 2025

station = "ATL"
horizon = 3
if len(sys.argv) > 1:
    station = sys.argv[1]
if len(sys.argv) > 2:
    horizon = int(sys.argv[2])

print("Station:", station, "| Horizon:", horizon, "hours")
download_station(station, FIRST_YEAR, LAST_YEAR)
features = build_dataset(station, horizon, FIRST_YEAR, LAST_YEAR)

print(features.shape)
print(features["future_ifr"].value_counts())

filename = "data/" + station + "_h" + str(horizon) + "_features.csv"
features.to_csv(filename, index_label="hour")
print("Saved " + filename)