import os
import sys
import requests
import pandas as pd

URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/taf.py"
FIRST_YEAR = 2019
LAST_YEAR = 2025

station = "ATL"
if len(sys.argv) > 1:
    station = sys.argv[1]

if not os.path.exists("data"):
    os.makedirs("data")

for year in range(FIRST_YEAR, LAST_YEAR + 1):
    filename = "data/TAF_" + station + "_" + str(year) + ".csv"
    if os.path.exists(filename):
        print("Already have " + filename)
        continue
    params = {
        "station": station,
        "sts": str(year) + "-01-01T00:00Z",
        "ets": str(year + 1) + "-01-01T00:00Z",
        "fmt": "csv",
    }
    response = requests.get(URL, params=params, timeout=300)
    response.raise_for_status()
    out_file = open(filename, "w")
    out_file.write(response.text)
    out_file.close()
    print("Saved " + filename + " (" + str(len(response.text)) + " characters)")

# Quick check of what kinds of forecast rows this airport has
frames = []
for year in range(FIRST_YEAR, LAST_YEAR + 1):
    filename = "data/TAF_" + station + "_" + str(year) + ".csv"
    frames.append(pd.read_csv(filename))
taf = pd.concat(frames, ignore_index=True)
print(station, "TAF rows:", len(taf))
print(taf["ftype"].value_counts())