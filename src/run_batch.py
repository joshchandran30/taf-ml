import os
import sys
import json
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

STATIONS = ["ATL"]
HORIZONS = [1, 2, 3, 4, 5, 6]

# Run the pipeline for every station and horizon that isn't done yet
for station in STATIONS:
    for horizon in HORIZONS:
        result_file = "results/cv_" + station + "_h" + str(horizon) + ".json"
        if os.path.exists(result_file):
            print("Skipping " + station + " horizon " + str(horizon) + " (already done)")
            continue
        print("=== " + station + " horizon " + str(horizon) + " ===")
        subprocess.run([sys.executable, "src/build_dataset.py", station, str(horizon)])
        subprocess.run([sys.executable, "src/run_cv.py", station, str(horizon)])

# Collect all results into one table
rows = []
for station in STATIONS:
    for horizon in HORIZONS:
        result_file = "results/cv_" + station + "_h" + str(horizon) + ".json"
        if not os.path.exists(result_file):
            continue
        in_file = open(result_file, "r")
        data = json.load(in_file)
        in_file.close()
        rows.append({
            "station": station,
            "horizon": horizon,
            "persistence": data["Persistence"]["mean_f1"],
            "logreg": data["Logistic regression"]["mean_f1"],
            "xgboost": data["XGBoost"]["mean_f1"],
        })

summary = pd.DataFrame(rows)
summary["xgb_minus_persistence"] = (summary["xgboost"] - summary["persistence"]).round(3)
print("")
print(summary)
summary.to_csv("results/summary.csv", index=False)

# One chart per station: skill versus horizon
for station in STATIONS:
    sub = summary[summary["station"] == station]
    if len(sub) == 0:
        continue
    plt.figure(figsize=(7, 5))
    plt.plot(sub["horizon"], sub["persistence"], marker="o", label="Persistence")
    plt.plot(sub["horizon"], sub["logreg"], marker="o", label="Logistic regression")
    plt.plot(sub["horizon"], sub["xgboost"], marker="o", label="XGBoost")
    plt.xlabel("Forecast horizon (hours)")
    plt.ylabel("Average F1 on IFR (test years 2021-2025)")
    plt.title(station + ": IFR forecast skill by horizon")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/horizons_" + station + ".png", dpi=150)
    plt.close()
    print("Saved results/horizons_" + station + ".png")