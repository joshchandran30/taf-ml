import os
import json
import pandas as pd
import matplotlib.pyplot as plt

STATIONS = ["ATL", "ORD", "SEA", "BOS", "DFW", "LAX"]

data = {}
horizons = []
for station in STATIONS:
    in_file = open("results/taf_horizons_" + station + ".json", "r")
    info = json.load(in_file)
    in_file.close()
    data[station] = info["avg_f1"]
    horizons = info["horizons"]

# Every F1 value in one tidy table
long_rows = []
for station in STATIONS:
    for method in data[station]:
        for i in range(len(horizons)):
            long_rows.append({"station": station, "method": method,
                              "horizon": horizons[i],
                              "avg_f1": data[station][method][i]})
pd.DataFrame(long_rows).to_csv("results/all_airports_f1.csv", index=False)

# The three comparisons: (file name, title, method A, method B) means A minus B
comparisons = [
    ("xgb_vs_persistence", "Local XGBoost minus persistence", "Local XGBoost", "Persistence"),
    ("xgb_vs_taf", "Local XGBoost minus TAF", "Local XGBoost", "TAF base"),
    ("hybrid_vs_taf", "Hybrid XGBoost minus TAF", "Hybrid XGBoost", "TAF base"),
]

tables = []
for name, title, method_a, method_b in comparisons:
    rows = []
    for station in STATIONS:
        row = []
        for i in range(len(horizons)):
            row.append(data[station][method_a][i] - data[station][method_b][i])
        rows.append(row)
    table = pd.DataFrame(rows, index=STATIONS, columns=horizons)
    tables.append(table)
    print("")
    print(title + " (average F1 difference, test years 2021-2025):")
    print(table.round(3))
    table.round(4).to_csv("results/comparison_" + name + ".csv")

# One heatmap figure with three panels
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
image = None
for k in range(3):
    table = tables[k]
    ax = axes[k]
    image = ax.imshow(table.values, cmap="RdBu", vmin=-0.18, vmax=0.18, aspect="auto")
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels(horizons)
    ax.set_yticks(range(len(STATIONS)))
    ax.set_yticklabels(STATIONS)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_title(comparisons[k][1])
    for r in range(len(STATIONS)):
        for c in range(len(horizons)):
            value = table.values[r][c]
            color = "black"
            if abs(value) > 0.1:
                color = "white"
            ax.text(c, r, str(round(value, 3)), ha="center", va="center",
                    fontsize=8, color=color)

plt.colorbar(image, ax=axes, shrink=0.85,
             label="F1 difference (blue = first method better, red = worse)")
if not os.path.exists("results"):
    os.makedirs("results")
plt.savefig("results/airport_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("")
print("Saved results/airport_comparison.png")