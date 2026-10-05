from pipeline import download_station, build_dataset

CANDIDATES = ["ATL", "ORD", "DFW", "SFO", "SEA", "JFK", "BOS", "MCO", "DEN", "LAX", "MSP"]
HORIZON = 3
FIRST_YEAR = 2019
LAST_YEAR = 2025

summary_lines = []

for station in CANDIDATES:
    try:
        download_station(station, FIRST_YEAR, LAST_YEAR)
        features = build_dataset(station, HORIZON, FIRST_YEAR, LAST_YEAR)
    except Exception as e:
        summary_lines.append(station + " | failed: " + str(e))
        continue

    filename = "data/" + station + "_h" + str(HORIZON) + "_features.csv"
    features.to_csv(filename, index_label="hour")

    ifr_hours = features["future_ifr"].sum()
    ifr_rate = features["future_ifr"].mean()
    recent = features[features.index.year >= 2024]
    summary_lines.append(station + " | rows " + str(len(features)) +
                         " | IFR hours " + str(ifr_hours) +
                         " | IFR rate " + str(round(ifr_rate, 3)) +
                         " | IFR hours in 2024-25 " + str(recent["future_ifr"].sum()))

print("")
print("=== Summary ===")
for line in summary_lines:
    print(line)