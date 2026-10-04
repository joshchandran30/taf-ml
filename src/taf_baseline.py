import ast
import bisect
import glob
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 40)

def parse_list(value):
    # Turns the text "['FEW', 'SCT']" back into a real Python list
    if pd.isnull(value):
        return []
    return ast.literal_eval(value)

def get_category(vis, covers, heights):
    ceiling = 99999
    for i in range(len(covers)):
        if covers[i] in ["BKN", "OVC", "VV"] and i < len(heights):
            if heights[i] is not None and heights[i] < ceiling:
                ceiling = heights[i]
    if pd.isnull(vis):
        vis = 6.01
    if ceiling < 500 or vis < 1:
        return "LIFR"
    if ceiling < 1000 or vis < 3:
        return "IFR"
    if ceiling <= 3000 or vis <= 5:
        return "MVFR"
    return "VFR"

def is_ifr(cat):
    if cat == "IFR" or cat == "LIFR":
        return 1
    return 0

# Load every TAF file
frames = []
for f in glob.glob("data/TAF_ATL_*.csv"):
    frames.append(pd.read_csv(f))
taf = pd.concat(frames, ignore_index=True)
taf["valid"] = pd.to_datetime(taf["valid"])
taf["fx_valid"] = pd.to_datetime(taf["fx_valid"])
print("Total TAF rows:", len(taf))

# Keep only the base forecast groups for now
base = taf[(taf["ftype"] == "Observation") | (taf["ftype"] == "Forecast")]
print("Base group rows:", len(base))
print("Base groups with missing visibility:", base["visibility"].isnull().sum())

# Build one entry per TAF: when it was issued, when each group starts,
# and the flight category of each group
all_tafs = []
for product_id, grp in base.groupby("product_id"):
    grp = grp.sort_values("fx_valid")
    issue_time = grp["valid"].iloc[0]
    starts = []
    cats = []
    for idx, row in grp.iterrows():
        covers = parse_list(row["skyc"])
        heights = parse_list(row["skyl"])
        starts.append(row["fx_valid"])
        cats.append(get_category(row["visibility"], covers, heights))
    all_tafs.append((issue_time, product_id, starts, cats))

all_tafs.sort()
issue_times = []
for t in all_tafs:
    issue_times.append(t[0])
print("TAFs loaded:", len(all_tafs))

def taf_category_at(decision_time, target_time):
    # bisect quickly finds the latest TAF issued at or before decision_time
    pos = bisect.bisect_right(issue_times, decision_time) - 1
    if pos < 0:
        return None
    issue_time, product_id, starts, cats = all_tafs[pos]
    if decision_time - issue_time > pd.Timedelta(hours=8):
        return None
    answer = None
    for i in range(len(starts)):
        if starts[i] <= target_time:
            answer = cats[i]
    return answer

# For every hour in the feature table, get the TAF's forecast 3 hours ahead
feat = pd.read_csv("data/ATL_features.csv", index_col="hour", parse_dates=True)

forecast_cats = []
for hour in feat.index:
    decision_time = hour + pd.Timedelta(minutes=59)
    target_time = decision_time + pd.Timedelta(hours=3)
    forecast_cats.append(taf_category_at(decision_time, target_time))

result = pd.DataFrame({"taf_category": forecast_cats}, index=feat.index)
result["future_ifr"] = feat["future_ifr"]
result["now_ifr"] = feat["now_ifr"]
result["has_taf"] = result["taf_category"].notnull()
result["taf_ifr"] = result["taf_category"].apply(is_ifr)

print("")
print("Hours with a TAF forecast:", result["has_taf"].sum(), "of", len(result))
print(result["taf_category"].value_counts())
print(result.head(8))

# Score it, year by year, on hours that have a TAF forecast
scored = result[result["has_taf"]]
print("")
print("Year | TAF F1 | Persistence F1 (same rows)")
for year in [2021, 2022, 2023, 2024, 2025]:
    sub = scored[scored.index.year == year]
    taf_f1 = f1_score(sub["future_ifr"], sub["taf_ifr"])
    pers_f1 = f1_score(sub["future_ifr"], sub["now_ifr"])
    print(year, "|", round(taf_f1, 3), "|", round(pers_f1, 3))

recent = scored[scored.index.year >= 2024]
print("")
print("TAF on 2024-2025:")
print("  precision:", round(precision_score(recent["future_ifr"], recent["taf_ifr"]), 3))
print("  recall:   ", round(recall_score(recent["future_ifr"], recent["taf_ifr"]), 3))
print("  F1:       ", round(f1_score(recent["future_ifr"], recent["taf_ifr"]), 3))

result.to_csv("data/ATL_taf_baseline.csv", index_label="hour")

# A peek at TEMPO and PROB30 rows so we can handle them next
print("")
print("Sample TEMPO / PROB30 rows:")
other = taf[(taf["ftype"] == "Temporary") | (taf["ftype"] == "Probability 30")]
print(other[["valid", "fx_valid", "fx_valid_end", "visibility", "skyc", "skyl", "ftype", "raw"]].head(8))