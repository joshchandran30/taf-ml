import ast
import bisect
import glob
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

RANK = {"VFR": 0, "MVFR": 1, "IFR": 2, "LIFR": 3}

def parse_list(value):
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

def worse(a, b):
    if RANK[b] > RANK[a]:
        return b
    return a

# Load every TAF file
frames = []
for f in glob.glob("data/TAF_ATL_*.csv"):
    frames.append(pd.read_csv(f))
taf = pd.concat(frames, ignore_index=True)
taf["valid"] = pd.to_datetime(taf["valid"])
taf["fx_valid"] = pd.to_datetime(taf["fx_valid"])
taf["fx_valid_end"] = pd.to_datetime(taf["fx_valid_end"])
print("Total TAF rows:", len(taf))
print(taf["ftype"].value_counts())

# One entry per TAF: issue time, base groups, TEMPO windows, PROB windows
all_tafs = []
for product_id, grp in taf.groupby("product_id"):
    grp = grp.sort_values("fx_valid")
    issue_time = grp["valid"].iloc[0]
    starts = []
    cats = []
    tempo = []
    prob = []
    for idx, row in grp.iterrows():
        covers = parse_list(row["skyc"])
        heights = parse_list(row["skyl"])
        cat = get_category(row["visibility"], covers, heights)
        ftype = row["ftype"]
        if ftype == "Observation" or ftype == "Forecast":
            starts.append(row["fx_valid"])
            cats.append(cat)
        elif pd.notnull(row["fx_valid_end"]):
            window = (row["fx_valid"], row["fx_valid_end"], cat)
            if ftype == "Temporary":
                tempo.append(window)
            elif ftype.startswith("Probability"):
                prob.append(window)
    all_tafs.append((issue_time, product_id, starts, cats, tempo, prob))

all_tafs.sort()
issue_times = []
for t in all_tafs:
    issue_times.append(t[0])
print("TAFs loaded:", len(all_tafs))

def taf_forecasts_at(decision_time, target_time):
    pos = bisect.bisect_right(issue_times, decision_time) - 1
    if pos < 0:
        return None
    issue_time, product_id, starts, cats, tempo, prob = all_tafs[pos]
    if decision_time - issue_time > pd.Timedelta(hours=8):
        return None

    base_cat = None
    for i in range(len(starts)):
        if starts[i] <= target_time:
            base_cat = cats[i]
    if base_cat is None:
        return None

    with_tempo = base_cat
    for window in tempo:
        if window[0] <= target_time and target_time < window[1]:
            with_tempo = worse(with_tempo, window[2])

    with_all = with_tempo
    for window in prob:
        if window[0] <= target_time and target_time < window[1]:
            with_all = worse(with_all, window[2])

    return [base_cat, with_tempo, with_all]

# Ask the question for every hour in the feature table
feat = pd.read_csv("data/ATL_features.csv", index_col="hour", parse_dates=True)

base_list = []
tempo_list = []
all_list = []
for hour in feat.index:
    decision_time = hour + pd.Timedelta(minutes=59)
    target_time = decision_time + pd.Timedelta(hours=3)
    answer = taf_forecasts_at(decision_time, target_time)
    if answer is None:
        base_list.append(None)
        tempo_list.append(None)
        all_list.append(None)
    else:
        base_list.append(answer[0])
        tempo_list.append(answer[1])
        all_list.append(answer[2])

result = pd.DataFrame({"taf_base": base_list, "taf_tempo": tempo_list,
                       "taf_all": all_list}, index=feat.index)
result["future_ifr"] = feat["future_ifr"]
result["now_ifr"] = feat["now_ifr"]
result["has_taf"] = result["taf_base"].notnull()
variants = ["taf_base", "taf_tempo", "taf_all"]
for v in variants:
    result[v + "_ifr"] = result[v].apply(is_ifr)

print("Hours with a TAF forecast:", result["has_taf"].sum(), "of", len(result))
result.to_csv("data/ATL_taf_baseline.csv", index_label="hour")

# Score each variant, year by year
scored = result[result["has_taf"]]
years = [2021, 2022, 2023, 2024, 2025]
yearly = {}
for v in variants:
    yearly[v] = []
    for year in years:
        sub = scored[scored.index.year == year]
        yearly[v].append(f1_score(sub["future_ifr"], sub[v + "_ifr"]))

# Precision and recall on the 2024-2025 holdout
print("")
print("TAF variants on 2024-2025:")
recent = scored[scored.index.year >= 2024]
holdout = {}
for v in variants:
    p = precision_score(recent["future_ifr"], recent[v + "_ifr"])
    r = recall_score(recent["future_ifr"], recent[v + "_ifr"])
    f = f1_score(recent["future_ifr"], recent[v + "_ifr"])
    print(v, "| precision", round(p, 3), "| recall", round(r, 3), "| F1", round(f, 3))
    holdout[v] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}

# Combined table with the models from cross-validation
cv_file = open("results/cv_metrics.json", "r")
cv = json.load(cv_file)
cv_file.close()

print("")
print("Year | Persistence | LogReg | XGBoost | TAF base | TAF+TEMPO | TAF+TEMPO+PROB")
for i in range(len(years)):
    print(years[i], "|",
          round(cv["Persistence"]["fold_f1"][i], 3), "|",
          round(cv["Logistic regression"]["fold_f1"][i], 3), "|",
          round(cv["XGBoost"]["fold_f1"][i], 3), "|",
          round(yearly["taf_base"][i], 3), "|",
          round(yearly["taf_tempo"][i], 3), "|",
          round(yearly["taf_all"][i], 3))
print("Average |",
      round(cv["Persistence"]["mean_f1"], 3), "|",
      round(cv["Logistic regression"]["mean_f1"], 3), "|",
      round(cv["XGBoost"]["mean_f1"], 3), "|",
      round(np.mean(yearly["taf_base"]), 3), "|",
      round(np.mean(yearly["taf_tempo"]), 3), "|",
      round(np.mean(yearly["taf_all"]), 3))

if not os.path.exists("results"):
    os.makedirs("results")
out_file = open("results/taf_comparison.json", "w")
json.dump({"years": years, "taf_yearly_f1": yearly, "taf_holdout_2024_2025": holdout},
          out_file, indent=2)
out_file.close()