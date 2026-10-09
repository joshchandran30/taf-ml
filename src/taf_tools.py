import ast
import bisect
import os
import pandas as pd

RANK = {"VFR": 0, "MVFR": 1, "IFR": 2, "LIFR": 3}

def parse_list(value):
    if pd.isnull(value):
        return []
    return ast.literal_eval(value)

def category_from(vis, covers, heights):
    ceiling = 99999
    for i in range(len(covers)):
        cover = covers[i]
        if isinstance(cover, str):
            cover = cover.strip()
        if cover in ["BKN", "OVC", "VV"] and i < len(heights):
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

def worse(a, b):
    if RANK[b] > RANK[a]:
        return b
    return a

def load_tafs(station, first_year, last_year):
    frames = []
    for year in range(first_year, last_year + 1):
        filename = "data/TAF_" + station + "_" + str(year) + ".csv"
        if os.path.exists(filename):
            frames.append(pd.read_csv(filename))
    taf = pd.concat(frames, ignore_index=True)
    taf["valid"] = pd.to_datetime(taf["valid"])
    taf["fx_valid"] = pd.to_datetime(taf["fx_valid"])
    taf["fx_valid_end"] = pd.to_datetime(taf["fx_valid_end"])

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
            cat = category_from(row["visibility"], covers, heights)
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
    return all_tafs, issue_times

def forecasts_at(all_tafs, issue_times, decision_time, target_time):
    # Returns [base, base+TEMPO, base+TEMPO+PROB] categories, or None
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