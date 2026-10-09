import pandas as pd
from pipeline import load_station

STATIONS = ["ATL", "ORD", "SEA", "BOS", "DFW", "LAX"]

def ceiling_of(row, fix):
    lowest = 99999
    for i in range(1, 5):
        cover = row["skyc" + str(i)]
        height = row["skyl" + str(i)]
        if fix and isinstance(cover, str):
            cover = cover.strip()
        if cover in ["BKN", "OVC", "VV"] and pd.notnull(height):
            if height < lowest:
                lowest = height
    return lowest

def category_of(ceiling, vis):
    if ceiling < 500 or vis < 1:
        return "LIFR"
    if ceiling < 1000 or vis < 3:
        return "IFR"
    if ceiling <= 3000 or vis <= 5:
        return "MVFR"
    return "VFR"

def is_ifr_cat(cat):
    return cat == "IFR" or cat == "LIFR"

print("Station | observations | ceiling changes | category changes | IFR-or-worse status changes")
for station in STATIONS:
    df = load_station(station, 2019, 2025)
    df = df.dropna(subset=["vsby"]).copy()
    old_ceiling = df.apply(ceiling_of, axis=1, args=(False,))
    new_ceiling = df.apply(ceiling_of, axis=1, args=(True,))

    differ = df[old_ceiling != new_ceiling]
    changed_cat = 0
    changed_ifr = 0
    for idx in differ.index:
        vis = differ.loc[idx, "vsby"]
        old_cat = category_of(old_ceiling[idx], vis)
        new_cat = category_of(new_ceiling[idx], vis)
        if old_cat != new_cat:
            changed_cat = changed_cat + 1
            if is_ifr_cat(old_cat) != is_ifr_cat(new_cat):
                changed_ifr = changed_ifr + 1

    print(station, "|", len(df), "|", len(differ), "|", changed_cat, "|", changed_ifr)