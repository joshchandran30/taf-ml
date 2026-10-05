import os
import numpy as np
import pandas as pd
import requests

URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

DATA_COLUMNS = ["tmpf", "dwpf", "relh", "drct", "sknt", "gust",
                "alti", "mslp", "vsby", "skyc1", "skyc2", "skyc3",
                "skyc4", "skyl1", "skyl2", "skyl3", "skyl4", "wxcodes"]

FEATURE_COLS = ["tmpf", "dwpf", "relh", "sknt", "gust", "alti", "mslp",
                "vsby", "ceiling", "spread", "wind_sin", "wind_cos",
                "pressure_chg_1h", "pressure_chg_3h", "pressure_chg_6h",
                "spread_chg_3h", "vsby_chg_3h", "temp_chg_3h",
                "vsby_min_6h", "ceiling_min_6h", "ifr_hours_6h",
                "wx_FG", "wx_BR", "wx_RA", "wx_TS",
                "hour_of_day", "month", "now_ifr"]

def download_station(station, first_year, last_year):
    if not os.path.exists("data"):
        os.makedirs("data")
    for year in range(first_year, last_year + 1):
        filename = "data/" + station + "_" + str(year) + ".csv"
        if os.path.exists(filename):
            print("Already have " + filename)
            continue
        params = {
            "station": station,
            "data": DATA_COLUMNS,
            "year1": year, "month1": 1, "day1": 1,
            "year2": year + 1, "month2": 1, "day2": 1,
            "tz": "Etc/UTC",
            "format": "onlycomma",
            "latlon": "no",
            "missing": "M",
            "trace": "T",
            "report_type": [3, 4],
        }
        response = requests.get(URL, params=params, timeout=120)
        response.raise_for_status()
        out_file = open(filename, "w")
        out_file.write(response.text)
        out_file.close()
        print("Saved " + filename)

def load_station(station, first_year, last_year):
    # Loop over exact years, so we never pick up our own output files
    frames = []
    for year in range(first_year, last_year + 1):
        filename = "data/" + station + "_" + str(year) + ".csv"
        if os.path.exists(filename):
            frames.append(pd.read_csv(filename, na_values=["M"]))
    df = pd.concat(frames, ignore_index=True)
    df["valid"] = pd.to_datetime(df["valid"])
    df = df.sort_values("valid")
    df = df.drop_duplicates(subset="valid")
    df = df.reset_index(drop=True)
    return df

def get_ceiling(row):
    lowest = 99999
    for i in range(1, 5):
        cover = row["skyc" + str(i)]
        height = row["skyl" + str(i)]
        if cover in ["BKN", "OVC", "VV"] and pd.notnull(height):
            if height < lowest:
                lowest = height
    return lowest

def get_category(row):
    ceiling = row["ceiling"]
    vis = row["vsby"]
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

def label_observations(df):
    df = df.dropna(subset=["vsby"]).copy()
    df["ceiling"] = df.apply(get_ceiling, axis=1)
    df["category"] = df.apply(get_category, axis=1)
    return df

def make_hourly(df, horizon):
    df = df.copy()
    df["hour"] = df["valid"].dt.floor("h")
    df = df.drop_duplicates(subset="hour", keep="last")
    df = df.set_index("hour")

    full_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_range)

    # The target: the category "horizon" hours from now
    df["future_category"] = df["category"].shift(-horizon)

    # Drop rows where the current or future category is unknown
    # (same order of steps as the earlier scripts, so results match)
    df = df.dropna(subset=["category", "future_category"]).copy()
    df["now_ifr"] = df["category"].apply(is_ifr)
    df["future_ifr"] = df["future_category"].apply(is_ifr)
    return df

def has_code(value, code):
    if pd.isnull(value):
        return 0
    if code in str(value):
        return 1
    return 0

def add_features(df):
    # Rebuild the full hourly timeline so "3 rows back" always means 3 hours
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_range)

    df["gust"] = df["gust"].fillna(0)
    df["spread"] = df["tmpf"] - df["dwpf"]
    df["wind_sin"] = np.sin(np.radians(df["drct"])).fillna(0)
    df["wind_cos"] = np.cos(np.radians(df["drct"])).fillna(0)

    df["pressure_chg_1h"] = df["alti"].diff(1)
    df["pressure_chg_3h"] = df["alti"].diff(3)
    df["pressure_chg_6h"] = df["alti"].diff(6)
    df["spread_chg_3h"] = df["spread"].diff(3)
    df["vsby_chg_3h"] = df["vsby"].diff(3)
    df["temp_chg_3h"] = df["tmpf"].diff(3)

    df["vsby_min_6h"] = df["vsby"].rolling(6).min()
    df["ceiling_min_6h"] = df["ceiling"].rolling(6).min()
    df["ifr_hours_6h"] = df["now_ifr"].rolling(6).sum()

    for code in ["FG", "BR", "RA", "TS"]:
        df["wx_" + code] = df["wxcodes"].apply(has_code, args=(code,))

    df["hour_of_day"] = df.index.hour
    df["month"] = df.index.month

    out = df[FEATURE_COLS + ["future_ifr"]].dropna().copy()
    out["future_ifr"] = out["future_ifr"].astype(int)
    return out

def build_dataset(station, horizon, first_year, last_year):
    raw = load_station(station, first_year, last_year)
    labeled = label_observations(raw)
    hourly = make_hourly(labeled, horizon)
    return add_features(hourly)