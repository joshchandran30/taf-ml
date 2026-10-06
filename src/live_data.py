import numpy as np
import pandas as pd
import requests
from pipeline import label_observations, is_ifr, compute_features, FEATURE_COLS

AWC_URL = "https://aviationweather.gov/api/data/metar"
HEADERS = {"User-Agent": "taf-ml-student-project"}

def fetch_awc(station, hours):
    icao = station
    if len(station) == 3:
        icao = "K" + station
    params = {"ids": icao, "format": "json", "hours": hours}
    response = requests.get(AWC_URL, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    if response.status_code == 204 or response.text.strip() == "":
        return []
    return response.json()

def parse_visibility(value):
    if value is None:
        return np.nan
    if not isinstance(value, str):
        return float(value)
    text = value.replace("+", "").strip()
    total = 0.0
    try:
        for part in text.split(" "):
            if "/" in part:
                pieces = part.split("/")
                total = total + float(pieces[0]) / float(pieces[1])
            elif part != "":
                total = total + float(part)
    except ValueError:
        return np.nan
    return total

def celsius_to_f(value):
    if value is None:
        return np.nan
    return value * 9.0 / 5.0 + 32.0

def relative_humidity(temp_c, dewp_c):
    if temp_c is None or dewp_c is None:
        return np.nan
    sat_temp = np.exp(17.625 * temp_c / (243.04 + temp_c))
    sat_dew = np.exp(17.625 * dewp_c / (243.04 + dewp_c))
    return 100.0 * sat_dew / sat_temp

def convert_observation(ob):
    # Turn one AWC report into one row in the same format as the training data
    row = {}
    row["valid"] = pd.to_datetime(ob["obsTime"], unit="s")
    row["raw"] = ob.get("rawOb")
    row["tmpf"] = celsius_to_f(ob.get("temp"))
    row["dwpf"] = celsius_to_f(ob.get("dewp"))
    row["relh"] = relative_humidity(ob.get("temp"), ob.get("dewp"))

    wdir = ob.get("wdir")
    if wdir is None or isinstance(wdir, str):
        row["drct"] = np.nan
    else:
        row["drct"] = float(wdir)

    wspd = ob.get("wspd")
    if wspd is None:
        wspd = 0
    row["sknt"] = float(wspd)

    wgst = ob.get("wgst")
    if wgst is None:
        row["gust"] = np.nan
    else:
        row["gust"] = float(wgst)

    altim = ob.get("altim")
    if altim is None:
        row["alti"] = np.nan
    else:
        row["alti"] = round(altim / 33.8639, 2)

    slp = ob.get("slp")
    if slp is None:
        slp = altim
    if slp is None:
        row["mslp"] = np.nan
    else:
        row["mslp"] = float(slp)

    row["vsby"] = parse_visibility(ob.get("visib"))

    layers = ob.get("clouds")
    if layers is None:
        layers = []
    for i in range(4):
        cover = np.nan
        base = np.nan
        if i < len(layers):
            cover = layers[i].get("cover")
            base = layers[i].get("base")
            if base is None:
                base = np.nan
            if cover == "OVX":
                cover = "VV"
                if ob.get("vertVis") is not None:
                    base = float(ob.get("vertVis"))
            elif cover == "CLR" or cover == "CAVOK":
                cover = "CLR"
                base = np.nan
        row["skyc" + str(i + 1)] = cover
        row["skyl" + str(i + 1)] = base

    wx = ob.get("wxString")
    if wx is None:
        row["wxcodes"] = np.nan
    else:
        row["wxcodes"] = wx
    return row

def fetch_live_observations(station, hours):
    raw_obs = fetch_awc(station, hours)
    rows = []
    for ob in raw_obs:
        rows.append(convert_observation(ob))
    if len(rows) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("valid")
    df = df.drop_duplicates(subset="valid")
    df = df.dropna(subset=["vsby"])
    return df.reset_index(drop=True)

def features_from_observations(obs):
    # Observations (from either source) -> table of features for every hour
    labeled = label_observations(obs)
    labeled["hour"] = labeled["valid"].dt.floor("h")
    labeled = labeled.drop_duplicates(subset="hour", keep="last")
    labeled = labeled.set_index("hour")

    full_range = pd.date_range(labeled.index.min(), labeled.index.max(), freq="h")
    hourly = labeled.reindex(full_range)
    hourly["now_ifr"] = hourly["category"].apply(is_ifr).astype(float)
    hourly.loc[hourly["category"].isnull(), "now_ifr"] = np.nan
    return compute_features(hourly)

def build_live_features(station, hours, max_age_hours=3):
    # Returns (one-row table of the 28 features, "ok") or (None, reason)
    obs = fetch_live_observations(station, hours)
    if len(obs) == 0:
        return None, "No observations returned"

    features = features_from_observations(obs)

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    if now - features.index[-1] > pd.Timedelta(hours=max_age_hours):
        return None, "Latest observation is more than " + str(max_age_hours) + " hours old"

    latest = features.iloc[[-1]][FEATURE_COLS]
    if latest.isnull().any(axis=1).iloc[0]:
        return None, "Not enough consecutive hourly observations to compute features"
    return latest, "ok"