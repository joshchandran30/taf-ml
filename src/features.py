import pandas as pd
import numpy as np

df = pd.read_csv("data/ATL_hourly.csv", index_col="hour", parse_dates=True)

# Rebuild the full hourly timeline so changes over "3 rows" mean 3 hours
full_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
df = df.reindex(full_range)

# Gusts are only reported when there are gusts, so missing means none
df["gust"] = df["gust"].fillna(0)

# Temperature minus dewpoint: small spread means fog and low cloud risk
df["spread"] = df["tmpf"] - df["dwpf"]

# Wind direction as two numbers, so 359 degrees and 1 degree are close
df["wind_sin"] = np.sin(np.radians(df["drct"])).fillna(0)
df["wind_cos"] = np.cos(np.radians(df["drct"])).fillna(0)

# How things have been changing
df["pressure_chg_1h"] = df["alti"].diff(1)
df["pressure_chg_3h"] = df["alti"].diff(3)
df["pressure_chg_6h"] = df["alti"].diff(6)
df["spread_chg_3h"] = df["spread"].diff(3)
df["vsby_chg_3h"] = df["vsby"].diff(3)
df["temp_chg_3h"] = df["tmpf"].diff(3)

# Worst conditions over the last 6 hours
df["vsby_min_6h"] = df["vsby"].rolling(6).min()
df["ceiling_min_6h"] = df["ceiling"].rolling(6).min()
df["ifr_hours_6h"] = df["now_ifr"].rolling(6).sum()

# Weather codes: fog, mist, rain, thunderstorm
def has_code(value, code):
    if pd.isnull(value):
        return 0
    if code in str(value):
        return 1
    return 0

for code in ["FG", "BR", "RA", "TS"]:
    df["wx_" + code] = df["wxcodes"].apply(has_code, args=(code,))

# Time of day and season
df["hour_of_day"] = df.index.hour
df["month"] = df.index.month

feature_cols = ["tmpf", "dwpf", "relh", "sknt", "gust", "alti", "mslp",
                "vsby", "ceiling", "spread", "wind_sin", "wind_cos",
                "pressure_chg_1h", "pressure_chg_3h", "pressure_chg_6h",
                "spread_chg_3h", "vsby_chg_3h", "temp_chg_3h",
                "vsby_min_6h", "ceiling_min_6h", "ifr_hours_6h",
                "wx_FG", "wx_BR", "wx_RA", "wx_TS",
                "hour_of_day", "month", "now_ifr"]

out = df[feature_cols + ["future_ifr"]]
out = out.dropna()
out["future_ifr"] = out["future_ifr"].astype(int)

print(out.shape)
print(out["future_ifr"].value_counts())
out.to_csv("data/ATL_features.csv", index_label="hour")