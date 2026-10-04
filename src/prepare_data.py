import pandas as pd
import glob

def load_station(station):
    files = glob.glob("data/" + station + "_*.csv")
    frames = []
    for f in files:
        frames.append(pd.read_csv(f, na_values=["M"]))
    df = pd.concat(frames, ignore_index=True)

    df["valid"] = pd.to_datetime(df["valid"])
    df = df.sort_values("valid")
    df = df.drop_duplicates(subset="valid")
    df = df.reset_index(drop=True)
    return df

def get_ceiling(row):
    # Ceiling = lowest broken/overcast layer or vertical visibility
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

df = load_station("ATL")
df = df.dropna(subset=["vsby"])
df["ceiling"] = df.apply(get_ceiling, axis=1)
df["category"] = df.apply(get_category, axis=1)

print(df.shape)
print(df[["valid", "vsby", "ceiling", "category"]].head(10))
print(df["category"].value_counts())
print(df["category"].value_counts(normalize=True))

df.to_csv("data/ATL_clean.csv", index=False)