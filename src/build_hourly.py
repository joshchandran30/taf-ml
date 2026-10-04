import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("data/ATL_clean.csv")
df["valid"] = pd.to_datetime(df["valid"])

# Round each timestamp down to the hour (02:52 becomes 02:00)
df["hour"] = df["valid"].dt.floor("h")

# If several reports fall in one hour, keep the last one
df = df.drop_duplicates(subset="hour", keep="last")
df = df.set_index("hour")

# Build a complete hourly timeline so missing hours show up as empty rows
full_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
df = df.reindex(full_range)
print("Hours in timeline:", len(df))
print("Hours with no report:", df["category"].isnull().sum())

# The target: what will the category be 3 hours from now?
HORIZON = 3
df["future_category"] = df["category"].shift(-HORIZON)

def is_ifr(cat):
    if cat == "IFR" or cat == "LIFR":
        return 1
    return 0

# Drop rows where we don't know the current or the future category
df = df.dropna(subset=["category", "future_category"])

df["now_ifr"] = df["category"].apply(is_ifr)
df["future_ifr"] = df["future_category"].apply(is_ifr)

print(df[["category", "future_category", "future_ifr"]].head(10))
print(df["future_ifr"].value_counts())
print(df["future_ifr"].value_counts(normalize=True))

# Baseline 1: persistence (predict that conditions stay the same)
actual = df["future_ifr"]
guess = df["now_ifr"]
print("Persistence precision:", precision_score(actual, guess))
print("Persistence recall:", recall_score(actual, guess))
print("Persistence F1:", f1_score(actual, guess))

df.to_csv("data/ATL_hourly.csv", index_label="hour")