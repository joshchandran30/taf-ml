import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 50)

df = pd.read_csv("data/TAF_ATL_2024.csv")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("")
print(df.drop(columns=["raw"]).head(15))
print("")
print(df["ftype"].value_counts())
print("")
print("Distinct TAF issuances:", df["valid"].nunique())
print("")
print("One full raw TAF:")
print(df["raw"].iloc[0])