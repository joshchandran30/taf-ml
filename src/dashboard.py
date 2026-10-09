import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

STATIONS = ["ATL", "ORD", "SEA", "BOS", "DFW", "LAX"]

st.set_page_config(page_title="IFR forecast", layout="wide")
st.title("Airport IFR forecast, 1 to 6 hours ahead")
st.warning("Experimental research project. Not an official weather product. "
           "Do not use for flight planning; get an official weather briefing.")

st.sidebar.header("Settings")
station = st.sidebar.selectbox("Airport", STATIONS)
api_url = st.sidebar.text_input("API address", "http://127.0.0.1:8000")
st.sidebar.button("Refresh")

# Ask the API for the forecast
try:
    response = requests.get(api_url + "/predict/" + station, timeout=30)
except requests.exceptions.RequestException:
    st.error("Could not reach the API at " + api_url + ". Is it running? "
             "Start it with: python -m uvicorn api:app --app-dir src")
    st.stop()

if response.status_code != 200:
    detail = response.json().get("detail", "Unknown error")
    st.error("The API returned an error: " + str(detail))
    st.stop()

data = response.json()

# Current conditions
col1, col2, col3 = st.columns(3)
col1.metric("Airport", data["station"])
if data["currently_ifr"]:
    col2.metric("Right now", "IFR or worse")
else:
    col2.metric("Right now", "VFR or MVFR")
col3.metric("Latest report (UTC)", data["latest_observation_utc"])
st.code(data["raw_metar"])

# Gather the forecast values
hours = []
probs = []
thresholds = []
colors = []
rows = []
weak = []
for p in data["predictions"]:
    hours.append(p["horizon_hours"])
    probs.append(p["probability_ifr"] * 100)
    thresholds.append(p["threshold"] * 100)

    if p["ifr_expected"]:
        colors.append("#d9534f")
        alert = "Yes"
    else:
        colors.append("#5b8def")
        alert = "No"

    if p["model_adds_skill"]:
        adds_skill = "Yes"
    else:
        adds_skill = "No"
        weak.append(p["horizon_hours"])

    rows.append({
        "Hours ahead": p["horizon_hours"],
        "Valid hour (UTC)": p["valid_hour_utc"],
        "IFR probability (%)": round(p["probability_ifr"] * 100),
        "Above alert level": alert,
        "Test F1: model": p["cv_f1_model"],
        "Test F1: 'stays the same'": p["cv_f1_persistence"],
        "Model beats 'stays the same'": adds_skill,
    })

# Chart: bars are probabilities, black ticks are each horizon's alert level
st.subheader("Chance of IFR conditions")
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(hours, probs, color=colors)
ax.plot(hours, thresholds, marker="_", markersize=36, linestyle="none",
        color="black", label="Alert level")
ax.set_xlabel("Hours after the latest observation")
ax.set_ylabel("Probability of IFR (%)")
ax.set_ylim(0, 100)
ax.set_xticks(hours)
ax.legend()
st.pyplot(fig)

st.subheader("Details")
st.dataframe(pd.DataFrame(rows), hide_index=True)

if len(weak) > 0:
    weak_text = ""
    for h in weak:
        if weak_text != "":
            weak_text = weak_text + ", "
        weak_text = weak_text + "+" + str(h) + "h"
    st.info("In testing, the model was no better than assuming conditions stay the same at: "
            + weak_text + ". Treat those bars with extra caution.")

with st.expander("How to read this"):
    st.write("Each bar is the model's estimated chance that the airport will be IFR "
             "(ceiling below 1,000 ft or visibility below 3 miles) at that hour. "
             "The black tick on each bar is that horizon's alert level, chosen to give "
             "the best F1 score in testing. IFR is rare, so alert levels sit well below "
             "50%: a red bar means elevated risk, not a certainty.")
    st.write("The test scores come from five years of held-out testing. 'Stays the same' "
             "means predicting that the airport will be in whatever state it is in now. "
             "A model that doesn't beat it at some horizon adds little there.")
    st.write("Data: current METAR reports from the Aviation Weather Center. "
             "Models: XGBoost trained on 2019 to 2023 observations. "
             "Valid hour is the UTC hour of the report the forecast refers to.")