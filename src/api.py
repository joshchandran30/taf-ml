import json
import time
import pandas as pd
from fastapi import FastAPI, HTTPException
from xgboost import XGBClassifier
from pipeline import FEATURE_COLS
from live_data import fetch_live_observations, features_from_observations

STATIONS = ["ATL", "ORD", "SEA", "BOS", "DFW", "LAX"]
HORIZONS = [1, 2, 3, 4, 5, 6]
CACHE_SECONDS = 300
MAX_AGE_HOURS = 3
HISTORY_HOURS = 24
DISCLAIMER = ("Experimental research project. Not an official weather product. "
              "Do not use for flight planning; get an official weather briefing.")

app = FastAPI(title="IFR forecast API",
              description="Probability that an airport will be IFR 1 to 6 hours ahead, "
                          "from XGBoost models trained on 2019-2025 observations.")

# Load every saved model, its settings, and its cross-validated skill once at startup
models = {}
infos = {}
skill = {}
for station in STATIONS:
    for horizon in HORIZONS:
        base = "models/" + station + "_h" + str(horizon)
        model = XGBClassifier()
        model.load_model(base + ".json")
        models[(station, horizon)] = model

        in_file = open(base + "_info.json", "r")
        infos[(station, horizon)] = json.load(in_file)
        in_file.close()

        cv_file = open("results/cv_" + station + "_h" + str(horizon) + ".json", "r")
        cv = json.load(cv_file)
        cv_file.close()
        skill[(station, horizon)] = {
            "model": cv["XGBoost"]["mean_f1"],
            "persistence": cv["Persistence"]["mean_f1"],
        }

cache = {}

def get_snapshot(station):
    # Latest observation and its features, reused for CACHE_SECONDS
    now = time.time()
    if station in cache:
        saved_time, saved_value = cache[station]
        if now - saved_time < CACHE_SECONDS:
            return saved_value

    obs = fetch_live_observations(station, HISTORY_HOURS)
    if len(obs) == 0:
        raise HTTPException(status_code=503,
                            detail="No observations available from the Aviation Weather Center")

    features = features_from_observations(obs)
    latest_hour = features.index[-1]
    age = pd.Timestamp.now(tz="UTC").tz_localize(None) - latest_hour
    if age > pd.Timedelta(hours=MAX_AGE_HOURS):
        raise HTTPException(status_code=503,
                            detail="The latest observation is more than " + str(MAX_AGE_HOURS) + " hours old")

    latest = features.iloc[[-1]][FEATURE_COLS]
    if latest.isnull().any(axis=1).iloc[0]:
        raise HTTPException(status_code=503,
                            detail="Not enough consecutive hourly observations to compute features")

    snapshot = {
        "hour": latest_hour,
        "features": latest,
        "raw": obs["raw"].iloc[-1],
        "obs_time": obs["valid"].iloc[-1],
    }
    cache[station] = (now, snapshot)
    return snapshot

@app.get("/")
def home():
    return {"message": "IFR forecast API. Try /airports, /predict/ATL, or /docs.",
            "disclaimer": DISCLAIMER}

@app.get("/airports")
def airports():
    return {"airports": STATIONS, "horizons_hours": HORIZONS}

@app.get("/predict/{station}")
def predict(station: str):
    station = station.upper()
    if station not in STATIONS:
        raise HTTPException(status_code=404,
                            detail="Unsupported airport. Choose from: " + ", ".join(STATIONS))

    snapshot = get_snapshot(station)
    features = snapshot["features"]

    predictions = []
    for horizon in HORIZONS:
        model = models[(station, horizon)]
        info = infos[(station, horizon)]
        probability = float(model.predict_proba(features[info["features"]])[0, 1])
        model_f1 = skill[(station, horizon)]["model"]
        persistence_f1 = skill[(station, horizon)]["persistence"]
        predictions.append({
            "horizon_hours": horizon,
            "valid_hour_utc": str(snapshot["hour"] + pd.Timedelta(hours=horizon)),
            "probability_ifr": round(probability, 3),
            "ifr_expected": probability >= info["threshold"],
            "threshold": info["threshold"],
            "cv_f1_model": model_f1,
            "cv_f1_persistence": persistence_f1,
            "model_adds_skill": model_f1 > persistence_f1 + 0.01,
        })

    return {
        "station": station,
        "latest_observation_utc": str(snapshot["obs_time"]),
        "observation_hour_utc": str(snapshot["hour"]),
        "raw_metar": snapshot["raw"],
        "currently_ifr": bool(features["now_ifr"].iloc[0] == 1.0),
        "predictions": predictions,
        "disclaimer": DISCLAIMER,
    }