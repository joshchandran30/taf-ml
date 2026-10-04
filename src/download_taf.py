import requests
import os

URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/taf.py"

def download_year(station, year):
    params = {
        "station": station,
        "sts": str(year) + "-01-01T00:00Z",
        "ets": str(year + 1) + "-01-01T00:00Z",
        "fmt": "csv",
    }
    response = requests.get(URL, params=params, timeout=300)
    response.raise_for_status()

    filename = "data/TAF_" + station + "_" + str(year) + ".csv"
    out_file = open(filename, "w")
    out_file.write(response.text)
    out_file.close()
    print("Saved " + filename + " (" + str(len(response.text)) + " characters)")

if not os.path.exists("data"):
    os.makedirs("data")

for year in range(2019, 2026):
    download_year("ATL", year)