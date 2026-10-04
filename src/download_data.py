import requests
import os

URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

def download_year(station, year):
    params = {
        "station": station,
        "data": ["tmpf", "dwpf", "relh", "drct", "sknt", "gust",
                 "alti", "mslp", "vsby", "skyc1", "skyc2", "skyc3",
                 "skyc4", "skyl1", "skyl2", "skyl3", "skyl4", "wxcodes"],
        "year1": year, "month1": 1, "day1": 1,
        "year2": year + 1, "month2": 1, "day2": 1,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "missing": "M",
        "trace": "T",
        "report_type": [3, 4],   # routine METARs + specials
    }
    response = requests.get(URL, params=params, timeout=120)
    response.raise_for_status()

    filename = "data/" + station + "_" + str(year) + ".csv"
    out_file = open(filename, "w")
    out_file.write(response.text)
    out_file.close()
    print("Saved " + filename)

if not os.path.exists("data"):
    os.makedirs("data")

for year in range(2019, 2026):
    download_year("ATL", year)