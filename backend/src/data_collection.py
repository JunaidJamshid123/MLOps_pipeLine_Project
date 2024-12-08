import requests
import pandas as pd
from datetime import datetime

API_KEY = "ed1d944ff7ad3ff52830c829d5f89d23"
CITY = "Islamabad"
URL = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}"

def fetch_weather_data():
    response = requests.get(URL)
    data = response.json()
    weather_list = []
    for entry in data['list']:
        weather_list.append({
            "Date": entry['dt_txt'],
            "Temperature": entry['main']['temp'] - 273.15,  # Convert from Kelvin to Celsius
            "Humidity": entry['main']['humidity'],
            "Wind Speed": entry['wind']['speed'],
            "Weather Condition": entry['weather'][0]['description'],
        })
    df = pd.DataFrame(weather_list)
    df.to_csv("data/raw/raw_data.csv", index=False)

if __name__ == "__main__":
    fetch_weather_data()
