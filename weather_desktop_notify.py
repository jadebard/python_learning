import requests 
from plyer import notification

# Search API didn't exist, so using the API Docs for Open-Meteo to get the current weather for Indianapolis, IN
latitude = 39.7684
longitude = -86.1583

api = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": latitude, 
    "longitude": longitude,
    "current": ["temperature_2m", "wind_speed_10m"],
    "temperature_unit": "fahrenheit"
    }
response = requests.get(api, params=params).json()

temp = response["current"]["temperature_2m"];
wind = response["current"]["wind_speed_10m"];

print(f"Current temperature: {temp}°F")
print(f"Current wind speed: {wind} mph")
notification.notify(
    title="Current Weather",
    message=f"Temperature: {temp}°F\nWind Speed: {wind} mph",
    timeout=10
)
