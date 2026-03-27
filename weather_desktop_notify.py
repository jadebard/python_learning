import requests 
from plyer import notification

city = "Indianapolis"
api = "https://api.open-meteo.com/v1/search"
params = {"name": city, "count": 1}
response = requests.get(api, params=params).json()
print(response)