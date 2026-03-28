import requests

api = "https://us.api.blizzard.com"

def get_character_achievements(realm, character_name):
    url = f"{api}/profile/wow/character/{realm}/{character_name}/achievements"
    headers = {
        "Authorization": "Bearer USfGIL6eQZlkUq8f7H8JKhthnzftu38s21"
    }
    params = {
        "namespace": "profile-us",
        "locale": "en_US",
        "character_name": character_name,
        "realm": realm,
        "region": "us"
    }
    response = requests.get(url, headers=headers, params=params).json()
    return response

print(get_character_achievements("terenas", "twopoc").get("total_points"))