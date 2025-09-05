import requests

with open("output.txt", mode="r", encoding="utf-8") as text_file: 
    text = '''
    But hey, at least you didn’t have to worry about forgetting to charge your phone. 
    Breakfast is a humble affair—bread so stale it could double as a weapon, washed down with small beer . 
    Then it’s off to the fields, where you’ll spend *hours* bent over like a question mark, 
    praying the lord of the manor doesn’t decide today’s the day to ‘inspect’ your work . 
    But wait—it"But wait—it’s not *all* backbreaking labor. Sometimes, you get the medieval 
    equivalent of a coffee break: a quick gossip session by the communal well, where you’ll 
    hear the latest scandal—like how the blacksmith’s wife *allegedly* traded extra bread for a love potion. 
    Spoiler: it was just beet juice. By midday, you’re starving, but lunch is… more bread'''

    url = "https://edo-luc53--orpheus-tts-app-generate-speech-endpoint.modal.run"
    params = {"prompt": text, "voice": "leo"}

    response = requests.get(url, params=params)

    if response.status_code == 200:
        with open("test.wav", "wb") as f:
            f.write(response.content)
    else:
        print(f"Error: {response.status_code}")
