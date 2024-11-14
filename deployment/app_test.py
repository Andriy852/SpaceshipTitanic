from tqdm import tqdm
import time
import requests
import numpy as np

passenger_info = {
    "PassengerId": "0000_01",
    "HomePlanet": "Europa",
    "CryoSleep": "True",
    "CabinNumber": 3,
    "Deck": "G",
    "Side": "S",
    "Destination": "l",
    "Age": 55,
    "VIP": "False",
    "RoomService": 0,
    "FoodCourt": 0,
    "ShoppingMall": 0,
    "Spa": 0,
    "VRDeck": 0,
    "Name": "Andrii Zhurba",
    "MultipleGroup": 1,
    "CabinCount": 21
}

url = "https://spaceship-titanic-predictor-1084830087867.europe-west1.run.app/predict"

all_times = []
for i in tqdm(range(500)):
  t0 = time.time_ns() // 1_000_000
  resp = requests.post(url, json=passenger_info)
  t1 = time.time_ns() // 1_000_000
  time_taken = t1 - t0
  all_times.append(time_taken)

print("Response time in ms:")
print("Median:", np.quantile(all_times, 0.5))
print("95th precentile:", np.quantile(all_times, 0.95))
print("Max:", np.max(all_times))