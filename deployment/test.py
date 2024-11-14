import requests

# list of object type features
obj_features = ["HomePlanet", "CryoSleep",
                "Destination", "VIP", "Deck", "Side"]

# list of numeric features
num_features = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa",
                "VRDeck", "CabinCount", "CabinNumber", "MultipleGroup"]

passenger = {}

# possible values for object type features
feature_values = {"HomePlanet": ['Europa', 'Earth', 'Mars'],
                  "CryoSleep": ['True', 'False'],
                  "Destination": ['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'],
                  "VIP": ['True', 'False'],
                  "Deck": ['B', 'F', 'A', 'G', 'E', 'D', 'C', 'T'],
                  "Side": ['P', 'S']}

# Passenger_id
passenger["PassengerId"] = input("Enter the ID of the passenger in the following format:\n"
                                 "(GroupNumber)_(ID): ")

# last_name
passenger["Name"] = input("Enter first and last name of the passenger: ")

# add object features
for feature in obj_features:
    passenger[feature] = input(f"Enter the value of {feature}({'/'.join(feature_values[feature])}): ")
    if passenger[feature] not in feature_values[feature]:
        passenger[feature] = None

# add numeric features
for feature in num_features:
    try:
        if feature in ["CabinCount", "MultipleGroup"]:
            value = int(input(f"Enter the value of {feature}: "))
        else:
            value = float(input(f"Enter the value of {feature}: "))
        passenger[feature] = value
    except ValueError:
        passenger[feature] = None

url = "https://spaceship-titanic-predictor-1084830087867.europe-west1.run.app/predict"
resp = requests.post(url, json=passenger)

print(resp.json())