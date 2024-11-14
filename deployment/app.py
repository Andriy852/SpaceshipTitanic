from pydantic import BaseModel
from fastapi import FastAPI
import joblib
import pandas as pd
from typing import Optional
import numpy as np
import helper_functions


class Passenger(BaseModel):
    PassengerId: str
    HomePlanet: Optional[str]
    CryoSleep: Optional[str]
    CabinNumber: Optional[float]
    Deck: Optional[str]
    Side: Optional[str]
    Destination: Optional[str]
    Age: Optional[float]
    VIP: Optional[str]
    RoomService: Optional[float]
    FoodCourt: Optional[float]
    ShoppingMall: Optional[float]
    Spa: Optional[float]
    VRDeck: Optional[float]
    Name: Optional[str]
    MultipleGroup: Optional[int]
    CabinCount: Optional[int]

class Prediction(BaseModel):
    probability: float

loaded_pipeline = joblib.load('pipeline_with_catboost.pkl')
app = FastAPI()

@app.post("/predict", response_model=Prediction)
def predict(passenger: Passenger):
    dataframe = pd.DataFrame([passenger.dict()])
    dataframe.replace({None: np.nan}, inplace=True)
    dataframe = helper_functions.create_new_features(dataframe)
    preds = loaded_pipeline.predict_proba(dataframe)
    result = {"probability": preds[0, 1]}
    return result


