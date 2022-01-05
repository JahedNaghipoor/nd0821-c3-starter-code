"""
This code is used to test the API.
"""
import os
import pickle
from fastapi import FastAPI
from pydantic import BaseModel  # pydantic is a python data validation library
import pandas as pd
import numpy as np

from starter.ml.data import process_data
from starter.ml.model import inference, load_model

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")
    
columns = 'age,workclass,fnlgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country'.replace('-', '_').split(',')

categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/lr_model.pkl')
encoder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/encoder.pkl')

with open(model_path, 'rb') as file:
    model = pickle.load(file)
with open(encoder_path, 'rb') as file:
    encoder = pickle.load(file)
    
app = FastAPI()  # initialize the application


class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


@app.get("/")  # greetings route
def greetings():
    """
    greetings route

    Returns:
        string: a greetings message
    """
    return "Greetings!!!"

@app.post("/predict")
def predict_salary_level(item: Data):
    item_dict = item.dict()
    X = pd.DataFrame([[item_dict[column] for column in columns]], columns=columns)

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    X_categorical = encoder.transform(X_categorical)

    X = np.concatenate([X_continuous, X_categorical], axis=1)


    pred = int(model.predict(X)[0])

    return {"prediction": pred}