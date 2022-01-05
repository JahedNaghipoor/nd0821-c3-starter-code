"""
This code is used to test the API.
"""
import os
from fastapi import FastAPI
from pydantic import BaseModel  # pydantic is a python data validation library
import pandas as pd

from starter.ml.data import process_data
from starter.ml.model import inference, load_model

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

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


@app.post("/inference/")  # greetings route
def inference_route(data: Data):
    """
    inference_route [summary]

    [extended_summary]

    Args:
        data (Data):  base model for data

    Returns:
        float: prediction
    """
    dictionary = {
        "age": data.age,
        "workclass": data.workclass,
        "fnlgt": data.fnlgt,
        "education": data.education,
        "education-num": data.education_num,
        "marital-status": data.marital_status,
        "occupation": data.occupation,
        "relationship": data.relationship,
        "race": data.race,
        "sex": data.sex,
        "capital-gain": data.capital_gain,
        "capital-loss": data.capital_loss,
        "hours-per-week": data.hours_per_week,
        "native-country": data.native_country
    }

    data = pd.DataFrame.from_dict(dictionary)

    encoder = load_model('encoder.pkl')
    model = load_model('model.pkl')
    label = load_model('lb.pkl')

    X, _, _, _ = process_data(
        data, categorical_features=categorical_features, training=False, encoder=encoder)
    predictions = inference(model, X.reshape(1, 108))
    predictions = label.inverse_transform(predictions)
    return predictions[0]
