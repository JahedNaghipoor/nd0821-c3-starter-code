import os
import pickle

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from .data import load_data
from .ml.data import process_data
from .ml.model import compute_model_metrics, inference, train_model



data_dir = "./starter/data/"
model_dir = "./starter/model/"
@pytest.fixture 
def root(): # root directory
    return os.getcwd()

@pytest.fixture
def files(root):
    
    load_path = os.path.join(root, data_dir, "census_cleaned.csv")
    data = load_data(load_path)
    #data = pd.read_csv("../data/census.csv")
    
    model = os.path.join(root, model_dir, "lr_model.pkl")
    with open(model, "rb") as f:
        model = pickle.load(f)

    encoder = os.path.join(root, model_dir, "encoder.pkl")
    with open(encoder, "rb") as f:
        encoder = pickle.load(f)

    lb = os.path.join(root, model_dir, "lb.pkl")
    with open(lb, "rb") as f:
        lb = pickle.load(f)

    return data, model, encoder, lb

@pytest.fixture
def train_test_data(files):
    data, model, encoder, lb = files
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    return train, test

def test_train_model(files, root):
    data, model, encoder, lb = files
    train, test = train_test_split(data, test_size=0.20)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    # X_train, y_train, encoder, lb = process_data(
    #     train, categorical_features=cat_features, label="salary", training=True
    # )
    filepath = os.path.join(root, "../model/lr_model.pkl")

    assert os.path.exists(filepath)
