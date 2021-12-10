import os
import pickle

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from data import *
from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model


data_dir = "../data/"
model_dir = "../model/"
@pytest.fixture 
def root(): # root directory
    return os.getcwd()

@pytest.fixture
def files(root):
    
    load_path = os.path.join(root, data_dir, "census.csv")
    data = load_data(load_path)
    
    model = os.path.join(root, model_dir, "gbclassifier.pkl")
    with open(model, "rb") as f:
        model = pickle.load(f)

    encoder = os.path.join(root,model_dir, "encoder.pkl")
    with open(encoder, "rb") as f:
        encoder = pickle.load(f)

    lb = os.path.join(root, model_dir, "lb.pkl")
    with open(lb, "rb") as f:
        lb = pickle.load(f)

    return data, model, encoder, lb


@pytest.fixture
def train_test_data(files):
    data, _, _, _ = files
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    return train, test

def test_train_model(files, root):
    data, model, encoder, lb = files
    train, test = train_test_split(data, test_size=0.20, random_state=42)
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
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    filepath = os.path.join(root, model_dir, "gbclassifier_test.pkl")
    model = train_model(X_train, y_train, filepath=filepath)

    assert os.path.exists(filepath)
    return X_train, y_train, model, encoder, lb