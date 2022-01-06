import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


@pytest.fixture(scope='session')
def greater_than_fifty_sample():
    sample = {
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital_gain": 15024,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    return sample


@pytest.fixture(scope='session')
def less_than_fifty_sample():
    sample = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    return sample


def test_route():
    """
    test_route tests the route /
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == ["Greetings!"]


def test_post_greater_than_fifty(greater_than_fifty_sample):
    """
    test_post_greater_than_fifty tests the route /predict for the greater than 50K sample

    Args:
        greater_than_fifty_sample (json):   The sample to be tested against the model for the greater than 50K sample
    """
    r = client.post("/predict", json=greater_than_fifty_sample)
    assert r.status_code == 200
    assert r.json() == ">50K"


def test_post_less_than_fifty(less_than_fifty_sample):
    """
    test_post_less_than_fifty tests the route /predict for the less than 50K sample

    Args:
        less_than_fifty_sample (json): The sample to be tested against the model for the less than 50K sample
    """

    response = client.post("/predict", json=less_than_fifty_sample)
    assert response.status_code == 200
    assert response.json() == "<=50K"