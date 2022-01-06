import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .ml.model import train_model, inference, compute_model_metrics


@pytest.fixture(scope='session')
def data():
    """
    data fixture

    Returns:
        np.array, np.array: features and label
    """
    X_train = np.array([[1, 2, 3.0, 4, 5, 6],
                  [7, 8, 9.0, 10, 11, 12],
                  [14, 15, 1.0, 16, 17, 18],
                  [1, 2, 5.0, 4, 5, 6],
                  [2, 3, 4.0, 5, 5, 6]])

    y_train = np.array([1, 0, 1, 1, 0])
    return X_train, y_train


@pytest.fixture(scope='session')
def prediction():
    """
    prediction fixture

    Returns:
        np.array: prediction 
    """
    return np.array([0, 1, 1, 1, 0])

def test_train_model(data):
    """
    test_train_model tests the type of train_model function

    Args:
        data (np.array, np.array): features and label arrays
    """
    
    X_train, y_train = data
    lr = RandomForestClassifier()
    model = train_model(X_train, y_train)

    assert type(lr) == type(model)
  
def test_inference(data):
    """
    test_inference tests the type of inference function as well as the length of the prediction array

    Args:
        data (np.array, np.array): features and label arrays
    """
    X_train, y_train = data
    lr = RandomForestClassifier()
    model = lr.fit(X_train, y_train)
    prediction = inference(model, X_train)

    assert isinstance(prediction, np.ndarray), \
        f'prediction of type np.ndarray expected, but got type: {type(prediction)}'
    assert len(prediction) == len(y_train), \
        f'length of predicted values do not match, expected: {len(y_train)}, instead got: {len(prediction)}'    

def test_compute_model_metrics(data, prediction):
    """
    test_compute_model_metrics tests the type of precision, recall, and f1-score

    Args:
        data (np.array, np.ndarray): features and label arrays
        prediction (np.array): prediction array
    """
    _, y = data
    precision, recall, fbeta = compute_model_metrics(y, prediction)
    assert isinstance(precision, float), f'precision of type float expected, but got type: {type(precision)}'
    assert isinstance(recall, float), 'recall of type float expected, but got type: {type(recall)}'
    assert isinstance(fbeta, float), 'fbeta of type float expected, but got type: {type(fbeta)}'