"""
This file contains the API for the application.
Author: Jahed Naghipoor
Date: December 2021
Pylint score: 10/10
autopep8: autopep8 --in-place --aggressive --aggressive app.py
"""
import json
import requests

PREDICT_URI = "https://udacity-mlops-nanodegree-app.herokuapp.com/predict"

sample = {'age': 42, 'workclass': 'Private', 'fnlgt': 159449, 'education': 'Bachelors', 'education-num': 13, 'marital-status': 'Married-civ-spouse', 'occupation': 'Exec-managerial', 'relationship': 'Husband', 'race': 'White', 'sex': 'Male', 'capital-gain': 5178, 'capital-loss': 0, 'hours-per-week': 40, 'native-country': 'United-States'}


response = requests.post(PREDICT_URI, json = sample)

dictionary = {
    'Request body': json.dumps(sample),
    'Status code': response.status_code,
}
print(json.dumps(dictionary, indent=4))
