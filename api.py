import requests
import json

predict_uri = "https://udacity-mlops-nanodegree-app.herokuapp.com/predict"

sample = {
    'age': 42,
    'workclass': 'Private',
    'fnlgt': 159449,
    'education': 'Bachelors',
    'education-num': 13,
    'marital-status': 'Married-civ-spouse',
    'occupation': 'Exec-managerial',
    'relationship': 'Husband',
    'race': 'White',
    'sex': 'Male',
    'capital-gain': 5178,
    'capital-loss': 0,
    'hours-per-week': 40,
    'native-country': 'United-States'}

request = requests.post(predict_uri, json=sample)


assert request.status_code == 200
assert request.json() == {"prediction": 1}

dict = {
    'Request body':sample,
    'Status code': request.status_code,
    'Response': request.json()   
}
print(json.dumps(dict, indent=4))
