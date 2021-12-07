# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from data import load_data
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import pickle
import os

# Add code to load in the data.
data_dir = "../data/"   # the directory where the data is stored.
data_path = os.path.join(data_dir + "census_clean.csv") # path to the clean data
data = load_data(data_path)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)    # 20% of the data will be used for testing

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

pickle.dump(lb, open('../model/lb.pkl', "wb"))  # save the label encoder
pickle.dump(encoder, open('../model/encoder.pkl', 'wb'))    # save the encoder


# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)    # load the encoder and label encoder

# Train and save a model.
model_dir = "../model"  # the directory where the model will be stored.
model_path = os.path.join(model_dir + "/gbc_model.pkl") # path to the model

classifier = train_model(X_train, y_train, model_path)  # train the model
y_train_predict = inference(classifier, X_train)    # make predictions on the training data
train_precision, train_recall, train_fbeta = compute_model_metrics(y_train, y_train_predict)    # compute the metrics
print(f"train_precision: {train_precision}, train_recall: {train_recall}, train_fbeta: {train_fbeta}")  # print the metrics

y_test_predict = inference(classifier, X_test)  # make predictions on the test data
test_precision, test_recall, test_fbeta = compute_model_metrics(y_test, y_test_predict) # compute the metrics
print(f"test_precision: {test_precision}, test_recall: {test_recall}, test_fbeta: {test_fbeta}")    # print the metrics