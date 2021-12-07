from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import pickle


RANDOM_STATE = 42

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, path):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    gbc= GradientBoostingClassifier(random_state=RANDOM_STATE)  # Gradient Boosting Classifier
    parameters = {"n_estimators": (5, 10),
                  "learning_rate": (0.1, 0.01, 0.001),
                  "max_depth": [2, 3, 4],
                  "max_features": ("auto", "log2")
                  } # Hyperparameters
    clf = GridSearchCV(gbc, parameters) # GridSearchCV
    clf.fit(X_train, y_train)   # Fit the model
    with open(path, 'wb') as file:
        pickle.dump(clf.best_estimator_, file)  # Save the model
    model = clf.best_estimator_ 
    return model    # Return the best model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta # Return the metrics for the model.


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn model
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X) # Return the predictions
