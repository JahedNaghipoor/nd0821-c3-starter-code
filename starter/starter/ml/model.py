from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import pickle


RANDOM_STATE = 42
# save the model
def save_model(model, file):
    """
    save_model: saves a pickled model to a file.

    Args:
        model: The model to save
        file: The file to save the model to.
    """
    with open(file, "wb") as f:
        pickle.dump(model, f)

# load the model
def load_model(file):
    """
    load_model: loads a pickled model from a file.

    Args:
        file: The file to load the model from.

    Returns:
        model: logistic regression model
    """
    with open(file, "rb") as f:
        model = pickle.load(f)
    return model

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
    return LogisticRegression().fit(X_train,y_train)


def compute_model_metrics(y, predictions):
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
    fbeta = fbeta_score(y, predictions, beta=1, zero_division=1)
    precision = precision_score(y, predictions, zero_division=1)
    recall = recall_score(y, predictions, zero_division=1)
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

def evaluate_model_on_column_slices(df, column, y, predictions):
    """
    Validates the trained machine learning model on column slices
    using precision, recall, and F1.
    Inputs
    ------
    df: pd.DataFrame
        Test dataset used for creating predictions
    column: str
        Column name to create slices on
    y : np.array
        Known labels, binarized.
    predictions : np.array
        Predicted labels, binarized.
    Returns
    -------
    predictions : list
        Prediction on column slices
        (Precision, recall, fbeta)
    """
    slices = df[column].unique()
    slice_metrics = []
    for slice in slices:
        fill = df[column] == slice
        slice_metrics.append(
            (slice,) + compute_model_metrics(y[fill], predictions[fill])
        )
    return slice_metrics