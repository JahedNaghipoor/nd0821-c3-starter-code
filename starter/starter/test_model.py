import numpy as np
from sklearn.linear_model import LogisticRegression

from .ml.model import train_model

def test_train_model():
    
    rf = train_model(np.array([[1.0, 2.0], [2.0, 3.0]]), np.array([[1], [0]]))

    assert type(rf) == LogisticRegression


def test_inference_output_shape():
    model = LogisticRegression()
    model.fit(np.array([[1.0, 2.0], [2.0, 3.0]]), np.array([[1], [0]]))

    assert len(model.predict(np.arange(4).reshape(2, 2))) == 2