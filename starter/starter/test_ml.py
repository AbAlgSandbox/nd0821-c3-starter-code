import pytest
import pandas as pd
import numpy as np
import joblib
import os

from typing import List

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

from ml.data import process_data
from ml.model import compute_model_metrics, inference

# Need to persist some sample clean raw test data and proper lb object and X and y shapes for process_data test
# Same can be used for testing inference
# Also place process_data test before inference test as inference makes use of process_data, leverage fixtures to use test output at inference test


#@pytest.fixture(scope='session')
#def trained_model():
#    # Load trained model for testing from a fixed location
#    #model = joblib.load(os.path.join(os.getcwd(), 'model', 'test_salary_prediction_model.pkl'))
#    model = joblib.load('test_salary_prediction_model.pkl')
#    return model
    

def test_process_data(ref_data: pd.DataFrame,
                    cat_features: List[str],
                    X_features: int,
                    lb_classes: str,
                    trained_encoder: OneHotEncoder,
                    trained_lb: LabelBinarizer):
    """
    Check that data is as expected in shape and meaning after processing for training or inference.
    """
    train, test = train_test_split(ref_data, test_size=0.20)
    
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=False,
        encoder=trained_encoder, lb=trained_lb
    )
    assert X_train.shape[1] == X_features
    assert len(y_train.shape) == 1
    assert repr(lb.classes_) == lb_classes
    
    X_test2, y_test2, encoder2, lb2 = process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )
    assert X_test2.shape[1] == X_features
    assert len(y_test2.shape) == 1
    assert repr(lb2.classes_) == lb_classes
    assert lb is lb2
    assert encoder is encoder2
    
    X_test3, y_test3, encoder3, lb3 = process_data(
        test, categorical_features=cat_features, label="salary", training=True
    )
    assert len(y_test3.shape) == 1
    assert lb3 is not lb
    assert encoder3 is not encoder
    
    
def test_inference(trained_model: RandomForestClassifier,
                ref_data: pd.DataFrame,
                cat_features: List[str],
                trained_encoder: OneHotEncoder,
                trained_lb: LabelBinarizer):
    """
    Check that predictions are being obtained and that they make sense.
    """
    X_test, y_test, encoder, lb = process_data(
        ref_data, categorical_features=cat_features, label="salary", training=False,
        encoder=trained_encoder, lb=trained_lb
    )
    
    func_preds = inference(trained_model, X_test)
    assert len(func_preds.shape) == 1
    assert func_preds.shape[0] == X_test.shape[0]
    
    # Check model output data type
    assert type(func_preds) == np.ndarray
    
    # Assert that prediction behavior in function is as expected
    assert (func_preds == trained_model.predict(X_test)).all()
    

def test_compute_model_metrics():
    """
    Check that statistics are being determined, that they're within valid range
    and that they make sense by using edge cases.
    """
    # Define true values
    y_label = np.array([0, 1, 1, 0])
    
    # Perfect predictions
    y_pred_perfect = np.array([0, 1, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y_label, y_pred_perfect)
    assert 0 <= precision <= 1 and 0 <= recall <= 1 and 0 <= fbeta <= 1
    assert precision == 1 and recall == 1 and fbeta == 1

    # All wrong predictions
    y_pred_none = np.array([1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_label, y_pred_none)
    assert 0 <= precision <= 1 and 0 <= recall <= 1 and 0 <= fbeta <= 1
    assert precision == 0 and recall == 0 and fbeta == 0
    
    # Regular predictions
    y_pred = np.array([1, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y_label, y_pred)
    assert 0 <= precision <= 1 and 0 <= recall <= 1 and 0 <= fbeta <= 1

