import pytest
import pandas as pd
import numpy as np
import joblib
import os


def pytest_addoption(parser):
    parser.addoption("--modelpath", action="store")
    parser.addoption("--testdata", action="store")
    

@pytest.fixture(scope='session')
def trained_model(request):
    """
    Load trained model for testing from passed or from a fixed location.
    """
    model_path = request.config.getoption("--modelpath")
    if model_path is None:
        model_path = os.path.join(os.getcwd(), 'model', 'salary_prediction_model.pkl')
        
    model = joblib.load(model_path)
    
    return model
    
    
@pytest.fixture(scope='session')
def trained_encoder(request):
    """
    Load trained model encoder for data processing.
    """
    model_path = request.config.getoption("--modelpath")
    if model_path is None:
        encoder_path = os.path.join(os.getcwd(), 'model', 'encoder.pkl')
    else:
        directory_path = os.path.dirname(model_path)
        encoder_path = os.path.join(directory_path, 'encoder.pkl')
        
    encoder = joblib.load(encoder_path)
    
    return encoder
    
    
@pytest.fixture(scope='session')
def trained_lb(request):
    """
    Load trained model label binarizer for data processing.
    """
    model_path = request.config.getoption("--modelpath")
    if model_path is None:
        lb_path = os.path.join(os.getcwd(), 'model', 'label_binarizer.pkl')
    else:
        directory_path = os.path.dirname(model_path)
        lb_path = os.path.join(directory_path, 'label_binarizer.pkl')
        
    lb = joblib.load(lb_path)
    
    return lb


@pytest.fixture(scope='session')
def ref_data(request):
    """
    Load reference data either from a given path or from a default test data file.
    """
    data_path = request.config.getoption("--testdata")

    if data_path is None:
        data_path = os.path.join(os.getcwd(), 'data', 'test_data.csv')

    df = pd.read_csv(data_path)

    return df


@pytest.fixture(scope='session')
def cat_features():
    """
    Categorical features list.
    """
    features_list = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    
    return features_list
    
    
@pytest.fixture(scope='session')
def X_features():
    """
    Size of dimension 1 of model input after data processing.
    """
    X_dim1_size = 108
    
    return X_dim1_size


@pytest.fixture(scope='session')
def lb_classes():
    """
    String representation of classes encoded by LabelBinarizer.
    """
    lb_classes_repr = "array(['<=50K', '>50K'], dtype='<U5')"
    
    return lb_classes_repr