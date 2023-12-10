import requests
import os

import pandas as pd


def test_get_root():
    """
    Test the root entry which should return a greeting.
    Will do this by checking for a succesful response and
    that the response contents contain text.
    """
    
    response = requests.get("http://127.0.0.1:8000/")
    
    assert response.status_code == 200
    assert len(response.text) > 0

def test_post_inference_under_50k():
    """
    Test for the inference method.
    This test will use a sample that should be under the 50k threshhold,
    which is category 0 in the response.
    """
    
    request_body = {
        "age": 20,
        "workclass": "Private",
        "fnlgt": 104164,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    
    response = requests.post("http://127.0.0.1:8000/infer", json=request_body)
    
    assert response.status_code == 200
    assert response.json()["prediction"][0] == 0

def test_post_inference_over_50k():
    """
    Test for the inference method.
    This test will use a sample that should be over the 50k threshhold,
    which is category 1 in the response.
    """
    
    request_body = {
        "age": 26,
        "workclass": "Private",
        "fnlgt": 122999,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 8614,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    
    response = requests.post("http://127.0.0.1:8000/infer", json=request_body)
    
    assert response.status_code == 200
    assert response.json()["prediction"][0] == 1