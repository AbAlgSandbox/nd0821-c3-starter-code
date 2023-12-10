import requests
import os

import pandas as pd

url = "https://udacity-mlops-census-data-inference.onrender.com/"

prediction_interpretations = ['<=50K', '>50K']

def get_root():
    """
    Test the root entry which should return a greeting.
    Will do this by checking for a succesful response and
    that the response contents contain text.
    """
    
    print("Attempting to connect to root")
    response = requests.get(url)
    print("Request finished")
    
    print(f"Response code: {response.status_code}")
    print(f"Response text:\n{response.text}")

def post_inference_under_50k():
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
    print(f"Request body\n{request_body}")
    
    print("Attempting to connect to /infer")
    response = requests.post(url+"infer", json=request_body)
    print("Request finished")
    
    print(f"Response code: {response.status_code}")
    model_prediction = response.json()["prediction"][0]
    print(f"Model prediction: {str(model_prediction)}")
    print(f"Model interpretation: {prediction_interpretations[model_prediction]}")
    print(f"Passed test: {str(model_prediction == 0)}")

def post_inference_over_50k():
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
    print(f"Request body\n{request_body}")
    
    print("Attempting to connect to /infer")
    response = requests.post(url+"infer", json=request_body)
    print("Request finished")
    
    print(f"Response code: {response.status_code}")
    model_prediction = response.json()["prediction"][0]
    print(f"Model prediction: {str(model_prediction)}")
    print(f"Model interpretation: {prediction_interpretations[model_prediction]}")
    print(f"Passed test: {str(model_prediction == 1)}")

if __name__ == "__main__":
    print("Running script for testing live API")
    print(f"API location: {url}")
    print("First verifying that we get a succesful reponse root greeting")
    get_root()
    print("Next attempting to use inference model, when it should be a less than 50k result")
    post_inference_under_50k()
    print("Last attempting to use inference model, when it should be a more than 50k result")
    post_inference_over_50k()
    print("Test completed")