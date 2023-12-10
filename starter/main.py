# Put the code for your API here.

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import pandas as pd
import os
import joblib

from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference


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


class InputData(BaseModel):
    age: int = Field(..., example=26)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=122999)
    education: str = Field(..., example="Masters")
    education_num: int = Field(..., alias="education-num", example=14)
    marital_status: str = Field(..., alias="marital-status", example="Never-married")
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=8614)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")
    
    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
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
        }


app = FastAPI()


model = None
encoder = None
binarizer = None


@app.on_event("startup")
async def load_artifacts():
    global model, encoder, binarizer
    model = joblib.load(os.path.join(os.getcwd(), 'model', 'salary_prediction_model.pkl'))
    encoder = joblib.load(os.path.join(os.getcwd(), 'model', 'encoder.pkl'))
    binarizer = joblib.load(os.path.join(os.getcwd(), 'model', 'label_binarizer.pkl'))

@app.get("/")
async def main_greeting():
    return HTMLResponse('<div><p>Hello and welcome to the census information \
                            based earning threshold inference model.</p> \
                            <p>Make a POST request to the inference endpoint for predictions.</p> \
                            <p>API methods documentation at <a href="/docs">/docs</a>.</p></div>')

@app.post("/infer")
async def predict(data: InputData):
    try:
        input_df = pd.DataFrame([data.dict(by_alias=True)])
        
        processed_data, _, _, _ = process_data(
            input_df, categorical_features=cat_features, label=None, training=False,
            encoder=encoder, lb=binarizer
        )
        
        prediction = inference(model, processed_data)
        
        return {"prediction": prediction.tolist()}
    
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))