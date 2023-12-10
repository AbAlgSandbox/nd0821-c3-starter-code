import pandas as pd
import os
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference

def slicing_metrics(df, feature, model, encoder, lb, cat_features=None):
    """
    Function for calculating descriptive stats on performance
    of slices of the salary inference model trained on the census dataset.
    """
    if cat_features is None:
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
    
    print(f"Determining performance metrics for unique values of {feature}")
    
    for cls in df[feature].unique():
        df_temp = df[df[feature] == cls]
        inputs, labels, _, _ = process_data(
            df_temp, categorical_features=cat_features, label="salary", training=False,
            encoder=encoder, lb=lb
        )
        preds = inference(model, inputs)
        precision, recall, fbeta = compute_model_metrics(labels, preds)
        print(f"Feature: {feature}")
        print(f"Value: {cls}")
        print(f"{feature}: {cls}, precision: {precision:.4f}")
        print(f"{feature}: {cls}, recall: {recall:.4f}")
        print(f"{feature}: {cls}, fbeta: {fbeta:.4f}")
        
    return