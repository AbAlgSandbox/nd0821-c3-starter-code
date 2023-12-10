import os
import argparse
import pandas as pd
import joblib

from slicing import slicing_metrics


def go(args):

    feature = args.feature
    
    if feature is None or not feature:
        print("No valid value for feature was provided.")
        return
    
    data_path = args.data_path
    
    if data_path is None or not data_path:
        data_path = os.path.join(os.getcwd(), 'data', 'cleaned_census.csv')
    
    df = pd.read_csv(data_path)
    
    if feature not in df.columns:
        print(f"Data does not contain feature {feature}.")
        return
    
    model_path = args.model_path
    
    if model_path is None or not model_path:
        model_path = os.path.join(os.getcwd(), 'model', 'salary_prediction_model.pkl')
    
    directory_path = os.path.dirname(model_path)
    encoder_path = os.path.join(directory_path, 'encoder.pkl')
    lb_path = os.path.join(directory_path, 'label_binarizer.pkl')
    
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    lb = joblib.load(lb_path)
    
    slicing_metrics(df, feature, model, encoder, lb)
    
    return

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Cycle feature to obtain performance statistics")


    parser.add_argument(
        "--feature", 
        type=str,
        help="Feature through which unique values to cycle",
        required=True
    )

    parser.add_argument(
        "--model_path", 
        type=str,
        help="Path to where model, encoder and binarizer are located, including model filename",
        required=False
    )
    
    parser.add_argument(
        "--data_path", 
        type=str,
        help="Path to a valid census data dataset in csv format",
        required=False
    )




    args = parser.parse_args()

    go(args)