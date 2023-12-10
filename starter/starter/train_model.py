# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import os
from ml.data import process_data
from ml.model import compute_model_metrics, inference

# Add code to load in the data.
data = pd.read_csv(os.path.join('data', 'cleaned_census.csv'))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

print("Classes encoded by LabelBinarizer:", lb.classes_)

# Train and save a model.
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Test Precision: {precision}")
print(f"Test Recall: {recall}")
print(f"Test F-Beta Score: {fbeta}")

joblib.dump(model, os.path.join('model', 'salary_prediction_model.pkl'))
joblib.dump(encoder, os.path.join('model', 'encoder.pkl'))
joblib.dump(lb, os.path.join('model', 'label_binarizer.pkl'))

print("Model training completed")