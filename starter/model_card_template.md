# Model Card

## Model Details

Abdiel Algorri created the model. It is a Random Forest Classifier using the default hyperparameters in scikit-learn 1.3.2. Specifically the model is designed for binary classification of salary, to infer if it is lower or equal to 50K, or over 50K.

## Intended Use

This model is intended for predictive analysis over census data. Specifically the model is designed for binary classification of salary, to infer if it is lower or equal to 50K, or over 50K. The users are anyone trying to have some idea of an individual's earning capacity in a low-stake scenario.

## Training Data

Training data is publicly available Census Bureau data facilitated via the GitHub repository of an Udacity project intended to test student acquire skills on MLOps topics.
Repository link:
https://github.com/udacity/nd0821-c3-starter-code
Original data extraction was done from the 1994 Census database. Further details about this dataset may be found at it's UC Irvine Machine Learning Repository entry:
https://archive.ics.uci.edu/dataset/20/census+income

Dataset contained 32561 rows with 15 columns, out of which one was the target column salary and of the others 8 were determined to be categorical and one-hot encoded on processing. Original dataset was processed to remove heading whitespace for all entries.

## Evaluation Data

Evaluation data was obtained from the same dataset via a train/test split of 80-20, with no stratification. Data was processed separately but through the encoder and label binarizer that were obtained from processing the training data.

## Metrics
The classes encoded by label binarizer were less than or equal to 50K and larger than 50K for salary. Obtained validation statistics were:
Precision: 0.7657450076804916
Recall: 0.6302149178255373
F-Beta Score: 0.6914008321775312

## Ethical Considerations

The level of information in this dataset is very general, the nature of the type of personal data collected and how it was collected by humans introduces the possibility of multiple type of biases that remain unexplored at this time. Also the intended use for this model was not considered during the collection of the data used to train it, the purpose of the data collection being collecting census information.

## Caveats and Recommendations

The type of model was choosen to keep training light and fast and hyperparameter optimization was not performed at all, so plenty of potential for better performance remains. Without that enphasis use of k-fold cross validation during training would have been highly beneficial for this model choice. Due to bias remaining completely unexplored and the many possibilities for the introduction of both individual and societal bias into this dataset, combined with the lack of emphasis on obtaining the best model performance, it is not recommended for this model to be used for any kind of decision making that might be impactful on an individual.

