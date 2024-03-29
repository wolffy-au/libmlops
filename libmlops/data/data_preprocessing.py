# Example data_preprocessing.py
import numpy as np
import pandas as pd
import typing as List
from sklearn.model_selection import train_test_split


# Data Cleaning: Functions for cleaning the data, handling missing values, and addressing any inconsistencies in the dataset.
def handle_missing_values(data):
    # Code to fill or drop missing values
    pass


def clean_data(data):
    # Code for general data cleaning tasks
    pass


# Feature Engineering: Functions for creating new features from existing ones or transforming features to better suit the machine learning model.
def create_engineered_features(data):
    # Code for feature engineering tasks
    pass


# Data Scaling and Transformation: Preprocessing steps such as scaling numerical features or encoding categorical variables.
def scale_numerical_features(data):
    # Code for scaling numerical features
    pass


def encode_categorical_features(data):
    # Code for encoding categorical features
    pass


# Train-Test Splitting: Functions to split the data into training and testing sets, an essential step in model development.
def split_train_test(dataset, test_size=0.2, random_state=42):
    # Code for splitting the data into training and testing sets
    # Split-out validation dataset
    X, Y = get_xy(dataset)
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    return X_train, X_validation, Y_train, Y_validation


# Train-Test Splitting: Functions to split the data into training and testing sets, an essential step in model development.
def split_train_test_xy(X, Y, test_size=0.2, random_state=42):
    # Code for splitting the data into training and testing sets
    # Split-out validation dataset
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    return X_train, X_validation, Y_train, Y_validation


# Get XY: Functions to split the data into input and single output datasets
def get_xy(dataset: pd.DataFrame) -> pd.DataFrame:
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]
    return X, Y


# Get XY: Functions to split the data into input and single output datasets
# def get_xy_ndarray(dataset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     X = dataset[:, :-1]
#     Y = dataset[:, -1]
#     return X, Y


# # Get XY: Functions to split the data into input and single output datasets
# def get_xy_list(dataset: List) -> List:
#     X = [row[:-1] for row in dataset]
#     Y = [row[-1] for row in dataset]
#     return X, Y
