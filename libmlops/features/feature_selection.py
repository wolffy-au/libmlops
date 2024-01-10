# Example feature_selection.py

from typing import List
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    RFE,
    SelectFromModel,
    VarianceThreshold,
    SequentialFeatureSelector,
)
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor


# Feature Selection Functions: This file typically contains functions for selecting a subset of relevant features. This can help improve model performance and reduce overfitting.
def select_k_best_features(X, Y, score_func=f_classif, k=10, verbose=False):
    """
    Selects the top k best features using a given scoring function.

    Parameters:
    - X (array-like): Feature matrix.
    - Y (array-like): Target variable.
    - score_func (callable, optional): Scoring function for feature selection. Default is f_classif.
    - k (int, optional): Number of top features to select. Default is 10.
    - verbose (bool, optional): If True, print additional information. Default is False.

    Returns:
    - X_selected (array-like): Feature matrix with selected features.
    """
    selector = SelectKBest(score_func=score_func, k=k)
    X_selected = selector.fit_transform(X, Y)
    if verbose:
        # summarize scores
        print("X_selected:", X_selected)

    return X_selected


# Recursive Feature Elimination (RFE): Functions for recursively removing the least important features based on model performance.
def select_recursive_feature_elimination(X, Y, n_features_to_select=10, verbose=False):
    """
    Performs Recursive Feature Elimination (RFE) to recursively remove the least important features.

    Parameters:
    - X (array-like): Feature matrix.
    - Y (array-like): Target variable.
    - n_features_to_select (int, optional): Number of features to select. Default is 10.
    - verbose (bool, optional): If True, print additional information. Default is False.

    Returns:
    - X_selected (array-like): Feature matrix with selected features.
    """
    selector = RFE(
        LogisticRegression(solver="lbfgs", max_iter=1000),
        n_features_to_select=n_features_to_select,
    )
    X_selected = selector.fit_transform(X, Y)
    if verbose:
        # summarize scores
        print("X_selected:", X_selected)

    return X_selected


# Convert indices  to feature names
def convert_indices(dataset, features: List[int]):
    """
    Converts feature indices to corresponding feature names.

    Parameters:
    - dataset (DataFrame): The dataset with named columns.
    - features (List[int]): List of feature indices.

    Returns:
    - feature_names (List[str]): List of corresponding feature names.
    """
    return [dataset.columns[v] for v in sorted(features)]
