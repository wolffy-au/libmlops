# Load libraries
import numpy as np
import pandas as pd
from typing import List, Union
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def keep_features(data, features: Union[List[str], List[int]]):
    # sort features highest to lowest
    if all(isinstance(x, int) for x in features):
        features = sorted(features)
        if isinstance(data, pd.DataFrame):
            return data.iloc[:, features]
        elif isinstance(data, np.ndarray) or isinstance(data, list):
            return [[row[i] for i in features] for row in data]
    else:
        return data[features]


def compare_features(results, names):
    # Compare Features
    plt.boxplot(results, labels=names)
    plt.title("Algorithm Comparison")
    plt.show()
