# Load libraries
import numpy as np
import pandas as pd
from typing import List, Union
from matplotlib import pyplot as plt


def keep_features(data, features: Union[List[str], List[int]]):
    # sort features highest to lowest
    if all(isinstance(x, int) for x in features):
        features = sorted(features)
        if isinstance(data, pd.DataFrame):
            return data.iloc[:, features]
        elif isinstance(data, (np.ndarray, list)):
            return [[row[i] for i in features] for row in data]
    else:
        return data[features]


def compare_features(results, names):
    # Compare Features
    plt.boxplot(results, labels=names)
    plt.title("Algorithm Comparison")
    plt.show()
