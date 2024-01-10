# test_feature_selection.py

import numpy as np
import pandas as pd
from libmlops.features.feature_selection import (
    select_k_best_features,
    select_recursive_feature_elimination,
    convert_indices,
)


def test_select_k_best_features():
    # Create dummy data for testing
    X = np.random.rand(100, 20)  # 100 samples, 20 features
    Y = np.random.randint(0, 2, size=100)  # Binary classification labels

    # Test with default parameters
    X_selected = select_k_best_features(X, Y)

    # Check that the shape of the selected features is correct
    assert X_selected.shape == (X.shape[0], 10)  # Assuming default k=10

    # Test with custom parameters
    X_selected_custom = select_k_best_features(
        X, Y, score_func=lambda X, Y: np.ones(X.shape[1]), k=5
    )

    # Check that the shape of the selected features is correct for custom parameters
    assert X_selected_custom.shape == (X.shape[0], 5)


def test_select_recursive_feature_elimination():
    # Create dummy data for testing
    X = np.random.rand(100, 20)  # 100 samples, 20 features
    Y = np.random.randint(0, 2, size=100)  # Binary classification labels

    # Test with default parameters
    X_selected = select_recursive_feature_elimination(X, Y)

    # Check that the shape of the selected features is correct
    assert X_selected.shape == (
        X.shape[0],
        10,
    )  # Assuming default n_features_to_select=10

    # Test with custom parameters
    X_selected_custom = select_recursive_feature_elimination(
        X, Y, n_features_to_select=5
    )

    # Check that the shape of the selected features is correct for custom parameters
    assert X_selected_custom.shape == (X.shape[0], 5)


# Test convert_indices function
def test_convert_indices():
    # Mock dataset
    mock_data = {"feature_0": [1, 2, 3], "feature_1": [4, 5, 6], "feature_2": [7, 8, 9]}
    mock_dataset = pd.DataFrame(mock_data)

    feature_indices = [0, 2]
    feature_names = convert_indices(mock_dataset, feature_indices)

    # Assertions
    expected_feature_names = ["feature_0", "feature_2"]
    assert set(feature_names) == set(expected_feature_names)
    # Add more assertions based on the expected behavior of the function
