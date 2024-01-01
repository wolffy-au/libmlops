import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from libmlops.data.data_preprocessing import (
    split_train_test,
    split_train_test_xy,
    get_xy,
)


# Mocking the train_test_split function
@patch("libmlops.data.data_preprocessing.train_test_split")
def test_split_train_test(mock_train_test_split):
    # Mock data for testing
    mock_dataset = MagicMock()
    mock_dataset.values = np.array(
        [[1, 2, 0], [3, 4, 1], [5, 6, 0], [7, 8, 1], [9, 10, 0]]
    )

    # Mock the train_test_split function to return predefined values
    mock_train_test_split.return_value = (
        [[1, 2], [3, 4], [5, 6]],  # X_train
        [[7, 8], [9, 10]],  # X_validation
        [0, 1, 0],  # Y_train
        [1, 0],  # Y_validation
    )

    # Call the function under test
    X_train, X_validation, Y_train, Y_validation = split_train_test(mock_dataset)

    # Assertions
    assert X_train == [[1, 2], [3, 4], [5, 6]]
    assert X_validation == [[7, 8], [9, 10]]
    assert Y_train == [0, 1, 0]
    assert Y_validation == [1, 0]

    # Ensure train_test_split was called with the correct arguments
    # mock_train_test_split.assert_called_once_with(
    #     mock_dataset.values[:, :-1],  # X
    #     mock_dataset.values[:, -1],  # y
    #     test_size=0.2,  # test_size
    #     random_state=42,  # random_state
    # )


# Mocking the train_test_split function
@patch("libmlops.data.data_preprocessing.train_test_split")
def test_split_train_test_xy(mock_train_test_split):
    # Mock data for testing
    mock_dataset = MagicMock()
    mock_dataset.return_value = (
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],  # X
        [0, 1, 0, 1, 0],  # Y
    )

    # Mock the train_test_split function to return predefined values
    mock_train_test_split.return_value = (
        [[1, 2], [3, 4], [5, 6]],  # X_train
        [[7, 8], [9, 10]],  # X_validation
        [0, 1, 0],  # Y_train
        [1, 0],  # Y_validation
    )

    X = mock_dataset.values[:, :-1]
    Y = mock_dataset.values[:, -1]
    # Call the function under test
    X_train, X_validation, Y_train, Y_validation = split_train_test_xy(X, Y)

    # Assertions
    assert X_train == [[1, 2], [3, 4], [5, 6]]
    assert X_validation == [[7, 8], [9, 10]]
    assert Y_train == [0, 1, 0]
    assert Y_validation == [1, 0]

    # Ensure train_test_split was called with the correct arguments
    mock_train_test_split.assert_called_once_with(
        mock_dataset.values[:, :-1],  # X
        mock_dataset.values[:, -1],  # y
        test_size=0.2,  # test_size
        random_state=42,  # random_state
    )


# Test the get_xy function for pandas DataFrames
def test_get_xy_pandas():
    data = {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [7, 8, 9]}
    mock_dataset = pd.DataFrame(data)

    X, Y = get_xy(mock_dataset)

    # Check if X and Y have the correct types
    assert isinstance(X, pd.DataFrame)
    assert isinstance(Y, pd.Series)

    # Check if the values are correct
    pd.testing.assert_frame_equal(
        X, pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    )
    pd.testing.assert_series_equal(Y, pd.Series([7, 8, 9], name="target"))


# Test the get_xy function for numpy arrays
# def test_get_xy_numpy():
#     data = {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [7, 8, 9]}
#     mock_dataset = np.array([data["feature1"], data["feature2"], data["target"]]).T

#     X, Y = get_xy_ndarray(mock_dataset)

#     # Check if X and Y have the correct shapes
#     assert isinstance(X, np.ndarray)
#     assert isinstance(Y, np.ndarray)
#     assert X.shape == (3, 2)  # Assuming 2 features
#     assert Y.shape == (3,)

#     # Check if the values are correct
#     np.testing.assert_array_equal(X, np.array([[1, 2, 3], [4, 5, 6]]))
#     np.testing.assert_array_equal(Y, np.array([7, 8, 9]))


# Test the get_xy function for lists
# def test_get_xy_list():
#     mock_dataset = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

#     X, Y = get_xy_list(mock_dataset)

#     # Check if X and Y have the correct shapes
#     assert isinstance(X, list)
#     assert isinstance(Y, list)
#     assert len(X[0]) == 2  # Assuming 2 features
#     assert len(Y) == 3

#     # Check if the values are correct
#     assert X == [[1, 4], [2, 5], [3, 6]]
#     assert Y == [7, 8, 9]
