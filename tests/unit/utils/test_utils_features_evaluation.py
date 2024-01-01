import numpy as np
import pandas as pd

from libmlops.utils.features_evaluation import keep_features


# Test cases using pytest
def test_keep_features_dataframe():
    # Create a pandas DataFrame
    df = pd.DataFrame.from_dict(
        {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9], "D": [10, 11, 12]}
    )

    # Test keeping specific features
    result_df = keep_features(df, features=["B", "A"])
    expected_result = pd.DataFrame({"B": [4, 5, 6], "A": [1, 2, 3]})
    pd.testing.assert_frame_equal(result_df, expected_result)

    # Test keeping specific features with column indices
    result_df = keep_features(df, features=[2, 0])
    expected_result = pd.DataFrame({"A": [1, 2, 3], "C": [7, 8, 9]})
    pd.testing.assert_frame_equal(result_df, expected_result)

    # Test keeping all features
    result_df = keep_features(df, features=list(df.columns))
    pd.testing.assert_frame_equal(result_df, df)


def test_keep_features_numpy():
    # Create a numpy array
    np_array = np.array([[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]])

    # Test keeping features and target column
    result_np = keep_features(np_array, features=[3, 0, 1])
    expected_result = np.array([[1, 4, 10], [2, 5, 11], [3, 6, 12]])
    assert np.array_equal(result_np, expected_result)


def test_keep_features_list():
    # Create a list of lists
    list_of_lists = [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]

    # Test keeping features and target column
    result_list = keep_features(list_of_lists, features=[2, 3, 0, 1])
    expected_result = [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]
    assert result_list == expected_result


# Run the tests
if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
