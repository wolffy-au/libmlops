# test_model_evaluation.py

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from libmlops.utils.classifier_evaluation import (
    algorithm_evaluation,
    features_evaluation,
    model_evaluation,
)


@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=100, n_features=20, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test


def test_algorithm_evaluation():
    # Generate some random data for testing
    X_train, _, Y_train, _ = train_test_split(
        np.random.default_rng(0).random((100, 10)),  # 100 samples, 10 features
        np.random.default_rng(0).integers(
            0, 2, size=100
        ),  # Binary classification labels
        test_size=0.2,
        random_state=42,
    )

    # Test the algorithm_evaluation function
    results, names = algorithm_evaluation(X_train, Y_train)

    # Check if the results and names have the correct structure
    assert isinstance(results, list)
    assert isinstance(names, list)

    # Check if each result is a list with two elements
    for result in results:
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


def test_features_evaluation():
    # Generate some random data for testing
    X_train, _, Y_train, _ = train_test_split(
        np.random.default_rng(0).random((100, 10)),  # 100 samples, 10 features
        np.random.default_rng(0).integers(
            0, 2, size=100
        ),  # Binary classification labels
        test_size=0.2,
        random_state=42,
    )

    # Test the features_evaluation function
    features = features_evaluation(X_train, Y_train)

    # Check if the features result is a list
    assert isinstance(features, list)

    # Check if each feature is an integer
    for feature in features:
        assert isinstance(feature, int)


def test_model_evaluation(classification_data):
    model, X_test, y_test = classification_data
    accuracy, report, cv_results_mean, cv_results_std = model_evaluation(
        model, X_test, y_test
    )
    assert isinstance(accuracy, float)
    assert isinstance(report, str)
    assert isinstance(cv_results_mean, float)
    assert isinstance(cv_results_std, float)
