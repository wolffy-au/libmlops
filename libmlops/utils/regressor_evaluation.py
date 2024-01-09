# Load libraries
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

# from sklearn.linear_model import SGDRegressor
from matplotlib import pyplot as plt
from libmlops.features.feature_evaluation import normalise_feature_scores
from libmlops.models.model_evaluation import cross_validate_model
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
)

# Spot Check Algorithms
models = [
    # ("SGD", SGDRegressor(loss="epsilon_insensitive", max_iter=10000)),
    ("LINR", LinearRegression(n_jobs=-1)),
    ("RDG", Ridge()),
    ("LSO", Lasso()),
    ("ELN", ElasticNet()),
    ("DTR", DecisionTreeRegressor()),
    ("RFR", RandomForestRegressor(n_jobs=-1)),
    ("SVR", LinearSVR(dual="auto")),
    ("KNR", KNeighborsRegressor(n_jobs=-1)),
    ("GBR", GradientBoostingRegressor()),
]


def algorithm_evaluation(X_train, Y_train, verbose=False):
    """
    Evaluate multiple regression algorithms using cross-validation.

    Parameters:
    - X_train: The feature matrix for training.
    - Y_train: The target variable for training.
    - verbose: If True, print evaluation results for each algorithm (default is False).

    Returns:
    - results: Mean and standard deviation of cross-validated scores for each algorithm.
    - names: Names of the evaluated algorithms.
    """
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=10, random_state=1, shuffle=True)
        cv_results_mean, cv_results_std = cross_validate_model(
            model, X_train, Y_train, cv=kfold, scoring="neg_mean_absolute_error"
        )
        results.append([cv_results_mean, cv_results_std])
        names.append(name)
        if verbose:
            print("%s: %f (%f)" % (name, cv_results_mean, cv_results_std))
    return results, names


def features_evaluation(X_train, Y_train, verbose=False):
    """
    Evaluate feature importance for each regression algorithm.

    Parameters:
    - X_train: The feature matrix for training.
    - Y_train: The target variable for training.
    - verbose: If True, print feature importance results for each algorithm (default is False).

    Returns:
    - features: List of important features across all algorithms.
    """
    # evaluate each model in turn
    features = []
    for name, model in models:
        model.fit(X_train, Y_train)
        # imp_results = model.feature_importances_
        # perform permutation importance
        imp_results = permutation_importance(
            model, X_train, Y_train, scoring="neg_mean_squared_error"
        )
        imp_results_mean = normalise_feature_scores(imp_results["importances_mean"])
        # imp_results_std = normalise_feature_scores(imp_results['importances_std'])
        f = []
        for i, v in enumerate(imp_results_mean):
            if v >= 0.5:
                f.append(i)
                if i not in features:
                    features.append(i)

        if verbose:
            print(name, f)

    if verbose:
        print(features)

    return features


# Evaluation Metrics: This file typically contains functions for evaluating the performance of your trained models. These functions calculate metrics such as accuracy, precision, recall, F1 score, and others, depending on the nature of your problem (classification, regression, etc.).
def model_evaluation(model, X_test, y_test):
    """
    Evaluate the performance of a regression model on a test dataset.

    Parameters:
    - model: The trained regression model.
    - X_test: The feature matrix for testing.
    - y_test: The true target values for testing.

    Returns:
    - r2: R-squared (coefficient of determination) regression score.
    - mae: Mean Absolute Error of the model on the test dataset.
    - cv_results_mean: Mean cross-validated score.
    - cv_results_std: Standard deviation of cross-validated scores.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_results_mean, cv_results_std = cross_validate_model(
        model, X_test, y_test, scoring="neg_mean_absolute_error"
    )
    return r2, mae, cv_results_mean, cv_results_std


def compare_algorithms(results, names):
    """
    Compare the performance of different algorithms using boxplots.

    Parameters:
    - results: Mean and standard deviation of cross-validated scores for each algorithm.
    - names: Names of the evaluated algorithms.

    Displays a boxplot for visual comparison of algorithm performance.
    """
    # Compare Algorithms
    plt.boxplot(results, labels=names)
    plt.title("Algorithm Comparison")
    plt.show()
