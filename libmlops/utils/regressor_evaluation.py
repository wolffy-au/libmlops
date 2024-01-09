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


def compare_algorithms(results, names):
    # Compare Algorithms
    plt.boxplot(results, labels=names)
    plt.title("Algorithm Comparison")
    plt.show()
