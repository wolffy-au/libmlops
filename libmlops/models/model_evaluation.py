# Example model_evaluation.py

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score


# Cross-Validation: If you use cross-validation to assess model performance, you might include functions for performing cross-validation and summarizing the results.
def cross_validate_model(model, X, y, cv=5, scoring="accuracy"):
    """
    Perform cross-validation on a given model.

    Parameters:
    - model: The machine learning model to be evaluated.
    - X: The feature matrix.
    - y: The target variable.
    - cv: Number of cross-validation folds (default is 5).
    - scoring: The evaluation metric used (default is "accuracy").

    Returns:
    - mean_score: Mean cross-validated score.
    - std_score: Standard deviation of cross-validated scores.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=4)
    return scores.mean(), scores.std()


# Performance Functions: Functions that help you understand model performance, such as confusion matrices, ROC curves, and precision-recall curves.
def confusion_matrix_model(model, X_test, y_test):
    """
    Generate a confusion matrix for a given model.

    Parameters:
    - model: The machine learning model.
    - X_test: The feature matrix for testing.
    - y_test: The true labels for testing.

    Returns:
    - confusion_matrix: A confusion matrix showing the performance of the model.
    """
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred, normalize="true")


# Visualization Functions: Functions for creating visualizations that help you understand model performance, such as confusion matrices, ROC curves, and precision-recall curves.
def plot_confusion_matrix(confusion_matrix, model):
    """
    Plot a confusion matrix for visualization.

    Parameters:
    - confusion_matrix: The confusion matrix to be visualized.
    - model: The machine learning model.

    Displays a visual representation of the confusion matrix.
    """
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=model.classes_
    )
    disp.plot(cmap=plt.cm.Blues)

    plt.title("Confusion Matrix")
    plt.show()
