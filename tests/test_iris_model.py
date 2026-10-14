"""
Unit tests for the Iris Classification ML models.
Tests data validation, model training, accuracy benchmarks, and prediction consistency.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


@pytest.fixture(scope="module")
def iris_data():
    """Fixture providing train/test splits for Iris dataset."""
    iris = load_iris()
    X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return {
        "raw": iris,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "species_names": list(iris.target_names)
    }


def test_dataset_integrity(iris_data):
    """Test that dataset contains 150 samples, 4 features, 3 classes and no missing values."""
    X = iris_data["X"]
    y = iris_data["y"]

    assert X.shape == (150, 4), f"Expected shape (150, 4), got {X.shape}"
    assert len(y) == 150
    assert X.isnull().sum().sum() == 0, "Dataset contains unexpected null values"
    assert len(np.unique(y)) == 3, "Expected exactly 3 target classes"


def test_decision_tree_accuracy(iris_data):
    """Test Decision Tree model meets minimum accuracy threshold (90%)."""
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(iris_data["X_train"], iris_data["y_train"])
    preds = model.predict(iris_data["X_test"])
    acc = accuracy_score(iris_data["y_test"], preds)

    assert acc >= 0.90, f"Decision Tree accuracy {acc:.2f} is below 90% threshold"


def test_knn_accuracy(iris_data):
    """Test KNN model meets minimum accuracy threshold (90%)."""
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(iris_data["X_train"], iris_data["y_train"])
    preds = model.predict(iris_data["X_test"])
    acc = accuracy_score(iris_data["y_test"], preds)

    assert acc >= 0.90, f"KNN accuracy {acc:.2f} is below 90% threshold"


@pytest.mark.parametrize("sample,expected_class_idx", [
    ([5.1, 3.5, 1.4, 0.2], 0),  # Setosa: small petals
    ([6.0, 2.7, 4.2, 1.3], 1),  # Versicolor: medium petals
    ([6.5, 3.0, 5.5, 2.0], 2),  # Virginica: large petals
])
def test_model_predictions_on_known_samples(iris_data, sample, expected_class_idx):
    """Test both models predict expected flower species on known representative inputs."""
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(iris_data["X_train"], iris_data["y_train"])

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(iris_data["X_train"], iris_data["y_train"])

    sample_df = pd.DataFrame([sample], columns=iris_data["X"].columns)
    dt_pred = dt.predict(sample_df)[0]
    knn_pred = knn.predict(sample_df)[0]

    assert dt_pred == expected_class_idx, (
        f"DT predicted {dt_pred} ({iris_data['species_names'][dt_pred]}), "
        f"expected {expected_class_idx} ({iris_data['species_names'][expected_class_idx]})"
    )
    assert knn_pred == expected_class_idx, (
        f"KNN predicted {knn_pred} ({iris_data['species_names'][knn_pred]}), "
        f"expected {expected_class_idx} ({iris_data['species_names'][expected_class_idx]})"
    )


def test_probability_distributions(iris_data):
    """Test that predicted probabilities sum to 1.0 (100%)."""
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(iris_data["X_train"], iris_data["y_train"])

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(iris_data["X_train"], iris_data["y_train"])

    sample_df = pd.DataFrame([[5.8, 2.7, 4.1, 1.0]], columns=iris_data["X"].columns)
    dt_proba = dt.predict_proba(sample_df)[0]
    knn_proba = knn.predict_proba(sample_df)[0]

    assert np.isclose(np.sum(dt_proba), 1.0), "Decision tree probabilities must sum to 1.0"
    assert np.isclose(np.sum(knn_proba), 1.0), "KNN probabilities must sum to 1.0"

