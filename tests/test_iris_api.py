"""
Integration and API tests for the Iris Classification web application.
Tests HTTP endpoints, payload handling, prediction responses, and dataset inspection.
"""
import importlib

import pytest


@pytest.fixture(scope="module")
def client():
    """Create a Flask test client for the Iris web app."""
    app_module = importlib.import_module("01_iris_project.app")
    flask_app = app_module.app
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as test_client:
        yield test_client


def test_home_page_renders(client):
    """Test that the homepage loads successfully with HTML content."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Iris Flower Classifier" in response.data or b"Iris" in response.data


def test_predict_endpoint_success(client):
    """Test /predict with valid measurements for Setosa."""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.get_json()
    assert "dt" in data
    assert "knn" in data
    assert "species_info" in data

    # Decision tree checks
    assert data["dt"]["prediction"] == "setosa"
    assert "probabilities" in data["dt"]
    assert data["dt"]["probabilities"]["setosa"] > 80.0

    # KNN checks
    assert data["knn"]["prediction"] == "setosa"
    assert "probabilities" in data["knn"]
    assert data["knn"]["probabilities"]["setosa"] > 80.0


def test_predict_versicolor_sample(client):
    """Test /predict with medium measurements returns versicolor."""
    payload = {
        "sepal_length": 6.0,
        "sepal_width": 2.7,
        "petal_length": 4.5,
        "petal_width": 1.5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["dt"]["prediction"] in ["versicolor", "virginica"]
    assert data["knn"]["prediction"] in ["versicolor", "virginica"]


def test_dataset_endpoint(client):
    """Test /dataset returns complete dataset of 150 records."""
    response = client.get("/dataset")
    assert response.status_code == 200

    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) == 150

    first_item = data[0]
    expected_keys = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    assert expected_keys.issubset(first_item.keys())
