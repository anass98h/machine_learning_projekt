import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
import pandas as pd
import io
from app.main import app

client = TestClient(app)

client = TestClient(app)

@pytest.fixture
def sample_regression_data():
    """Create sample regression data."""
    df = pd.DataFrame({
        'feature1': [0.5],
        'feature2': [0.3]
    })
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

@pytest.fixture
def sample_classification_data():
    """Create sample classification data with the 38 features."""
    data = {f'feature{i}': [0.5] for i in range(1, 39)}
    df = pd.DataFrame(data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models_loaded" in data
    assert "available_models" in data

def test_list_regression_models():
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    for model in models:
        assert model["model_type"] == "regression"

def test_list_classification_models():
    response = client.get("/categorizing-models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    for model in models:
        assert model["model_type"] == "classifier"

def test_predict_endpoint(sample_regression_data):
    files = {
        'file': ('data.csv', sample_regression_data, 'text/csv')
    }
    response = client.post("/predict/model_v1.0.0", files=files)
    assert response.status_code in [200, 404]  # 404 if model not found
    if response.status_code == 200:
        data = response.json()
        assert "model_name" in data
        assert "category" in data
        assert "score" in data

def test_classify_endpoint(sample_classification_data):
    files = {
        'file': ('data.csv', sample_classification_data, 'text/csv')
    }
    response = client.post("/classify-weakest-link/classifier_v2.0.0", files=files)
    assert response.status_code in [200, 404]  # 404 if model not found
    if response.status_code == 200:
        data = response.json()
        assert "model_name" in data
        assert "weakest_link" in data