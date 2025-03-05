import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
import joblib
import numpy as np
from app.model_loader import ModelLoader

# Define proper classes instead of lambda functions
class SimpleRegressionModel:
    def predict(self, X):
        return np.array([0.75])  # Returns a score between 0 and 1

class SimpleClassifierModel:
    def predict(self, X):
        return np.array(["ForwardHead"])

@pytest.fixture
def temp_model_dirs(tmp_path):
    """Create temporary directories for test models."""
    regression_dir = tmp_path / "regression_models"
    classifier_dir = tmp_path / "classifier_models"
    regression_dir.mkdir()
    classifier_dir.mkdir()
    return regression_dir, classifier_dir

@pytest.fixture
def model_loader(temp_model_dirs):
    """Create ModelLoader instance with temp directories."""
    regression_dir, classifier_dir = temp_model_dirs
    return ModelLoader(
        regression_dir=str(regression_dir),
        classifier_dir=str(classifier_dir)
    )

@pytest.fixture
def sample_models(temp_model_dirs):
    """Create sample models for testing."""
    regression_dir, classifier_dir = temp_model_dirs
    
    # Create proper model instances
    reg_model = SimpleRegressionModel()
    clf_model = SimpleClassifierModel()
    
    # Save models
    joblib.dump((reg_model, "1.0.0"), regression_dir / "model_v1.0.0.pkl")
    joblib.dump((clf_model, "2.0.0"), classifier_dir / "classifier_v2.0.0.pkl")
    
    return {"regression": reg_model, "classifier": clf_model}

def test_directories_created(model_loader):
    """Test if directories are created on initialization."""
    assert model_loader.regression_dir.exists()
    assert model_loader.classifier_dir.exists()

def test_load_models(model_loader, sample_models):
    """Test loading models from directories."""
    model_loader.load_models()
    assert len(model_loader.models) == 2
    assert "model_v1.0.0" in model_loader.models
    assert "classifier_v2.0.0" in model_loader.models

def test_model_type_filtering(model_loader, sample_models):
    """Test filtering models by type."""
    model_loader.load_models()
    
    reg_models = model_loader.list_models(model_type="regression")
    clf_models = model_loader.list_models(model_type="classifier")
    
    assert len(reg_models) == 1
    assert len(clf_models) == 1
    assert reg_models[0].model_type == "regression"
    assert clf_models[0].model_type == "classifier"

def test_model_prediction(model_loader, sample_models):
    """Test model predictions."""
    model_loader.load_models()
    
    # Test regression model
    reg_model = model_loader.get_model("model_v1.0.0")
    assert reg_model is not None
    pred = reg_model.predict(np.array([[1, 2, 3]]))
    assert isinstance(pred[0], (float, np.float64))
    assert 0 <= pred[0] <= 1
    
    # Test classifier model
    clf_model = model_loader.get_model("classifier_v2.0.0")
    assert clf_model is not None
    pred = clf_model.predict(np.array([[1, 2, 3]]))
    assert pred[0] in ["ForwardHead"]  # Add other possible classes if needed