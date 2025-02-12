from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Tuple
from pathlib import Path
import logging
import joblib

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',  'ML')))



@dataclass
class ModelInfo:
    name: str
    version: str
    path: str
    model_type: str  # "regression" or "classifier"

class ModelLoader:
    def __init__(
        self, 
        regression_dir: str = "../ML/saved models",
        classifier_dir: str = "../ML/classifier_models"
    ):
        self.regression_dir = Path(regression_dir)
        self.classifier_dir = Path(classifier_dir)
        self.models: Dict[str, Any] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        
        # Create directories if they don't exist
        self.regression_dir.mkdir(exist_ok=True)
        self.classifier_dir.mkdir(exist_ok=True)

    def load_models(self) -> None:
        """Load all models from both directories."""
        try:
            # Clear existing models
            self.models = {}
            self.model_info = {}
            
            # Load regression models
            self._load_models_from_dir(self.regression_dir, "regression")
            
            # Load classifier models
            self._load_models_from_dir(self.classifier_dir, "classifier")
            
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            raise

    def _load_models_from_dir(self, directory: Path, model_type: str) -> None:
        """Load models from a specific directory."""
        model_files = list(directory.glob("*.pkl"))
        if not model_files:
            logging.warning(f"No model files found in {directory}")
            return

        for model_path in model_files:
            try:
                model_name = model_path.stem
                model_data = joblib.load(model_path)

                if isinstance(model_data, tuple) and len(model_data) == 2:
                    model, version = model_data
                else:
                    model = model_data
                    version = self._extract_version_from_filename(model_path.stem)

                # Store model and its info
                self.models[model_name] = model
                self.model_info[model_name] = ModelInfo(
                    name=model_name,
                    version=version,
                    path=str(model_path),
                    model_type=model_type
                )
                logging.info(f"Loaded {model_type} model: {model_name} (version {version})")
            except Exception as e:
                logging.error(f"Failed to load model {model_path}: {str(e)}")

    def _extract_version_from_filename(self, filename: str) -> str:
        try:
            if '_v' in filename:
                version_part = filename.split('_v')[-1]
                if all(part.isdigit() for part in version_part.split('.')):
                    return version_part
            return "1.0.0"
        except Exception:
            return "1.0.0"

    def save_model(self, model: Any, name: str, version: str, model_type: str) -> None:
        """Save model along with its version information."""
        try:
            directory = self.classifier_dir if model_type == "classifier" else self.regression_dir
            model_path = directory / f"{name}_v{version}.pkl"
            joblib.dump((model, version), model_path)
            logging.info(f"Saved {model_type} model {name} version {version}")
        except Exception as e:
            logging.error(f"Failed to save model {name}: {str(e)}")
            raise

    def get_model(self, name: str) -> Optional[Any]:
        return self.models.get(name)

    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        return self.model_info.get(name)

    def list_models(self, model_type: Optional[str] = None) -> List[ModelInfo]:
        """List models, optionally filtered by type."""
        if model_type:
            return [
                info for info in self.model_info.values() 
                if info.model_type == model_type
            ]
        return list(self.model_info.values())