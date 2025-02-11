from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Tuple
from pathlib import Path
import logging
import joblib
import os



@dataclass
class ModelInfo:
    name: str
    version: str
    path: str

class ModelLoader:
    def __init__(self, models_dir: str = "../ML/saved models"):
        self.models_dir = Path(models_dir)
        self.models: Dict[str, Any] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        # Create models directory if it doesn't exist
        self.models_dir.mkdir(exist_ok=True)

    def load_models(self) -> None:
        """Load all models from the models directory."""
        try:
            # Clear existing models
            self.models = {}
            self.model_info = {}
            
            # List all .pkl files in the models directory
            model_files = list(self.models_dir.glob("*.pkl"))
            if not model_files:
                logging.warning(f"No model files found in {self.models_dir}")
                return

            for model_path in model_files:
                try:
                    model_name = model_path.stem  # filename without extension
                    model_data = joblib.load(model_path)
                    print(model_data)
                    if isinstance(model_data, tuple) and len(model_data) == 2:
                        model, version = model_data
                    else:
                        # If no version found, try to extract from filename
                        model = model_data
                        version = self._extract_version_from_filename(model_path.stem)

                    # Store model and its info
                    self.models[model_name] = model
                    self.model_info[model_name] = ModelInfo(
                        name=model_name,
                        version=version,
                        path=str(model_path)
                    )
                    logging.info(f"Loaded model: {model_name} (version {version})")
                except Exception as e:
                    logging.error(f"Failed to load model {model_path}: {str(e)}")
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            raise

    def _extract_version_from_filename(self, filename: str) -> str:
        try:
            
            if '_v' in filename:
                version_part = filename.split('_v')[-1]
                
                if all(part.isdigit() for part in version_part.split('.')):
                    return version_part
            return "1.0.0"  # default version if no version found
        except Exception:
            return "1.0.0"

    def save_model(self, model: Any, name: str, version: str) -> None:
        """Save model along with its version information."""
        try:
            model_path = self.models_dir / f"{name}_v{version}.pkl"
            # Save model and version together
            joblib.dump((model, version), model_path)
            logging.info(f"Saved model {name} version {version}")
        except Exception as e:
            logging.error(f"Failed to save model {name}: {str(e)}")
            raise

    def get_model(self, name: str) -> Optional[Any]:
        return self.models.get(name)

    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return self.model_info.get(name)

    def list_models(self) -> List[ModelInfo]:
        return list(self.model_info.values())