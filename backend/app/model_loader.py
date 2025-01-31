from dataclasses import dataclass
from typing import Dict, Optional, List, Any
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
    def __init__(self, models_dir: str = "models"):
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
                    model = joblib.load(model_path)
                    
                    # Store model and its info
                    self.models[model_name] = model
                    self.model_info[model_name] = ModelInfo(
                        name=model_name,
                        version="1.0.0",  # You might want to store this in the model
                        path=str(model_path)
                    )
                    
                    logging.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logging.error(f"Failed to load model {model_path}: {str(e)}")
                    
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            raise
    
    def get_model(self, name: str) -> Optional[Any]:
        """Get a specific model by name."""
        return self.models.get(name)
    
    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return self.model_info.get(name)
    
    def list_models(self) -> List[ModelInfo]:
        """List all available models."""
        return list(self.model_info.values())