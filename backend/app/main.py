from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from typing import List
from enum import Enum
from pydantic import BaseModel
import logging
from app.model_loader import ModelLoader, ModelInfo

class WeakLink(str, Enum):
    FORWARD_HEAD = "ForwardHead"
    LEFT_ARM_FALL_FORWARD = "LeftArmFallForward"
    RIGHT_ARM_FALL_FORWARD = "RightArmFallForward"
    LEFT_SHOULDER_ELEVATION = "LeftShoulderElevation"
    RIGHT_SHOULDER_ELEVATION = "RightShoulderElevation"
    EXCESSIVE_FORWARD_LEAN = "ExcessiveForwardLean"
    LEFT_ASYMMETRICAL_WEIGHT_SHIFT = "LeftAsymmetricalWeightShift"
    RIGHT_ASYMMETRICAL_WEIGHT_SHIFT = "RightAsymmetricalWeightShift"
    LEFT_KNEE_MOVES_INWARD = "LeftKneeMovesInward"
    RIGHT_KNEE_MOVES_INWARD = "RightKneeMovesInward"
    LEFT_KNEE_MOVES_OUTWARD = "LeftKneeMovesOutward"
    RIGHT_KNEE_MOVES_OUTWARD = "RightKneeMovesOutward"
    LEFT_HEEL_RISES = "LeftHeelRises"
    RIGHT_HEEL_RISES = "RightHeelRises"

class ModelInfo(BaseModel):
    name: str
    version: str
    path: str
    model_type: str = "predictor"

class WeakestLinkResponse(BaseModel):
    model_name: str
    weakest_link: WeakLink

class ErrorResponse(BaseModel):
    detail: str
    error_type: str

# Mock data
MOCK_MODELS = [
    ModelInfo(
        name="movement_classifier_v1",
        version="1.0.0",
        path="models/movement_classifier_v1.pkl",
        model_type="categorizer"
    ),
    ModelInfo(
        name="movement_classifier_v2",
        version="2.0.0",
        path="models/movement_classifier_v2.pkl",
        model_type="categorizer"
    ),
    ModelInfo(
        name="experimental_classifier",
        version="0.1.0",
        path="models/experimental_classifier.pkl",
        model_type="categorizer"
    )
]

# Response models
class PredictionResponse(BaseModel):
    model_name: str
    category: str
    score: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    available_models: List[str]

class ErrorResponse(BaseModel):
    detail: str
    error_type: str

def categorize_score(score: float) -> str:
    """Categorize a score based on defined thresholds."""
    if score < 40:
        return "Bad"
    elif score < 70:
        return "Good"
    elif score < 90:
        return "Great"
    else:
        return "Excellent"

app = FastAPI(title="Model Serving API")
model_loader = ModelLoader()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.on_event("startup")
async def startup_event():
    """Load all models during startup."""
    try:
        model_loader.load_models()
    except Exception as e:
        logging.error(f"Failed to load models during startup: {str(e)}")

@app.post("/predict/{model_name}", 
         response_model=PredictionResponse,
         responses={
             404: {"model": ErrorResponse},
             500: {"model": ErrorResponse}
         })
async def predict(
    model_name: str,
    file: UploadFile = File(...)
):
    # First check if model exists
    model = model_loader.get_model(model_name)
    model_info = model_loader.get_model_info(model_name)
    
    if not model or not model_info:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )
    
    try:
        # Read and process the file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Make prediction
        prediction = model.predict(df)
        score = float(np.clip(prediction[0] , 0, 1))
        category = categorize_score(score * 100)
        
        return PredictionResponse(
            model_name=model_name,
            category=category,
            score=score
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
            headers={"error_type": "prediction_error"}
        )

@app.get("/models", 
         response_model=List[ModelInfo],
         responses={
             500: {"model": ErrorResponse}
         })
async def list_models():
    """List all available models."""
    try:
        models = model_loader.list_models()
        return models
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}",
            headers={"error_type": "model_list_error"}
        )

@app.post("/refresh-models", 
         response_model=dict,
         responses={
             500: {"model": ErrorResponse}
         })
async def refresh_models(background_tasks: BackgroundTasks):
    """Refresh models from the models directory."""
    try:
        background_tasks.add_task(model_loader.load_models)
        return {"message": "Model refresh started"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start model refresh: {str(e)}",
            headers={"error_type": "refresh_error"}
        )

@app.get("/health", 
         response_model=HealthResponse,
         responses={
             500: {"model": ErrorResponse}
         })
async def health():
    """Health check endpoint."""
    try:
        # Check if models directory exists
        if not model_loader.models_dir.exists():
            return HealthResponse(
                status="unhealthy",
                models_loaded=0,
                available_models=[],
            )
        
        # Check if we have any models loaded
        models = model_loader.list_models()
        if not models:
            return HealthResponse(
                status="degraded",
                models_loaded=0,
                available_models=[],
            )
        
        # Everything is fine
        return HealthResponse(
            status="healthy",
            models_loaded=len(model_loader.models),
            available_models=[info.name for info in models]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}",
            headers={"error_type": "health_check_error"}
        )

@app.get("/categorizing-models", 
         response_model=List[ModelInfo],
         responses={
             500: {"model": ErrorResponse}
         })
async def list_categorizing_models():
    """List all models used for weakest link categorization."""
    return MOCK_MODELS

@app.post("/classify-weakest-link/{model_name}", 
         response_model=WeakestLinkResponse,
         responses={
             404: {"model": ErrorResponse},
             500: {"model": ErrorResponse},
             422: {"model": ErrorResponse}
         })
async def classify_weakest_link(
    model_name: str,
    file: UploadFile = File(...)
):
    # Check if model exists in our mock data
    model_exists = any(model.name == model_name for model in MOCK_MODELS)
    if not model_exists:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )
    
    try:
        # Mock prediction - in reality, this would analyze the CSV
        # Here we'll return a random weakest link for testing
        import random
        predicted_class = random.choice(list(WeakLink))
        
        return WeakestLinkResponse(
            model_name=model_name,
            weakest_link=predicted_class
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
            headers={"error_type": "prediction_error"}
        )