from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from typing import List
from pydantic import BaseModel
import logging
from app.model_loader import ModelLoader, ModelInfo

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
        score = float(np.clip(prediction[0], 0, 100))
        category = categorize_score(score)
        
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
