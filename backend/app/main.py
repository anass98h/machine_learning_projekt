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
import time
import os
from pathlib import Path

# Define the directory to save Json files
POSENET_DATA = Path("posenet_data")
POSENET_DATA.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

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

class WeakestLinkResponse(BaseModel):
    model_name: str
    weakest_link: WeakLink

class ErrorResponse(BaseModel):
    detail: str
    error_type: str

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
async def list_regression_models():
    """List all regression models."""
    try:
        # Only return regression models
        return model_loader.list_models(model_type="regression")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list regression models: {str(e)}",
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
        # Check if both directories exist
        if not model_loader.regression_dir.exists() or not model_loader.classifier_dir.exists():
            return HealthResponse(
                status="unhealthy",
                models_loaded=0,
                available_models=[],
            )
        
        # Check if we have any models loaded
        models = model_loader.list_models()  # This gets all models, both regression and classifier
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
    try:
        # Only return classifier models
        return model_loader.list_models(model_type="classifier")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list categorizing models: {str(e)}",
            headers={"error_type": "model_list_error"}
        )

class WeakestLinkResponse(BaseModel):
    model_name: str
    weakest_link: WeakLink
    processing_time_ms: float  # Added field for processing time in milliseconds

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
    # Get model and verify it's a classifier
    model_info = model_loader.get_model_info(model_name)
    if not model_info or model_info.model_type != "classifier":
        raise HTTPException(
            status_code=404,
            detail=f"Classifier model '{model_name}' not found"
        )
    
    model = model_loader.get_model(model_name)
    
    try:
        
        
        # Read and process the file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        # Start timing
        start_time = time.time()
        # Make prediction
        predicted_class = model.predict(df)[0]
        
        # End timing and calculate duration in milliseconds
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        return WeakestLinkResponse(
            model_name=model_name,
            weakest_link=predicted_class,
            processing_time_ms=processing_time_ms 
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
            headers={"error_type": "prediction_error"}
        )

@app.post("/upload-posenet-data", 
          response_model=dict,
          responses={
              400: {"model": ErrorResponse},
              500: {"model": ErrorResponse}
          })
async def upload_posenet_data(file: UploadFile = File(...)):
    """
    Endpoint to upload PoseNet JSON data and save it to the server.
    """
    try:
        # Validate file type
        if not file.filename.endswith(".json"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Only JSON files are allowed.",
                headers={"error_type": "file_format_error"}
            )
        
        # Save the file to the posenet_data directory
        file_path = POSENET_DATA / file.filename
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        return {"message": f"File '{file.filename}' uploaded successfully."}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload file: {str(e)}",
            headers={"error_type": "upload_error"}
        )