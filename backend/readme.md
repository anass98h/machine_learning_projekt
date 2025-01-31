# Model Serving API Documentation

## Overview

This API provides endpoints for model serving and management. It loads machine learning models from files and serves predictions through a REST API. The service manages model loading, health monitoring, and prediction serving.

## Base URL - For now, we will probably change it later

```
http://localhost:8000
```

## Endpoints

### Health Check

Check the health status of the service and loaded models.

**Endpoint:** `GET /health`

**Purpose:**
This endpoint checks:

- If the models directory exists and is accessible
- If any models are currently loaded
- The total number of loaded models

**Status Definitions:**

- `healthy`: Models directory exists and at least one model is loaded and ready to serve predictions
- `degraded`: Models directory exists but no models are currently loaded (service can run but can't make predictions)
- `unhealthy`: Models directory doesn't exist or isn't accessible (service cannot function properly)

**Response Examples:**

_Healthy Response (200 OK) - Service is fully operational_

```json
{
  "status": "healthy",
  "models_loaded": 2,
  "available_models": ["model1", "model2"]
}
```

_Degraded Response (200 OK) - Service is running but cannot make predictions_

```json
{
  "status": "degraded",
  "models_loaded": 0,
  "available_models": []
}
```

_Unhealthy Response (200 OK) - Service is not operational_

```json
{
  "status": "unhealthy",
  "models_loaded": 0,
  "available_models": []
}
```

_Error Response (500 Internal Server Error) - Service encountered an error during health check_

```json
{
  "detail": "Health check failed: could not access models directory",
  "error_type": "health_check_error"
}
```

### Make Prediction

Get a prediction from a specific model.

**Endpoint:** `POST /predict/{model_name}`

**Purpose:**
This endpoint:

- Accepts input data for prediction
- Processes it through the specified model
- Returns a score (0-100) and its corresponding category

**Parameters:**

- `model_name` (path parameter): Name of the model to use for prediction
- `file` (form data): CSV file containing the input data

**Notes:**

- The score is a value between 0 and 100
- The category is determined based on the score ranges (see Score Categories section)
- The input CSV must match the model's expected features

**Response Examples:**

_Success Response (200 OK) - Prediction successfully generated_

```json
{
  "model_name": "model1",
  "category": "Great",
  "score": 85.5
}
```

_Model Not Found Response (404 Not Found) - Requested model isn't loaded_

```json
{
  "detail": "Model 'model1' not found",
  "error_type": "http_error"
}
```

_Invalid Input Response (500 Internal Server Error) - Input data couldn't be processed_

```json
{
  "detail": "Failed to process prediction: invalid input data",
  "error_type": "prediction_error"
}
```

### List Models

Get information about all available models in the system.

**Endpoint:** `GET /models`

**Purpose:**
This endpoint provides:

- List of all loaded models
- Version information for each model
- File path where each model is stored

**Response Examples:**

_Success Response (200 OK) - List of available models_

```json
[
  {
    "name": "model1",
    "version": "1.0.0",
    "path": "models/model1.pkl"
  },
  {
    "name": "model2",
    "version": "1.0.0",
    "path": "models/model2.pkl"
  }
]
```

_Error Response (500 Internal Server Error) - Failed to list models_

```json
{
  "detail": "Failed to list models: could not read models directory",
  "error_type": "model_list_error"
}
```

### Refresh Models

Trigger a reload of all models from the models directory.

**Endpoint:** `POST /refresh-models`

**Purpose:**
This endpoint:

- Initiates a background task to reload all models
- Useful when new models are added or existing models are updated
- Doesn't interrupt current predictions

**Notes:**

- The refresh happens asynchronously
- Current predictions continue using existing loaded models until refresh completes
- Success response means refresh started, not that it completed

**Response Examples:**

_Success Response (200 OK) - Refresh task started_

```json
{
  "message": "Model refresh started"
}
```

_Error Response (500 Internal Server Error) - Couldn't start refresh_

```json
{
  "detail": "Failed to start model refresh: could not access models directory",
  "error_type": "refresh_error"
}
```

## Score Categories

The prediction endpoint categorizes scores as follows:

- **Bad (0-39)**: Indicates poor performance or high risk
- **Good (40-69)**: Indicates acceptable performance or moderate risk
- **Great (70-89)**: Indicates strong performance or low risk
- **Excellent (90-100)**: Indicates exceptional performance or minimal risk

## Input Data Format

The prediction endpoint expects a CSV file with the appropriate features. The format must match what the model was trained on.

Example format:

```csv
feature1,feature2,feature3
0.5,0.3,0.8
0.7,0.2,0.9
```

**Important Notes:**

- Column names must match exactly what the model expects
- Data types must be appropriate for each feature
- No missing values are allowed unless the model specifically handles them

## Error Handling

All error responses follow a consistent format for easy parsing and handling:

```json
{
  "detail": "Description of what went wrong",
  "error_type": "type_of_error"
}
```

Common error types:

- `http_error`: General HTTP-related errors
- `prediction_error`: Errors during prediction processing
- `model_list_error`: Errors when listing models
- `health_check_error`: Errors during health check
- `refresh_error`: Errors during model refresh

## cURL Examples

### Health Check

```bash
curl http://localhost:8000/health
```

### Make Prediction

```bash
curl -X POST http://localhost:8000/predict/model1 \
     -F "file=@data.csv"
```

### List Models

```bash
curl http://localhost:8000/models
```

### Refresh Models

```bash
curl -X POST http://localhost:8000/refresh-models
```
