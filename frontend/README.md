## Overview

This is the frontend for the Automated Movement Assessment web application. It provides an interface for users to interact with the model-serving backend, allowing them to upload CSV files and receive predictions. The frontend is built using **React** and includes a **dashboard** to monitor system health, manage models, and visualize prediction results.

## Features

- **Health Status Monitoring**: Displays the system health and number of loaded models.
- **Model Selection**: Allows users to select a model from the list of available models.
- **File Upload & Prediction**: Users can upload a CSV file and request a prediction.
- **Prediction Visualization**: Displays results with a **gauge chart** and categorized score.
- **Model Management**: Provides functionality to refresh the list of available models.

## Prerequisites

Before running the frontend, ensure you have:

- **Node.js**
- **npm** or **yarn** installed
- The backend API running at `http://localhost:8000`

## Running the Application

To start the development server, run:

```bash
npm start
```

The application will be available at `http://localhost:3000`.

## API Integration

The frontend communicates with the backend API via the following endpoints:

### 1. Fetch Available Models
- **Endpoint:** `GET /models`
- **Purpose:** Retrieves a list of models available for prediction.
- **Frontend Implementation:** Models are displayed in a dropdown for user selection.

### 2. Health Check
- **Endpoint:** `GET /health`
- **Purpose:** Retrieves the system health status and number of loaded models.
- **Frontend Implementation:** Displays health status and model count.

### 3. Refresh Models
- **Endpoint:** `POST /refresh-models`
- **Purpose:** Triggers a reload of all models.
- **Frontend Implementation:** Calls this endpoint when users request to refresh the model list.

### 4. Make Prediction
- **Endpoint:** `POST /predict/{model_name}`
- **Purpose:** Submits a CSV file for prediction and receives a score and category.
- **Frontend Implementation:** The score is visualized using a **gauge chart**.

## User Interface

The UI consists of the following sections:

1. **Navigation Bar**: Displays project information.
2. **System Health**: Shows the backend health status and loaded models.
3. **Model Management**: Lists available models and includes a refresh button.
4. **File Upload & Prediction**: Allows users to upload a CSV file and make predictions.
5. **Results Dashboard**: Displays the prediction results, including:
   - The **selected model name**
   - The **raw score**
   - The **prediction category**
   - A **gauge chart** visualization
   - A **legend** explaining the score categories

## Score Categories

Predictions are classified into categories:

- **Bad (0-39)**: Poor performance or high risk.
- **Good (40-69)**: Acceptable performance or moderate risk.
- **Great (70-89)**: Strong performance or low risk.
- **Excellent (90-100)**: Exceptional performance or minimal risk.

## Error Handling

Errors are displayed in a user-friendly way, with messages for:
- Failed API requests
- Invalid file formats
- Missing input fields