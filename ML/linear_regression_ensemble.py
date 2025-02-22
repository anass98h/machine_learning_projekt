import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import custom_transformers

# Initiate mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("ensemble_regression")

# Load dataset
df = pd.read_excel("ML/data/AimoScore_WeakLink_big_scores.xls")
x = df.iloc[:, 1:-1]
y = df.iloc[:, 0]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define feature weights and symmetrical column pairs (as in your original code)
feature_weights = {
    7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2,  # FSM important features
    16: 2, 17: 2, 22: 2, 23: 4, 24: 4, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2,
    33: 2, 34: 2, 35: 2, 36: 2,              # NASM key features
}

symmetricalColumns = [
    (3, 5), (4, 6), (7, 10), (8, 11), (9, 12),   # FMS symmetry
    (13, 14), (16, 17), (20, 21), (23, 24)         # NASM symmetry
]

# Columns to drop (irrelevant ones)
irrelevant_columns = [27, 1, 25, 31, 3, 4, 7, 8, 9]

# Define a transformer to drop irrelevant columns
column_dropper = ColumnTransformer(
    [('columns_to_drop', 'drop', irrelevant_columns)],
    remainder="passthrough"
)

# Custom transformer to combine correlated features
feature_combiner = custom_transformers.CombineCorrelatedFeatures(symmetricalColumns)

# Create a preprocessing pipeline that applies the custom feature combination,
# drops irrelevant columns, and then normalizes the features.
preprocessor = Pipeline([
    ('combine_sym', feature_combiner),
    ('columndrop', column_dropper),
    ('normalize', StandardScaler())
])

# Create heterogeneous base models using bootstrap (bagging) for quasi‐independent training sets.
bagged_lr = BaggingRegressor(LinearRegression(), n_estimators=10, bootstrap=True, random_state=42)
bagged_dt = BaggingRegressor(DecisionTreeRegressor(random_state=42), n_estimators=10, bootstrap=True, random_state=42)
rf = RandomForestRegressor(n_estimators=50, random_state=42)

# Define the base estimators as a list of tuples.
estimators = [
    ('bagged_lr', bagged_lr),
    ('bagged_dt', bagged_dt),
    ('rf', rf)
]

# Use a stacking regressor as the ensemble aggregator (meta-learner).
# The meta-learner (here, LinearRegression) learns to combine the base models’ predictions.
stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
    cv=5,             # 5-fold cross validation for meta-learner training
    passthrough=True  # Optionally include original features along with predictions
)

# Build the full ensemble pipeline including preprocessing and the stacking ensemble.
ensemble_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('stacking', stacking_regressor)
])

# Train the ensemble model and log the experiment using mlflow.
with mlflow.start_run():
    mlflow.set_tag("mlflow.runName", "ensemble_regression_LR_DT_RF")
    ensemble_pipeline.fit(x_train, y_train)
    y_pred = ensemble_pipeline.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    model_name = "ensemble_regression_model_v1"
    mlflow.sklearn.log_model(ensemble_pipeline, model_name)

    mlflow.register_model(
        model_uri=f"runs:/{mlflow.active_run().info.run_id}/{model_name}",
        name=model_name
    )

    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)


    print("Model saved successfully!")
