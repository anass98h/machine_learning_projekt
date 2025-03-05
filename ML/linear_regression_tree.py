import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import custom_transformers
from lib.data_loader import load_data

# Initiate mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tree_regression")

# Load dataset
df = load_data()
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

# Features for polynomial transformations
features_to_square = [
    "No_1_Angle_Deviation", "No_2_Angle_Deviation", "No_3_NASM_Deviation",
    "No_12_NASM_Deviation", "No_17_NASM_Deviation", "No_1_Time_Deviation",
    "No_2_Time_Deviation", "No_18_NASM_Deviation", "No_9_NASM_Deviation",
    "No_8_Angle_Deviation", "No_13_Angle_Deviation", "No_5_NASM_Deviation",
    "No_7_Angle_Deviation", "No_2_NASM_Deviation", "No_9_Angle_Deviation",
    "No_10_NASM_Deviation"
]

features_to_cubic = [
    "No_10_NASM_Deviation", "No_5_Angle_Deviation", "No_1_NASM_Deviation",
    "No_7_Angle_Deviation", "No_2_NASM_Deviation", "No_4_Angle_Deviation",
    "No_2_Angle_Deviation", "No_6_NASM_Deviation", "No_12_NASM_Deviation",
    "No_14_NASM_Deviation", "No_15_NASM_Deviation", "No_6_Angle_Deviation",
    "No_7_NASM_Deviation"
]

# Columns to drop
irrelevant_columns = [27, 1, 25, 31, 3, 4, 7, 8, 9]

# Create preprocessing pipeline
column_dropper = ColumnTransformer(
    [('columns_to_drop', 'drop', irrelevant_columns)],
    remainder="passthrough"
)

feature_combiner = custom_transformers.CombineCorrelatedFeatures(symmetricalColumns)
features_to_square_transformer = custom_transformers.SquareFeatures(columns=features_to_square, replace=False)
features_to_cubic_transformer = custom_transformers.CubicFeatures(columns=features_to_cubic, replace=False)

preprocessor = Pipeline([
    ('features_to_cubic', features_to_cubic_transformer),
    ('features_to_square', features_to_square_transformer),
    ('combine_sym', feature_combiner),
    ('columndrop', column_dropper),
    ('normalize', StandardScaler())
])

# Define our tree models with different configurations
models = {
    'decision_tree': DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),
    'random_forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),
    'gradient_boosting': GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
}

# Compare all models
results = {}

for name, model in models.items():
    # Create pipeline with preprocessing and the model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Use cross-validation for more robust evaluation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {'mse': 'neg_mean_squared_error', 'r2': 'r2'}
    
    with mlflow.start_run(run_name=f"tree_regression_{name}"):
        # Perform cross-validation
        cv_results = cross_validate(
            pipeline, x, y, cv=cv, scoring=scoring, return_train_score=True
        )
        
        # Calculate metrics
        train_mse = -np.mean(cv_results['train_mse'])
        test_mse = -np.mean(cv_results['test_mse'])
        train_r2 = np.mean(cv_results['train_r2'])
        test_r2 = np.mean(cv_results['test_r2'])
        
        # Store results
        results[name] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        # Log parameters and metrics
        mlflow.log_param("model_type", name)
        if name == 'decision_tree':
            mlflow.log_param("max_depth", model.max_depth)
            mlflow.log_param("min_samples_split", model.min_samples_split)
            mlflow.log_param("min_samples_leaf", model.min_samples_leaf)
        elif name == 'random_forest':
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("max_depth", model.max_depth)
        elif name == 'gradient_boosting':
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("learning_rate", model.learning_rate)
            mlflow.log_param("max_depth", model.max_depth)
            mlflow.log_param("subsample", model.subsample)
        
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        
        # Train the final model on the entire training set
        pipeline.fit(x_train, y_train)
        
        # Evaluate on test set
        y_pred = pipeline.predict(x_test)
        test_mse_final = mean_squared_error(y_test, y_pred)
        test_r2_final = r2_score(y_test, y_pred)
        
        mlflow.log_metric("final_test_mse", test_mse_final)
        mlflow.log_metric("final_test_r2", test_r2_final)
        
        # Log the model
        model_name = f"tree_regression_{name}_model_v1"
        mlflow.sklearn.log_model(pipeline, model_name)
        
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=model_name
        )
        
        mlflow.register_model(
            model_uri=model_info.model_uri,
            name=model_name
        )
        
        print(f"\nModel: {name}")
        print(f"Cross-validation Train MSE: {train_mse:.4f}")
        print(f"Cross-validation Test MSE: {test_mse:.4f}")
        print(f"Cross-validation Train R²: {train_r2:.4f}")
        print(f"Cross-validation Test R²: {test_r2:.4f}")
        print(f"Final Test MSE: {test_mse_final:.4f}")
        print(f"Final Test R²: {test_r2_final:.4f}")

# Print a summary of all models for comparison
print("\n========== SUMMARY OF TREE-BASED MODELS ==========")
for name, metrics in results.items():
    print(f"\nModel: {name}")
    print(f"CV Test MSE: {metrics['test_mse']:.4f}")
    print(f"CV Test R²: {metrics['test_r2']:.4f}")
    print(f"Train-Test R² difference: {metrics['train_r2'] - metrics['test_r2']:.4f}")