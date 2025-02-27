import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pathlib import Path
import os

def fit_linear_model(X, y):
    """Fit a linear model and return the model and R²."""
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    return model, y_pred, r2

def fit_quadratic_model(X, y):
    """Fit a quadratic model and return the model and R²."""
    X_quad = np.hstack([X, X**2])
    model = LinearRegression().fit(X_quad, y)
    y_pred = model.predict(X_quad)
    r2 = r2_score(y, y_pred)
    return model, y_pred, r2, X_quad

def fit_cubic_model(X_quad, y):
    """Fit a cubic model and return the model and R²."""
    X = X_quad[:, 0].reshape(-1, 1)  # Extract original X from X_quad
    X_cubic = np.hstack([X_quad, X**3])
    model = LinearRegression().fit(X_cubic, y)
    y_pred = model.predict(X_cubic)
    r2 = r2_score(y, y_pred)
    return model, y_pred, r2

def compare_linear_vs_quadratic(X, y, feature_name):
    """
    Compare linear vs quadratic fits for a feature and return the feature name
    and improvement if the quadratic model shows improvement over the linear model.
    """
    # Fit models
    lin_model, y_pred_lin, r2_lin = fit_linear_model(X, y)
    quad_model, y_pred_quad, r2_quad, X_quad = fit_quadratic_model(X, y)
    
    # Calculate improvement
    improvement = r2_quad - r2_lin
    
    # Print results
    print(f"Feature: {feature_name}")
    print(f"  Linear R²:    {r2_lin:.4f}")
    print(f"  Quadratic R²: {r2_quad:.4f}")
    print(f"  Improvement:  {improvement:.4f}")
    print("-" * 60)
    
    return (feature_name, improvement) if improvement > 0 else None

def compare_quadratic_vs_cubic(X, y, feature_name):
    """
    Compare quadratic vs cubic fits for a feature and return the feature name
    and improvement if the cubic model shows improvement over the quadratic model.
    """
    # Fit models
    lin_model, y_pred_lin, r2_lin = fit_linear_model(X, y)
    quad_model, y_pred_quad, r2_quad, X_quad = fit_quadratic_model(X, y)
    cubic_model, y_pred_cubic, r2_cubic = fit_cubic_model(X_quad, y)
    
    # Calculate improvements
    quad_improvement = r2_quad - r2_lin
    cubic_improvement = r2_cubic - r2_quad
    total_improvement = r2_cubic - r2_lin
    
    # Print results
    print(f"Feature: {feature_name}")
    print(f"  Linear R²:     {r2_lin:.4f}")
    print(f"  Quadratic R²:  {r2_quad:.4f}")
    print(f"  Cubic R²:      {r2_cubic:.4f}")
    print(f"  Quad vs Lin:   {quad_improvement:.4f}")
    print(f"  Cubic vs Quad: {cubic_improvement:.4f}")
    print(f"  Total gain:    {total_improvement:.4f}")
    print("-" * 60)
    
    return (feature_name, cubic_improvement) if cubic_improvement > 0 else None

# Change this path to your .xls file location
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "data", "AimoScore_WeakLink_big_scores.xls")

# Load the data
df = pd.read_excel(file_path)

# Assume the first column is the target (response) and the rest are features
target = df.iloc[:, 0]
features = df.iloc[:, 1:-2]

print("Comparing linear vs. quadratic fits for each feature:")
print("-" * 60)

# Array to store features with positive improvement
improved_features_quad = []
improved_features_cubic = []

# Loop through each feature column
for col in features.columns:
    X = features[[col]]
    y = target
    
    # Compare linear vs quadratic
    quad_result = compare_linear_vs_quadratic(X, y, col)
    if quad_result:
        improved_features_quad.append(quad_result)
    
    # Compare quadratic vs cubic
    cubic_result = compare_quadratic_vs_cubic(X, y, col)
    if cubic_result:
        improved_features_cubic.append(cubic_result)

# Sort by improvement value in descending order
improved_features_quad.sort(key=lambda x: x[1], reverse=True)
improved_features_cubic.sort(key=lambda x: x[1], reverse=True)

# Print summary of features with positive improvement for quadratic terms
print("\nFeatures with positive improvement when using quadratic terms (sorted by improvement):")
if improved_features_quad:
    for feature_name, improvement in improved_features_quad:
        print(f"- {feature_name}: {improvement:.4f}")
    print(f"\nTotal: {len(improved_features_quad)} features showed improvement with quadratic terms")
else:
    print("No features showed improvement with quadratic terms")

# Print summary of features with positive improvement for cubic terms
print("\nFeatures with additional improvement when using cubic terms (sorted by improvement):")
if improved_features_cubic:
    for feature_name, improvement in improved_features_cubic:
        print(f"- {feature_name}: {improvement:.4f}")
    print(f"\nTotal: {len(improved_features_cubic)} features showed additional improvement with cubic terms")
else:
    print("No features showed additional improvement with cubic terms")
