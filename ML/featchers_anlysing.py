import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Change this path to your .xls file location
file_path = "ML/data/AimoScore_WeakLink_big_scores.xls"

# Load the data
df = pd.read_excel(file_path)

# Assume the first column is the target (response) and the rest are features
target = df.iloc[:, 0]
features = df.iloc[:, 1:]

print("Comparing linear vs. quadratic fits for each feature:")
print("-" * 60)

# Loop through each feature column
for col in features.columns:
    X = features[[col]]
    y = target

    # Fit the simple linear model
    lin_model = LinearRegression().fit(X, y)
    y_pred_lin = lin_model.predict(X)
    r2_lin = r2_score(y, y_pred_lin)
    
    # Prepare a quadratic feature: [x, x^2]
    X_quad = np.hstack([X, X**2])
    quad_model = LinearRegression().fit(X_quad, y)
    y_pred_quad = quad_model.predict(X_quad)
    r2_quad = r2_score(y, y_pred_quad)
    
    improvement = r2_quad - r2_lin
    print(f"Feature: {col}")
    print(f"  Linear R²:    {r2_lin:.4f}")
    print(f"  Quadratic R²: {r2_quad:.4f}")
    print(f"  Improvement:  {improvement:.4f}")
    print("-" * 60)
    
    # Optionally, plot the data and quadratic fit
    plt.figure(figsize=(6, 4))
    plt.scatter(X, y, alpha=0.5, label='Data')
    
    # To plot a smooth curve, sort the values
    sort_idx = X[col].argsort()
    X_sorted = X.iloc[sort_idx]
    # Re-compute predictions on sorted X for a smooth line
    X_sorted_array = np.hstack([X_sorted, X_sorted**2])
    y_sorted_quad = quad_model.predict(X_sorted_array)
    
    plt.plot(X_sorted, y_sorted_quad, color='red', label='Quadratic fit')
    plt.xlabel(col)
    plt.ylabel("Target")
    plt.title(f"Feature: {col}")
    plt.legend()
    plt.tight_layout()
    plt.show()
