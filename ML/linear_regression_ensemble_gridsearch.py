import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import custom_transformers

# Start timing
start_time = time.time()

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
features_to_square = [
    "No_1_Angle_Deviation",
    "No_2_Angle_Deviation",
    "No_3_NASM_Deviation",
    "No_12_NASM_Deviation",
    "No_17_NASM_Deviation",
    "No_1_Time_Deviation",
    "No_2_Time_Deviation"
]

# Define a transformer to drop irrelevant columns
column_dropper = ColumnTransformer(
    [('columns_to_drop', 'drop', irrelevant_columns)],
    remainder="passthrough"
)

# Custom transformer to combine correlated features
feature_combiner = custom_transformers.CombineCorrelatedFeatures(symmetricalColumns)
features_to_square = custom_transformers.SquareFeatures(columns=features_to_square, replace=False) 

# Create a preprocessing pipeline that applies the custom feature combination,
# drops irrelevant columns, and then normalizes the features.
preprocessor = Pipeline([
    ('features_to_square', features_to_square),
    ('combine_sym', feature_combiner),
    ('columndrop', column_dropper),
    ('normalize', StandardScaler())
])

# Create base models with parameters we'll tune
lr = LinearRegression()
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(random_state=42)

# Define the ensemble pipeline with placeholders for base models
ensemble_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('stacking', StackingRegressor(
        estimators=[
            ('lr', lr),
            ('dt', dt),
            ('rf', rf)
        ],
        final_estimator=Ridge(),
        cv=5,
        passthrough=True
    ))
])

# Define the parameter grid for GridSearchCV
param_grid = {
    # RandomForest parameters
    'stacking__rf__n_estimators': [50, 100, 200],
    'stacking__rf__max_depth': [5, 10, 20, 30],  # Replace None with explicit values
    'stacking__rf__min_samples_leaf': [1, 2, 4],  # Add this parameter
    
    # DecisionTree parameters
    'stacking__dt__max_depth': [5, 10, 20, 30],  # Replace None with explicit values
    'stacking__dt__min_samples_split': [2, 5, 10],
    'stacking__dt__min_samples_leaf': [1, 2, 4],  # Add this parameter
    
    # Final estimator parameters
    'stacking__final_estimator__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]  # Wider range
}

print("Starting GridSearchCV...")
# Create GridSearchCV object
grid_search = GridSearchCV(
    estimator=ensemble_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,  # Use all available cores
    verbose=2
)

# Fit the grid search to find the best parameters
grid_search.fit(x_train, y_train)

# Print best parameters
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred = best_model.predict(x_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)  # Calculate MAE
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)  # Print MAE
print("R2 Score:", r2)
print(f"Training time: {time.time() - start_time:.2f} seconds")

# Save the best model locally
joblib.dump(best_model, 'best_ensemble_regression_model.pkl')
print("Best model saved successfully!")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values (with Optimized Ensemble)')
plt.savefig('optimized_regression_results.png')
plt.close()

# Optional: Feature importance analysis (if available for your model)
if hasattr(best_model['stacking'].estimators_[2], 'feature_importances_'):
    # Get feature names after preprocessing (simplified approach)
    feature_names = x.columns
    rf_model = best_model['stacking'].estimators_[2]
    
    # Get feature importances for RandomForest
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot top 15 features
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances (RandomForest)')
    plt.bar(range(min(15, len(indices))), 
            importances[indices][:15], 
            align='center')
    plt.xticks(range(min(15, len(indices))), 
               [feature_names[i] for i in indices[:15]], 
               rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()