import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import custom_transformers


# Plots the effect of C on the CV performance
def plot_C(results_df, best_params):
    # Set remaining parameters to be fixed at their best values
    best_kernel = best_params['svr__kernel']
    results_filtered = results_df[results_df['param_svr__kernel'] == best_kernel]

    if best_kernel in ['rbf', 'sigmoid', 'poly']:
        best_gamma = best_params['svr__gamma']
        results_filtered = results_filtered[results_filtered['param_svr__gamma'] == best_gamma]
        if best_kernel == 'poly':
            best_degree = best_params['svr__degree']
            results_filtered = results_filtered[results_filtered['param_svr__degree'] == best_degree]

    # Find the R2 values for each C
    results_filtered['C'] = results_filtered['param_svr__C'].astype(float)
    grouped = results_filtered.groupby('C')['mean_test_score'].mean().reset_index()

    # Display plot
    plt.plot(grouped['C'], grouped['mean_test_score'], marker='o', linestyle='-')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Mean CV R2 Score')
    plt.title('Effect of C on CV Performance')
    plt.grid(True)
    plt.show()


# Plots the effect of gamma on the CV performance
def plot_gamma(results_df, best_params):
    # Set remaining parameters to be fixed at their best values
    best_kernel = best_params['svr__kernel']
    if best_kernel not in ['rbf', 'sigmoid', 'poly']:
        print(f"Kernel '{best_kernel}' does not use gamma. Cannot plot gamma effect.")
        return
    results_filtered = results_df[results_df['param_svr__kernel'] == best_kernel]
    
    best_C = best_params['svr__C']
    results_filtered = results_filtered[results_filtered['param_svr__C'] == best_C]

    if best_kernel == 'poly':
        best_degree = best_params['svr__degree']
        results_filtered = results_filtered[results_filtered['param_svr__degree'] == best_degree]
    
    # Find the R2 values for each gamma
    results_filtered = results_filtered.copy()
    results_filtered['gamma'] = results_filtered['param_svr__gamma'].astype(float)
    grouped = results_filtered.groupby('gamma')['mean_test_score'].mean().reset_index()
    
    # Display plot
    plt.plot(grouped['gamma'], grouped['mean_test_score'], marker='o', linestyle='-')
    plt.xscale('log')
    plt.xlabel('Gamma')
    plt.ylabel('Mean CV R2 Score')
    plt.title('Effect of Gamma on CV Performance')
    plt.grid(True)
    plt.show()


# Initiate mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("SVM_regression")

# Load dataset
df = pd.read_excel("ML/data/AimoScore_WeakLink_big_scores.xls")
x = df.iloc[:, 1:-2]
y = df.iloc[:, 0]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a pipeline that scales data and then applies SVR
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR())
])

# Parameter grid (degree is only used for the polynomial kernel, and gamma only used for the non-linear kernels)
C_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
gamma_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
degree_values = [3, 4, 5]
PARAM_GRID = [
    {
        'svr__kernel': ['linear'],
        'svr__C': C_values
    },
    {
        'svr__kernel': ['rbf', 'sigmoid'],
        'svr__C': C_values,
        'svr__gamma': gamma_values
    },
    {
        'svr__kernel': ['poly'],
        'svr__C': C_values,
        'svr__gamma': gamma_values,
        'svr__degree': degree_values
    }
]

grid_search = GridSearchCV(estimator=pipeline, param_grid=PARAM_GRID, scoring='r2', n_jobs=-1)

with mlflow.start_run():
    mlflow.set_tag("mlflow.runName", "SVR_CV")
    grid_search.fit(x_train, y_train)

    y_pred = grid_search.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results_df = pd.DataFrame(grid_search.cv_results_)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best parameters", best_params)
    print("mean CV R2", best_score)
    print("Test R2:", r2)
    print("Test MSE:", mse)

    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("best_params", best_params)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mean_CV_r2", best_score)

    model_name = "SVR_V1"
    mlflow.sklearn.log_model(grid_search, model_name)

    mlflow.register_model(
        model_uri=f"runs:/{mlflow.active_run().info.run_id}/{model_name}",
        name=model_name
    )

    print("Model saved successfully!")

    plot_C(results_df, best_params)
    plot_gamma(results_df, best_params)
