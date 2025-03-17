import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
import mlflow.sklearn
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import custom_transformers

# Create directory for plots
os.makedirs("plots", exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Initiate mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("ensemble_regression_grid_search")

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

# Process the data through the preprocessor once
X_train_processed = preprocessor.fit_transform(x_train)
X_test_processed = preprocessor.transform(x_test)

# Create combinations of parameters to grid search
all_results = []
best_score = -float('inf')
best_model = None
best_params = {}

with mlflow.start_run():
    mlflow.set_tag("mlflow.runName", "ensemble_regression_grid_search_simplified")
    
    # Log basic parameters
    mlflow.log_params({
        "test_size": 0.2,
        "random_state": 42
    })
    
    # Select a subset of parameter combinations to test
    for bagged_lr_n_estimators in [5, 10]:
        for bagged_dt_n_estimators in [5, 10]:
            for rf_n_estimators in [50, 100]:
                for rf_max_depth in [None, 10]:
                    for stacking_cv in [3, 5]:
                        for passthrough in [True, False]:
                            for meta_fit_intercept in [True]:
                                for meta_positive in [False]:
                                    print(f"Testing: BaggedLR={bagged_lr_n_estimators}, BaggedDT={bagged_dt_n_estimators}, "
                                          f"RF={rf_n_estimators}/{rf_max_depth}, CV={stacking_cv}, "
                                          f"Passthrough={passthrough}")
                                    
                                    # Create the models with current parameters
                                    bagged_lr = BaggingRegressor(
                                        LinearRegression(), 
                                        n_estimators=bagged_lr_n_estimators,
                                        bootstrap=True,
                                        max_samples=0.9,
                                        random_state=42
                                    )
                                    
                                    bagged_dt = BaggingRegressor(
                                        DecisionTreeRegressor(
                                            max_depth=10 if rf_max_depth else None,
                                            min_samples_split=2,
                                            random_state=42
                                        ), 
                                        n_estimators=bagged_dt_n_estimators,
                                        bootstrap=True,
                                        max_samples=0.9,
                                        random_state=42
                                    )
                                    
                                    rf = RandomForestRegressor(
                                        n_estimators=rf_n_estimators,
                                        max_depth=rf_max_depth,
                                        min_samples_split=2,
                                        max_features='sqrt',
                                        random_state=42
                                    )
                                    
                                    # Define the base estimators
                                    estimators = [
                                        ('bagged_lr', bagged_lr),
                                        ('bagged_dt', bagged_dt),
                                        ('rf', rf)
                                    ]
                                    
                                    # Create meta-learner (final estimator)
                                    meta_learner = LinearRegression(
                                        fit_intercept=meta_fit_intercept,
                                        positive=meta_positive
                                    )
                                    
                                    # Create the stacking ensemble
                                    stacking_regressor = StackingRegressor(
                                        estimators=estimators,
                                        final_estimator=meta_learner,
                                        cv=stacking_cv,
                                        passthrough=passthrough
                                    )
                                    
                                    # Train and evaluate
                                    try:
                                        stacking_regressor.fit(X_train_processed, y_train)
                                        y_pred = stacking_regressor.predict(X_test_processed)
                                        
                                        mse = mean_squared_error(y_test, y_pred)
                                        r2 = r2_score(y_test, y_pred)
                                        
                                        # Store results
                                        result = {
                                            'bagged_lr_n_estimators': bagged_lr_n_estimators,
                                            'bagged_dt_n_estimators': bagged_dt_n_estimators,
                                            'rf_n_estimators': rf_n_estimators,
                                            'rf_max_depth': 'None' if rf_max_depth is None else str(rf_max_depth),
                                            'stacking_cv': stacking_cv,
                                            'passthrough': passthrough,
                                            'meta_fit_intercept': meta_fit_intercept,
                                            'meta_positive': meta_positive,
                                            'mse': mse,
                                            'r2': r2
                                        }
                                        all_results.append(result)
                                        
                                        print(f"  R²: {r2:.4f}, MSE: {mse:.4f}")
                                        
                                        # Check if this is the best model so far
                                        if r2 > best_score:
                                            best_score = r2
                                            best_params = {
                                                'bagged_lr_n_estimators': bagged_lr_n_estimators,
                                                'bagged_dt_n_estimators': bagged_dt_n_estimators,
                                                'rf_n_estimators': rf_n_estimators,
                                                'rf_max_depth': rf_max_depth,
                                                'stacking_cv': stacking_cv,
                                                'passthrough': passthrough,
                                                'meta_fit_intercept': meta_fit_intercept,
                                                'meta_positive': meta_positive
                                            }
                                            best_model = stacking_regressor
                                            print(f"  New best model found!")
                                    
                                    except Exception as e:
                                        print(f"  Error: {str(e)}")
    
    # Create a DataFrame of all results for analysis
    results_df = pd.DataFrame(all_results)
    
    # Sort by R² score (descending)
    results_df = results_df.sort_values('r2', ascending=False)
    
    # Display the top 5 configurations
    print("\nTop 5 Configurations:")
    for i, row in results_df.head(5).iterrows():
        print(f"R²: {row['r2']:.4f}, MSE: {row['mse']:.4f}, Params: {row[['bagged_lr_n_estimators', 'bagged_dt_n_estimators', 'rf_n_estimators', 'rf_max_depth', 'stacking_cv', 'passthrough']].to_dict()}")
    
    # Create just one key visualization - Top 10 configurations
    plt.figure(figsize=(12, 8))
    top_configs = results_df.head(10).copy()
    
    # Create clear configuration names
    top_configs['config_name'] = top_configs.apply(
        lambda x: f"BLR{x['bagged_lr_n_estimators']}-BDT{x['bagged_dt_n_estimators']}-"
                 f"RF{x['rf_n_estimators']}/{x['rf_max_depth']}-"
                 f"CV{x['stacking_cv']}-PT{'T' if x['passthrough'] else 'F'}",
        axis=1
    )
    
    # Sort by R² for better visualization
    top_configs = top_configs.sort_values('r2')
    
    # Create horizontal bar chart
    sns.barplot(x='r2', y='config_name', data=top_configs, palette='viridis')
    plt.title('Top 10 Ensemble Configurations by R² Score', fontsize=16)
    plt.xlabel('R² Score', fontsize=14)
    plt.ylabel('Configuration', fontsize=14)
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig('plots/top_configurations.png', dpi=300)
    mlflow.log_artifact('plots/top_configurations.png')
    plt.close()
    
    # Create the full pipeline with the best model
    best_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('stacking', best_model)
    ])
    
    # Refit on the entire training set
    best_pipeline.fit(x_train, y_train)
    final_y_pred = best_pipeline.predict(x_test)
    final_mse = mean_squared_error(y_test, final_y_pred)
    final_r2 = r2_score(y_test, final_y_pred)
    
    # Log the best parameters and metrics
    mlflow.log_params(best_params)
    mlflow.log_metrics({
        "best_mse": final_mse,
        "best_r2": final_r2
    })
    
    # Log the best model
    model_name = "ensemble_regression_model_grid_search"
    mlflow.sklearn.log_model(best_pipeline, model_name)
    
    # Register the model
    mlflow.register_model(
        model_uri=f"runs:/{mlflow.active_run().info.run_id}/{model_name}",
        name=model_name
    )
    
    print("\n===== Final Results =====")
    print(f"Best Parameters: {best_params}")
    print(f"Best MSE: {final_mse:.6f}")
    print(f"Best R² Score: {final_r2:.6f}")
    print("Model saved successfully!")
    print(f"Run ID: {mlflow.active_run().info.run_id}")