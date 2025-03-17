import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import mlflow
import mlflow.sklearn
import custom_transformers
from lib.data_loader import load_data

# Initialize MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("linear_regression_grid_search_viz")

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Load data
df = load_data()
x = df.iloc[:,1:-1]
y = df.iloc[:,0]

# Feature engineering configurations
feature_weights = {
    7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2,  # FSM important features
    16: 2, 17: 2, 22: 2, 23: 4, 24: 4, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2, 33: 2, 34: 2, 35: 2, 36: 2,  # NASM key features
}

symmetricalColumns = [
    (3, 5), (4, 6), (7, 10), (8, 11), (9, 12),  # FMS Symmetry
    (13, 14), (16, 17), (20, 21), (23, 24)  # NASM symmetry
]

features_to_square = [
    "No_1_Angle_Deviation",
    "No_2_Angle_Deviation",
    "No_3_NASM_Deviation",
    "No_12_NASM_Deviation",
    "No_17_NASM_Deviation",
    "No_1_Time_Deviation",
    "No_2_Time_Deviation"
]

features_to_cubic = [
    "No_10_NASM_Deviation",
    "No_5_Angle_Deviation",
    "No_1_NASM_Deviation",
    "No_7_Angle_Deviation",
    "No_2_NASM_Deviation",
    "No_4_Angle_Deviation",
    "No_2_Angle_Deviation",
    "No_6_NASM_Deviation",
    "No_12_NASM_Deviation",
    "No_14_NASM_Deviation",
    "No_15_NASM_Deviation",
    "No_6_Angle_Deviation",
    "No_7_NASM_Deviation"
]

irrelevant_columns = [27, 1, 25, 31, 3, 4, 7, 8, 9]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create transformers
column_dropper = ColumnTransformer(
    [('columns_to_drop', 'drop', irrelevant_columns)],
    remainder="passthrough"
)

feature_combiner = custom_transformers.CombineCorrelatedFeatures(symmetricalColumns)
symmetrical_Columns = custom_transformers.symmetricalColumns(symmetricalColumns)
feature_weights = custom_transformers.FeatureWeights(feature_weights)
features_to_square = custom_transformers.SquareFeatures(columns=features_to_square, replace=False) 
features_to_cubic = custom_transformers.CubicFeatures(columns=features_to_cubic, replace=False)

# Create pipeline
pipeline = Pipeline(
    steps=[       
        ('features_to_cubic', features_to_cubic),
        ('features_to_square', features_to_square),
        ('combine_sym', feature_combiner),
        ('columndrop', column_dropper),
        ('normalize', StandardScaler()),
        ('poly', PolynomialFeatures(include_bias=False)),
        ('model', LinearRegression())
    ]
)

# Parameter grid for GridSearchCV - same as in your original code
param_grid = {
    'poly__degree': [1, 2],  # Only linear and quadratic
    'model__fit_intercept': [True, False],
    'model__positive': [True, False]
}

# Create Grid Search
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=kf,
    scoring='r2',
    n_jobs=-1,
    verbose=1,
    return_train_score=True  # Added to get training scores for learning curves
)

with mlflow.start_run():
    mlflow.set_tag("mlflow.runName", "linear_regression_grid_search_viz")
    
    # Log parameters
    mlflow.log_params({
        "model_type": "LinearRegression with GridSearchCV and Visualization",
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 5
    })
    
    # Fit the grid search
    grid_search.fit(x_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Predict and evaluate
    y_pred = best_model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Display results
    print("Best parameters:", grid_search.best_params_)
    print("Test set R² score:", r2)
    print("Test set MSE:", mse)
    
    # Display all CV results in the same format as the original code
    results = grid_search.cv_results_
    for mean, std, params in zip(results['mean_test_score'], results['std_test_score'], results['params']):
        degree = params['poly__degree']
        print(f"Degree: {degree}, CV R² Score: {mean:.4f}, Std: {std:.4f}, Params: {params}")
    
    # Get results as DataFrame for easier plotting
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Create directory for plots
    import os
    os.makedirs("plots", exist_ok=True)
    
    # 1. Performance Comparison Bar Chart
    plt.figure(figsize=(14, 8))
    # Create a unique label for each configuration
    results_df['config'] = results_df.apply(
        lambda x: f"Degree={x['param_poly__degree']}, Intercept={x['param_model__fit_intercept']}, Positive={x['param_model__positive']}", 
        axis=1
    )
    # Sort by mean test score
    sorted_df = results_df.sort_values('mean_test_score', ascending=False)
    
    ax = sns.barplot(x='config', y='mean_test_score', data=sorted_df)
    plt.title('R² Score for Different Model Configurations')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Mean R² Score (Cross-Validation)')
    plt.tight_layout()
    plt.savefig('plots/performance_comparison.png', dpi=300)
    mlflow.log_artifact('plots/performance_comparison.png')
    plt.close()
    
    # 2. Parameter Importance Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Degree impact
    sns.boxplot(x='param_poly__degree', y='mean_test_score', data=results_df, ax=axes[0])
    axes[0].set_title('Impact of Polynomial Degree')
    axes[0].set_xlabel('Polynomial Degree')
    axes[0].set_ylabel('Mean R² Score')
    
    # Intercept impact
    sns.boxplot(x='param_model__fit_intercept', y='mean_test_score', data=results_df, ax=axes[1])
    axes[1].set_title('Impact of Fit Intercept')
    axes[1].set_xlabel('Fit Intercept')
    axes[1].set_ylabel('Mean R² Score')
    
    # Positive coefficient impact
    sns.boxplot(x='param_model__positive', y='mean_test_score', data=results_df, ax=axes[2])
    axes[2].set_title('Impact of Positive Coefficients')
    axes[2].set_xlabel('Positive Coefficients Only')
    axes[2].set_ylabel('Mean R² Score')
    
    plt.tight_layout()
    plt.savefig('plots/parameter_importance.png', dpi=300)
    mlflow.log_artifact('plots/parameter_importance.png')
    plt.close()
    
    # 3. Parameter Interaction Heatmap (for degree=1)
    plt.figure(figsize=(10, 8))
    pivot_df = results_df[results_df['param_poly__degree'] == 1].pivot_table(
        values='mean_test_score', 
        index='param_model__fit_intercept',
        columns='param_model__positive'
    )
    ax = sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt=".3f")
    plt.title('R² Score Heatmap for Linear Models (Degree=1)')
    plt.xlabel('Positive Coefficients')
    plt.ylabel('Fit Intercept')
    plt.tight_layout()
    plt.savefig('plots/parameter_interaction_linear.png', dpi=300)
    mlflow.log_artifact('plots/parameter_interaction_linear.png')
    plt.close()
    
    # 4. Parameter Interaction Heatmap (for degree=2)
    plt.figure(figsize=(10, 8))
    pivot_df = results_df[results_df['param_poly__degree'] == 2].pivot_table(
        values='mean_test_score', 
        index='param_model__fit_intercept',
        columns='param_model__positive'
    )
    ax = sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt=".3f")
    plt.title('R² Score Heatmap for Quadratic Models (Degree=2)')
    plt.xlabel('Positive Coefficients')
    plt.ylabel('Fit Intercept')
    plt.tight_layout()
    plt.savefig('plots/parameter_interaction_quadratic.png', dpi=300)
    mlflow.log_artifact('plots/parameter_interaction_quadratic.png')
    plt.close()
    
    # 5. CV Score Distribution
    plt.figure(figsize=(14, 8))
    
    # Extract scores for each fold for the top 4 configurations
    top_configs = sorted_df.head(4)['config'].values
    plot_data = []
    
    for idx, config in enumerate(top_configs):
        config_row = results_df[results_df['config'] == config].iloc[0]
        for fold in range(kf.n_splits):
            score = config_row[f'split{fold}_test_score']
            plot_data.append({
                'Configuration': config,
                'Fold': fold,
                'R² Score': score
            })
    
    plot_df = pd.DataFrame(plot_data)
    ax = sns.barplot(x='Configuration', y='R² Score', hue='Fold', data=plot_df)
    plt.title('Cross-Validation Scores Across Folds for Top Configurations')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='CV Fold')
    plt.tight_layout()
    plt.savefig('plots/cv_score_distribution.png', dpi=300)
    mlflow.log_artifact('plots/cv_score_distribution.png')
    plt.close()
    
    # 6. Residual Analysis Plots for the best model
    plt.figure(figsize=(12, 10))
    
    residuals = y_test - y_pred
    
    # Residuals vs Predicted
    plt.subplot(2, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Residuals vs Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    
    # Histogram of residuals
    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Histogram of Residuals')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    
    # Q-Q plot
    plt.subplot(2, 2, 3)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    
    # Actual vs Predicted
    plt.subplot(2, 2, 4)
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    plt.tight_layout()
    plt.savefig('plots/residual_analysis.png', dpi=300)
    mlflow.log_artifact('plots/residual_analysis.png')
    plt.close()
    
    # 7. Learning Curves - Train vs. CV score for best configuration
    best_config_idx = results_df['rank_test_score'] == 1
    train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
    train_scores = []
    test_scores = []
    
    # Recreate the best pipeline configuration
    best_degree = results_df.loc[best_config_idx, 'param_poly__degree'].iloc[0]
    best_fit_intercept = results_df.loc[best_config_idx, 'param_model__fit_intercept'].iloc[0]
    best_positive = results_df.loc[best_config_idx, 'param_model__positive'].iloc[0]
    
    best_pipeline = Pipeline(
        steps=[       
            ('features_to_cubic', features_to_cubic),
            ('features_to_square', features_to_square),
            ('combine_sym', feature_combiner),
            ('columndrop', column_dropper),
            ('normalize', StandardScaler()),
            ('poly', PolynomialFeatures(degree=best_degree, include_bias=False)),
            ('model', LinearRegression(fit_intercept=best_fit_intercept, positive=best_positive))
        ]
    )
    
    from sklearn.model_selection import learning_curve
    
    train_sizes_abs, train_scores, test_scores = learning_curve(
        best_pipeline, x_train, y_train, train_sizes=train_sizes, cv=kf, scoring='r2'
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='blue', label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='green', label='Cross-validation score')
    plt.title(f'Learning Curves for Best Model (Degree={best_degree}, {"with" if best_fit_intercept else "without"} intercept)')
    plt.xlabel('Training Set Size (Fraction)')
    plt.ylabel('R² Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/learning_curve.png', dpi=300)
    mlflow.log_artifact('plots/learning_curve.png')
    plt.close()
    
    # Extract and plot feature importances (coefficients) from best model
    # We need to transform the data first using the pipeline steps before the model
    preprocessing_pipeline = Pipeline(
        steps=[component for component in best_model.steps if component[0] != 'model']
    )
    X_processed = preprocessing_pipeline.transform(x_train)
    
    # Get the coefficients from the model
    coefficients = best_model.named_steps['model'].coef_
    
    # Create feature names - this is tricky because of the transformations
    # For simplicity, we'll just use feature indices
    feature_indices = range(len(coefficients))
    
    # Sort coefficients by absolute value to find most important
    sorted_idx = np.argsort(np.abs(coefficients))[::-1]
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    plt.bar(range(min(20, len(sorted_idx))), coefficients[sorted_idx[:20]])
    plt.xticks(range(min(20, len(sorted_idx))), sorted_idx[:20], rotation=45)
    plt.title('Top Feature Coefficients by Magnitude')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=300)
    mlflow.log_artifact('plots/feature_importance.png')
    plt.close()
    
    # Log metrics
    mlflow.log_metrics({
        "r2": r2,
        "mse": mse,
        "cv_best_score": grid_search.best_score_
    })
    
    # Log best parameters
    mlflow.log_params(grid_search.best_params_)
    
    # Log model
    mlflow.sklearn.log_model(best_model, "linear_regression_model")
    
    # Register model
    model_info = mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="linear_regression_model"
    )
    
    mlflow.register_model(
        model_uri=model_info.model_uri,
        name="linear_regression_model_with_viz"
    )
    
    print("MLflow run with visualizations completed successfully!")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print("Plots saved to 'plots' directory and logged to MLflow")