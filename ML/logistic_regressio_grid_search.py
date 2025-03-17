import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

# ======================
# MLflow Configuration
# ======================
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("logistic_regression_grid_search")

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# ======================
# Create directory for plots
# ======================
import os
os.makedirs("plots", exist_ok=True)

with mlflow.start_run():
    
    mlflow.set_tag("mlflow.runName", "logistic_regression_grid_search")
    
    # ======================
    # Data Preparation
    # ======================
    feature_df = pd.read_excel("ML/data/classifier/AimoScore_WeakLink_big_scores.xls")
    weaklink_df = pd.read_excel("ML/data/classifier/20190108 scores_and_weak_links.xlsx")

    # Identify the 'Weakest' link for each data point
    weakest = weaklink_df.iloc[:, 3:].idxmax(axis=1)
    weaklink_df["Weakest"] = weakest
    weaklink_df = weaklink_df[["ID", "Weakest"]]

    # Drop unneeded columns from feature DataFrame
    feature_df.drop(
        columns=["AimoScore", "No_1_Time_Deviation", "No_2_Time_Deviation", "EstimatedScore"],
        inplace=True, errors='ignore'
    )

    # Merge feature DataFrame with 'Weakest' info
    df = feature_df.merge(weaklink_df, on="ID", how="inner")
    df.drop(columns=["ID"], inplace=True)
    
    # Plot class distribution before balancing
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(x='Weakest', data=df)
    plt.title('Class Distribution Before Balancing')
    plt.xlabel('Weakest Link Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/class_distribution_before.png', dpi=300)
    mlflow.log_artifact('plots/class_distribution_before.png')
    plt.close()

    # ======================
    # Balance Classes by Oversampling
    # ======================
    max_count = df['Weakest'].value_counts().max()
    df_list = []
    for label, group in df.groupby('Weakest'):
        group_oversampled = group.sample(max_count, replace=True, random_state=42)
        df_list.append(group_oversampled)
    df_balanced = pd.concat(df_list).reset_index(drop=True)
    
    # Plot class distribution after balancing
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(x='Weakest', data=df_balanced)
    plt.title('Class Distribution After Balancing')
    plt.xlabel('Weakest Link Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/class_distribution_after.png', dpi=300)
    mlflow.log_artifact('plots/class_distribution_after.png')
    plt.close()

    # ======================
    # Train/Test Split
    # ======================
    X = df_balanced.drop(columns=["Weakest"])
    y = df_balanced["Weakest"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ======================
    # Pipeline Construction
    # ======================
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=5000, random_state=42))
    ])

    # ======================
    # Define Parameters and Run Individual Grid Searches
    # ======================
    # Define possible values for each parameter
    C_values = [0.01, 0.1, 0.5, 1.0, 10.0]
    penalty_values = ['l1', 'l2', 'elasticnet', None]  # Changed 'none' to None
    solver_values = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    
    # Log parameters to MLflow
    mlflow.log_params({
        "model_type": "LogisticRegression with GridSearchCV",
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 5,
        "C_values": str(C_values),
        "penalty_values": str(['l1', 'l2', 'elasticnet', 'None']),  # Fixed representation for logging
        "solver_values": str(solver_values)
    })
    
    # Store all results
    all_results = []
    best_score = -1
    best_params = None
    best_model = None
    
    # Cross-validation setup
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform grid search manually to avoid parameter combination issues
    for penalty in penalty_values:
        for solver in solver_values:
            # Skip invalid parameter combinations
            if penalty == 'elasticnet' and solver != 'saga':
                continue
            if penalty == 'l1' and solver not in ['liblinear', 'saga']:
                continue
            if penalty is None and solver in ['liblinear']:  # Updated for None
                continue
            
            for C in C_values:
                # Set parameters
                if penalty == 'elasticnet':
                    # Elasticnet needs l1_ratio
                    for l1_ratio in [0.25, 0.5, 0.75]:
                        # Create model with specific parameters
                        pipeline.set_params(
                            clf__C=C,
                            clf__penalty=penalty,
                            clf__solver=solver,
                            clf__l1_ratio=l1_ratio
                        )
                        
                        # Perform cross-validation
                        scores = []
                        for train_idx, val_idx in kf.split(X_train, y_train):
                            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
                            
                            pipeline.fit(X_train_fold, y_train_fold)
                            y_pred_fold = pipeline.predict(X_val_fold)
                            score = accuracy_score(y_val_fold, y_pred_fold)
                            scores.append(score)
                        
                        mean_score = np.mean(scores)
                        std_score = np.std(scores)
                        
                        # Store results
                        result = {
                            'params': {
                                'C': C,
                                'penalty': penalty,
                                'solver': solver,
                                'l1_ratio': l1_ratio
                            },
                            'mean_score': mean_score,
                            'std_score': std_score
                        }
                        all_results.append(result)
                        
                        # Check if this is the best model so far
                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = result['params']
                            # Train on full training set
                            pipeline.fit(X_train, y_train)
                            best_model = pipeline
                            
                        print(f"C={C}, penalty={penalty}, solver={solver}, l1_ratio={l1_ratio}: {mean_score:.4f} (±{std_score:.4f})")
                else:
                    # Create model with specific parameters
                    pipeline.set_params(
                        clf__C=C,
                        clf__penalty=penalty,
                        clf__solver=solver
                    )
                    
                    # Perform cross-validation
                    scores = []
                    for train_idx, val_idx in kf.split(X_train, y_train):
                        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
                        
                        pipeline.fit(X_train_fold, y_train_fold)
                        y_pred_fold = pipeline.predict(X_val_fold)
                        score = accuracy_score(y_val_fold, y_pred_fold)
                        scores.append(score)
                    
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    
                    # Store results
                    result = {
                        'params': {
                            'C': C,
                            'penalty': penalty,
                            'solver': solver
                        },
                        'mean_score': mean_score,
                        'std_score': std_score
                    }
                    all_results.append(result)
                    
                    # Check if this is the best model so far
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = result['params']
                        # Train on full training set
                        pipeline.fit(X_train, y_train)
                        best_model = pipeline
                        
                    print(f"C={C}, penalty={penalty}, solver={solver}: {mean_score:.4f} (±{std_score:.4f})")
    
    # Evaluate best model on test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy
    
    print("\n===== Results =====")
    print(f"Best parameters: {best_params}")
    print(f"Best CV score: {best_score:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Convert results to DataFrame for easier visualization
    results_df = pd.DataFrame(all_results)
    
    # Convert parameters dict to individual columns
    param_columns = pd.json_normalize(results_df['params'])
    results_df = pd.concat([results_df.drop('params', axis=1), param_columns], axis=1)
    
    # Add string representation for easier plotting
    if 'l1_ratio' in results_df.columns:
        results_df['param_string'] = results_df.apply(
            lambda x: f"C={x['C']}, penalty={x['penalty']}, solver={x['solver']}" + 
                     (f", l1_ratio={x['l1_ratio']}" if 'l1_ratio' in x and x['penalty'] == 'elasticnet' else ""),
            axis=1
        )
    else:
        results_df['param_string'] = results_df.apply(
            lambda x: f"C={x['C']}, penalty={x['penalty']}, solver={x['solver']}",
            axis=1
        )
    
    # Sort by performance
    results_df = results_df.sort_values('mean_score', ascending=False)
    
    # ======================
    # Generate Visualizations
    # ======================
    
    # 1. Top Models Comparison
    plt.figure(figsize=(16, 8))
    top_results = results_df.head(15)  # Top 15 configurations
    sns.barplot(x='param_string', y='mean_score', data=top_results)
    plt.title('Top 15 Model Configurations by Accuracy')
    plt.xticks(rotation=90)
    plt.ylabel('Mean Accuracy (Cross-Validation)')
    plt.tight_layout()
    plt.savefig('plots/top_models_comparison.png', dpi=300)
    mlflow.log_artifact('plots/top_models_comparison.png')
    plt.close()
    
    # 2. Regularization Strength (C) Impact
    plt.figure(figsize=(12, 6))
    C_impact = results_df.groupby('C')['mean_score'].mean().reset_index()
    sns.lineplot(x='C', y='mean_score', data=C_impact, marker='o')
    plt.title('Impact of Regularization Strength (C)')
    plt.xscale('log')
    plt.xlabel('C value (log scale)')
    plt.ylabel('Mean Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/C_parameter_impact.png', dpi=300)
    mlflow.log_artifact('plots/C_parameter_impact.png')
    plt.close()
    
    # 3. Penalty Type Impact
    plt.figure(figsize=(10, 6))
    penalty_impact = results_df.groupby('penalty')['mean_score'].mean().reset_index()
    sns.barplot(x='penalty', y='mean_score', data=penalty_impact)
    plt.title('Impact of Penalty Type')
    plt.xlabel('Penalty Type')
    plt.ylabel('Mean Accuracy')
    plt.tight_layout()
    plt.savefig('plots/penalty_impact.png', dpi=300)
    mlflow.log_artifact('plots/penalty_impact.png')
    plt.close()
    
    # 4. Solver Impact
    plt.figure(figsize=(10, 6))
    solver_impact = results_df.groupby('solver')['mean_score'].mean().reset_index()
    sns.barplot(x='solver', y='mean_score', data=solver_impact)
    plt.title('Impact of Solver Algorithm')
    plt.xlabel('Solver')
    plt.ylabel('Mean Accuracy')
    plt.tight_layout()
    plt.savefig('plots/solver_impact.png', dpi=300)
    mlflow.log_artifact('plots/solver_impact.png')
    plt.close()
    
    # 5. Confusion Matrix for Best Model
    plt.figure(figsize=(14, 12))
    cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=best_model.classes_, yticklabels=best_model.classes_)
    plt.title('Normalized Confusion Matrix - Best Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=300)
    mlflow.log_artifact('plots/confusion_matrix.png')
    plt.close()
    
    # 6. Top Feature Importance (from best model)
    logreg_model = best_model.named_steps['clf']
    feature_names = X.columns
    
    # Only for models that have coef_ attribute
    if hasattr(logreg_model, 'coef_'):
        # For multi-class, get average absolute coefficient values
        coefs = np.abs(logreg_model.coef_).mean(axis=0)
        
        # Create a DataFrame with feature names and their importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': coefs
        }).sort_values('Importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title('Top 20 Feature Importance (Average Absolute Coefficient Value)')
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png', dpi=300)
        mlflow.log_artifact('plots/feature_importance.png')
        plt.close()
    
    # ======================
    # Log Evaluation Metrics
    # ======================
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    mlflow.log_metrics({
        "accuracy": accuracy,
        "error_rate": error_rate,
        "weighted_precision": report_dict['weighted avg']['precision'],
        "weighted_recall": report_dict['weighted avg']['recall'],
        "weighted_f1": report_dict['weighted avg']['f1-score'],
        "cv_best_score": best_score
    })

    # Log the best hyperparameters
    for param, value in best_params.items():
        if isinstance(value, (int, float, str, bool)):
            mlflow.log_param(f"best_{param}", value)

    # ======================
    # Log and Register Model
    # ======================
    mlflow.sklearn.log_model(best_model, "logistic_model")
    
    model_name = "logistic_weaklink_classifier_grid_search"
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/logistic_model"
    mlflow.register_model(model_uri, model_name)

    print("MLflow run completed successfully!")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print("Plots saved to 'plots' directory and logged to MLflow")