import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

# ======================
# MLflow Configuration
# ======================
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("svm_grid_search_experiment")

# ======================
# Adjustable Parameters
# ======================
# Parameter grid for GridSearchCV; note the 'clf__' prefix refers to the SVC step in the pipeline.
SVM_GRID_PARAMS = {
    'clf__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'clf__C': [0.1, 1, 10, 100],
    'clf__gamma': [1, 0.1, 0.01, 0.001]
}
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS = 5  # For Stratified K-Fold cross-validation

with mlflow.start_run():
    
    mlflow.set_tag("mlflow.runName", "svm_grid_search_experiment")
    
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

    # ======================
    # Balance Classes by Oversampling
    # ======================
    max_count = df['Weakest'].value_counts().max()
    df_list = []
    for label, group in df.groupby('Weakest'):
        group_oversampled = group.sample(max_count, replace=True, random_state=RANDOM_STATE)
        df_list.append(group_oversampled)
    df_balanced = pd.concat(df_list).reset_index(drop=True)

    # ======================
    # Train/Test Split
    # ======================
    X = df_balanced.drop(columns=["Weakest"])
    y = df_balanced["Weakest"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ======================
    # Pipeline Construction
    # ======================
    # Scaling is important for SVM performance.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC())
    ])

    # ======================
    # Grid Search with Cross-Validation
    # ======================
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=SVM_GRID_PARAMS,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1  # Use all available cores
    )

    # Log basic parameters to MLflow
    mlflow.log_params({
        "model_type": "SVC with GridSearchCV",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS
    })

    # ======================
    # Run Grid Search Experiment
    # ======================
    grid_search.fit(X_train, y_train)

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Log evaluation metrics
    mlflow.log_metrics({
        "accuracy": accuracy,
        "error_rate": error_rate,
        "weighted_precision": report_dict['weighted avg']['precision'],
        "weighted_recall": report_dict['weighted avg']['recall'],
        "weighted_f1": report_dict['weighted avg']['f1-score'],
        "cv_best_score": grid_search.best_score_
    })

    # Log the best hyperparameters from Grid Search
    mlflow.log_params(grid_search.best_params_)

    # Log the best model with MLflow
    mlflow.sklearn.log_model(best_model, "svm_model")

    print("MLflow run completed successfully!")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
