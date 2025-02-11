import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn
import custom_transformers
from sklearn.preprocessing import FunctionTransformer





# ======================
# MLflow Configuration
# ======================
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("knn_classification_refined")

# ======================
# Adjustable parameters
# ======================
KNN_PARAMS = {
    'n_neighbors': 4,       # Number of neighbors
    'weights': 'uniform',   # 'uniform' or 'distance'
    'p': 2                  # Metric: 1=Manhattan, 2=Euclidean
}
TEST_SIZE = 0.2
RANDOM_STATE = 42

with mlflow.start_run():
    
    mlflow.set_tag("mlflow.runName", "without scaling")
    
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
    feature_df.drop(columns=["AimoScore", "No_1_Time_Deviation", 
                             "No_2_Time_Deviation", "EstimatedScore"], 
                    inplace=True, errors='ignore')

    # Merge feature DataFrame with 'Weakest' info
    df = feature_df.merge(weaklink_df, on="ID", how="inner")
    df.drop(columns=["ID"], inplace=True)

    # ======================
    # Handle Rare Classes
    # ======================
    # Remove classes with fewer than 2 instances
    # so stratified splitting won't fail
    counts = df['Weakest'].value_counts()
    rare_classes = counts[counts < 2].index
    df = df[~df['Weakest'].isin(rare_classes)]

    # ======================
    # Train/Test Split
    # ======================
    X = df.drop(columns=["Weakest"])
    y = df["Weakest"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


    pipeline = Pipeline([
        ('clf', KNeighborsClassifier(**KNN_PARAMS))
    ])

    # ======================
    # Log Parameters
    # ======================
    mlflow.log_params({
        "model_type": "KNeighborsClassifier",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        **KNN_PARAMS
    })

    # ======================
    # Model Training
    # ======================
    pipeline.fit(X_train, y_train)

    # ======================
    # Evaluation & Logging
    # ======================
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy

    # classification_report with output_dict for metric extraction
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Log main metrics
    mlflow.log_metrics({
        "accuracy": accuracy,
        "error_rate": error_rate,
        "weighted_precision": report_dict['weighted avg']['precision'],
        "weighted_recall": report_dict['weighted avg']['recall'],
        "weighted_f1": report_dict['weighted avg']['f1-score']
    })

    # ======================
    # Log & Register Model
    # ======================
    mlflow.sklearn.log_model(pipeline, "knn_model")

    model_name = "knn_weaklink_classifier"
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/knn_model"
    mlflow.register_model(model_uri, model_name)

    # ======================
    # Save Model Locally
    # ======================
    joblib.dump(pipeline, "ML/saved models/knn_weaklink_classifier.pkl")

    print("MLflow run completed successfully!")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
