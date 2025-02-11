import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import log_loss

# ======================
# MLflow Configuration
# ======================
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("logistic_regression_experiment")

# ======================
# Adjustable parameters
# ======================
LR_PARAMS = {
    'C': 0.5,                # Inverse of regularization strength
    'penalty': 'l1',         # or 'l1', 'elasticnet' (with solver='saga'), etc.
    'solver': 'liblinear',       # or 'liblinear', 'saga' for L1 regularization
    'max_iter': 1000         # increase if you get convergence warnings
}
TEST_SIZE = 0.2
RANDOM_STATE = 42

with mlflow.start_run():
    
    mlflow.set_tag("mlflow.runName", "logistic_regression_C=0.5_liblinear_l1")
    
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

    # ======================
    # Pipeline Construction
    # ======================
    # We'll often scale features for logistic regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(**LR_PARAMS))
    ])

    # ======================
    # Log Parameters
    # ======================
    mlflow.log_params({
        "model_type": "LogisticRegression",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        **LR_PARAMS
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

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)


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
    mlflow.sklearn.log_model(pipeline, "logistic_model")

    model_name = "logistic_weaklink_classifier"
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/logistic_model"
    mlflow.register_model(model_uri, model_name)

    # ======================
    # Save Model Locally
    # ======================
    joblib.dump(pipeline, "ML/saved models/logistic_weaklink_classifier.pkl")

    print("MLflow run completed successfully!")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
