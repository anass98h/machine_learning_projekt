import pandas as pd
import sys
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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
# Adjustable parameters test test
# ======================
LR_PARAMS = {
    'C': 0.5,                # Inverse of regularization strength
    'penalty': 'l2',         # or 'l1', 'elasticnet' (with solver='saga'), etc.
    'solver': 'lbfgs',       # or 'liblinear', 'saga' for L1 regularization
    'max_iter': 1000         # increase if you get convergence warnings
}
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS = 5  # For cross-validation

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
    # Balance Classes by Oversampling
    # ======================
    # Instead of dropping rare classes, we duplicate (oversample) undersampled ones.
    # We first determine the maximum count among the classes.
    max_count = df['Weakest'].value_counts().max()
    df_list = []
    for label, group in df.groupby('Weakest'):
        # Oversample this class to match max_count using replacement.
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
    # We'll often scale features for logistic regression.
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
        "n_splits": N_SPLITS,
        **LR_PARAMS
    })
    
    # ======================
    # 1) Cross-Validation on TRAIN
    # ======================
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    cv_scores = cross_val_score(
        pipeline, 
        X_train, 
        y_train, 
        cv=skf, 
        scoring='accuracy'
    )

    cv_mean_accuracy = cv_scores.mean()
    cv_std_accuracy = cv_scores.std()

    mlflow.log_metric("cv_mean_accuracy", cv_mean_accuracy)
    mlflow.log_metric("cv_std_accuracy", cv_std_accuracy)
    
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

    print("MLflow run completed successfully!")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
