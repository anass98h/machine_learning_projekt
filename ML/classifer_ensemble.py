import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (VotingClassifier, StackingClassifier,
                              GradientBoostingClassifier, BaggingClassifier,
                              RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

# ----------------------
# MLflow Configuration
# ----------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("ensemble_classifier_experiment")

# ----------------------
# Adjustable Parameters
# ----------------------
KNN_PARAMS = {
    'n_neighbors': 4,       # Number of neighbors
    'weights': 'uniform',   # 'uniform' or 'distance'
    'p': 2                  # Metric: 1=Manhattan, 2=Euclidean
}
LR_PARAMS = {
    'C': 0.5,                # Inverse of regularization strength
    'penalty': 'l1',         # Regularization type: 'l1'
    'solver': 'liblinear',   # Suitable solver for L1 penalty
    'max_iter': 1000         # Increase if convergence warnings occur
}
BOOSTED_TREE_PARAMS = {
    'n_estimators': 10,
    'random_state': 42
}
BAGGING_TREE_PARAMS = {
    'estimator': DecisionTreeClassifier(random_state=42),
    'n_estimators': 10,
    'random_state': 42
}
RF_PARAMS = {
    'n_estimators': 10,
    'random_state': 42
}
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS = 5  # For cross-validation

# ----------------------
# Data Preparation
# ----------------------
# Read feature and weak link files
feature_df = pd.read_excel("ML/data/classifier/AimoScore_WeakLink_big_scores.xls")
weaklink_df = pd.read_excel("ML/data/classifier/20190108 scores_and_weak_links.xlsx")

# Identify the 'Weakest' link per row (using the column with maximum score)
weaklink_df["Weakest"] = weaklink_df.iloc[:, 3:].idxmax(axis=1)
weaklink_df = weaklink_df[["ID", "Weakest"]]

# Remove unneeded columns from the feature DataFrame
feature_df.drop(columns=["AimoScore", "No_1_Time_Deviation", 
                          "No_2_Time_Deviation", "EstimatedScore"], 
                inplace=True, errors='ignore')

# Merge datasets on "ID" and drop the ID column
df = feature_df.merge(weaklink_df, on="ID", how="inner")
df.drop(columns=["ID"], inplace=True)

# ----------------------
# Balance Classes by Oversampling
# ----------------------
# Instead of dropping rare classes, duplicate (oversample) the underrepresented classes.
max_count = df['Weakest'].value_counts().max()
df_list = []
for label, group in df.groupby('Weakest'):
    group_oversampled = group.sample(max_count, replace=True, random_state=RANDOM_STATE)
    df_list.append(group_oversampled)
df_balanced = pd.concat(df_list).reset_index(drop=True)

# Define predictors and response using the balanced dataset
X = df_balanced.drop(columns=["Weakest"])
y = df_balanced["Weakest"]

# Train/Test Split (with stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ----------------------
# Define Base Pipelines (Base Models)
# ----------------------
# Pipeline for KNN classifier (with scaling)
pipeline_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(**KNN_PARAMS))
])

# Pipeline for Logistic Regression (with scaling)
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(**LR_PARAMS))
])

# Pipeline for Gradient Boosting (with scaling)
pipeline_boosted_tree = Pipeline([
    ('scaler', StandardScaler()),
    ('boosted_dt', GradientBoostingClassifier(**BOOSTED_TREE_PARAMS))
])

# Pipeline for Bagging Decision Trees (with scaling)
pipeline_bagging_tree = Pipeline([
    ('scaler', StandardScaler()),
    ('bagged_dt', BaggingClassifier(**BAGGING_TREE_PARAMS))
])

# Pipeline for Random Forest (with scaling)
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(**RF_PARAMS))
])

# ----------------------
# Define the Stacking Ensemble with SVM as the Final Estimator (Endpoint)
# ----------------------
# The base estimators are the ones defined above.
base_estimators = [
    ('knn', pipeline_knn),
    ('lr', pipeline_lr),
    ('boosted_dt', pipeline_boosted_tree),
    ('bagged_dt', pipeline_bagging_tree),
    ('rf', pipeline_rf)
]

# The final estimator (endpoint) is an SVM. We wrap it in a pipeline to include scaling.
final_estimator = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

# Construct the stacking classifier with the SVM as the final estimator.
stacking_clf_endpoint = StackingClassifier(
    estimators=base_estimators,
    final_estimator=final_estimator,
    cv=N_SPLITS,
    n_jobs=-1
)

# ----------------------
# Grid Search on the SVM Endpoint
# ----------------------
# Define a parameter grid for tuning the SVM used as the final estimator.
svm_endpoint_grid = {
    'final_estimator__svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'final_estimator__svm__C': [0.1, 1, 10, 100],
    'final_estimator__svm__gamma': [1, 0.1, 0.01, 0.001]
}

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
grid_search_endpoint = GridSearchCV(
    estimator=stacking_clf_endpoint,
    param_grid=svm_endpoint_grid,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1
)

# ----------------------
# MLflow Run: Training & Evaluation
# ----------------------
with mlflow.start_run():
    mlflow.set_tag("mlflow.runName", "ensemble_with_svm_endpoint")
    
    # Log ensemble-level parameters
    mlflow.log_params({
        "model_type": "Ensemble Stacking with SVM Endpoint",
        "base_models": ["KNN", "LogisticRegression", "BoostedTrees", "BaggedTrees", "RandomForest"],
        "final_estimator": "SVM",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        **KNN_PARAMS,
        **LR_PARAMS,
        **BOOSTED_TREE_PARAMS,
        **BAGGING_TREE_PARAMS,
        **RF_PARAMS
    })
    
    # ----------------------
    # Optimize the SVM endpoint via grid search
    # ----------------------
    grid_search_endpoint.fit(X_train, y_train)
    best_stacking_clf = grid_search_endpoint.best_estimator_
    mlflow.log_params(grid_search_endpoint.best_params_)  # Log best SVM endpoint parameters
    
    # Cross-validate the best stacking classifier on training data
    stacking_cv_scores = cross_val_score(best_stacking_clf, X_train, y_train, cv=skf, scoring='accuracy')
    
    # Fit the optimized stacking ensemble on the training data
    best_stacking_clf.fit(X_train, y_train)
    y_pred = best_stacking_clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - test_accuracy
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Log metrics
    mlflow.log_metric("stacking_cv_mean_accuracy", stacking_cv_scores.mean())
    mlflow.log_metric("stacking_cv_std_accuracy", stacking_cv_scores.std())
    mlflow.log_metric("stacking_test_accuracy", test_accuracy)
    
    # Log additional metrics from the classification report
    mlflow.log_metric("weighted_precision", report_dict['weighted avg']['precision'])
    mlflow.log_metric("weighted_recall", report_dict['weighted avg']['recall'])
    mlflow.log_metric("weighted_f1", report_dict['weighted avg']['f1-score'])
    
    # Print classification report for inspection
    print("Stacking Ensemble with SVM Endpoint Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    
    # Log the final stacking model in MLflow
    mlflow.sklearn.log_model(best_stacking_clf, "stacking_ensemble_svm_endpoint_model")
    
    print("MLflow run completed successfully!")
    print(f"Stacking Test Accuracy (with SVM Endpoint): {test_accuracy}")
