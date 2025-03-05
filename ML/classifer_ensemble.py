import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, StackingClassifier, GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import joblib

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
# Instead of dropping rare classes, we duplicate (oversample) the underrepresented classes.
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
# Define Base Pipelines
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

pipeline_boosted_tree = Pipeline([
    ('scaler', StandardScaler()),
    ('boosted_dt', GradientBoostingClassifier(**BOOSTED_TREE_PARAMS))
])

pipeline_bagging_tree = Pipeline([
    ('scaler', StandardScaler()),
    ('bagged_dt', BaggingClassifier(**BAGGING_TREE_PARAMS))
])

pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(**RF_PARAMS))
])

estimators = [('knn', pipeline_knn), ('lr', pipeline_lr), ('boosted_dt', pipeline_boosted_tree), ('bagged_dt', pipeline_bagging_tree), ('rf', pipeline_rf)]

# ----------------------
# Ensemble Construction
# ----------------------
# Voting ensemble using hard majority voting
voting_clf = VotingClassifier(
    estimators=estimators,
    voting='hard'
)

# Stacking ensemble: base models plus a meta-learner (here, logistic regression)
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=N_SPLITS
)

# ----------------------
# Cross-Validation
# ----------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
voting_cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=skf, scoring='accuracy')
stacking_cv_scores = cross_val_score(stacking_clf, X_train, y_train, cv=skf, scoring='accuracy')

# ----------------------
# MLflow Run: Training & Evaluation
# ----------------------
with mlflow.start_run():
    mlflow.set_tag("mlflow.runName", "ensemble_voting_and_stacking")
    
    # Log parameters
    mlflow.log_params({
        "model_type": "Ensemble",
        "base_models": ["KNN", "LogisticRegression, BoostedTrees", "BaggedTrees", "RandomForest"],
        "ensemble_methods": "Voting and Stacking",
        "voting": "hard",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        **KNN_PARAMS,
        **LR_PARAMS,
        **BOOSTED_TREE_PARAMS,
        **BAGGING_TREE_PARAMS,
        **RF_PARAMS
    })
    
    # Fit ensemble models on the training set
    voting_clf.fit(X_train, y_train)
    stacking_clf.fit(X_train, y_train)
    
    # Evaluate on the test set
    voting_pred = voting_clf.predict(X_test)
    stacking_pred = stacking_clf.predict(X_test)
    
    voting_accuracy = accuracy_score(y_test, voting_pred)
    stacking_accuracy = accuracy_score(y_test, stacking_pred)
    
    # Log cross-validation metrics for Voting ensemble
    mlflow.log_metric("voting_cv_mean_accuracy", voting_cv_scores.mean())
    mlflow.log_metric("voting_cv_std_accuracy", voting_cv_scores.std())
    mlflow.log_metric("voting_test_accuracy", voting_accuracy)
    
    # Log cross-validation metrics for Stacking ensemble
    mlflow.log_metric("stacking_cv_mean_accuracy", stacking_cv_scores.mean())
    mlflow.log_metric("stacking_cv_std_accuracy", stacking_cv_scores.std())
    mlflow.log_metric("stacking_test_accuracy", stacking_accuracy)
    
    # Log classification reports (could also be saved as artifacts)
    voting_report = classification_report(y_test, voting_pred, zero_division=0)
    stacking_report = classification_report(y_test, stacking_pred, zero_division=0)
    print("Voting Classifier Report:\n", voting_report)
    print("Stacking Classifier Report:\n", stacking_report)
    
    # Log the ensemble models in MLflow
    mlflow.sklearn.log_model(voting_clf, "voting_ensemble_model")
    mlflow.sklearn.log_model(stacking_clf, "stacking_ensemble_model")
    
    print("MLflow run completed successfully!")
    print(f"Voting Test Accuracy: {voting_accuracy}")
    print(f"Stacking Test Accuracy: {stacking_accuracy}")
