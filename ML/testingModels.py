import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set up MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")  # Ensure MLflow server is running
mlflow.set_experiment("weak_link_classification")

# Load datasets
feature_df = pd.read_excel("ML/data/classifier/AimoScore_WeakLink_big_scores.xls")
weaklink_df = pd.read_excel("ML/data/classifier/20190108 scores_and_weak_links.xlsx")

# Extract the weakest link (column with highest score)
weakest = weaklink_df.iloc[:, 3:].idxmax(axis=1)
weaklink_df["Weakest"] = weakest
weaklink_df = weaklink_df[["ID", "Weakest"]]

# Drop irrelevant columns from feature dataset
feature_df.drop(columns=["AimoScore", "No_1_Time_Deviation", "No_2_Time_Deviation", "EstimatedScore"], inplace=True)

# Merge features with weakest link labels
df = feature_df.merge(weaklink_df, on="ID", how="inner")
df.drop(columns=["ID"], inplace=True)  # Drop ID after merging

# Split features and target labels
X = df.drop(columns=["Weakest"])  # Features
y = df["Weakest"]  # Target variable (Weakest link category)

# Encode categorical target labels to numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Start MLflow run
with mlflow.start_run():
    # Train classification model (Random Forest)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", class_report)

    # Log parameters and metrics to MLflow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy)

    # Log model in MLflow
    mlflow.sklearn.log_model(clf, "random_forest_weak_link_classifier")

    # Save model, scaler, and label encoder locally
    joblib.dump(clf, "ML/saved_models/weak_link_classifier.pkl")
    joblib.dump(scaler, "ML/saved_models/scaler.pkl")
    joblib.dump(label_encoder, "ML/saved_models/label_encoder.pkl")

    print("Model and preprocessing objects saved successfully!")

print("MLflow run logged successfully!")
