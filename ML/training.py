import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
import custom_transformers
#initiate mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("linear_regression")

df = pd.read_excel("ML/data/AimoScore_WeakLink_big_scores.xls")


x = df.iloc[:,1:-1]
y = df.iloc[:,0]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

feature_weights = {
    7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2,  # FSM important features
    16: 2, 17: 2,22:2, 23: 4, 24: 4,25: 2,26: 2,27: 2, 28: 2, 29: 2, 29: 2, 33: 2, 34: 2, 35: 2, 36: 2,  # NASM key features
}


symmetricalColumns = [
    (3, 5), (4, 6), (7, 10), (8, 11), (9, 12),  # FMS Symmetry
    (13, 14), (16, 17), (20, 21), (23, 24)  # NASM symmetry
]





irrelevant_columns = [27, 1, 25, 31, 3, 4, 7, 8, 9]

column_dropper = ColumnTransformer(
    [
        ('columns_to_drop', 'drop', irrelevant_columns)
    ],
    remainder="passthrough"
)

feature_combiner = custom_transformers.CombineCorrelatedFeatures(symmetricalColumns)

pipeline = Pipeline(
    steps=[        
        #('feature_weights', FeatureWeights(feature_weights)),
        ('combine_sym', feature_combiner),
        ('columndrop', column_dropper),
        ('normalize', StandardScaler()),
        ('model', LinearRegression())
    ]
)

with mlflow.start_run():
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    
    mlflow.sklearn.log_model(pipeline, "linear_regression_model_v5")
    
    
    model_name = "linear_regression_model_v5"
    
    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path=model_name,
                                        )
    
    mlflow.register_model(
        model_uri=model_info.model_uri,
        name=model_name
    )

    print("Mean Squared Error: ", mse)
    print("R2 Score: ", r2)
    
    # Save the model
    joblib.dump(pipeline, "ML/saved models/linear_regression_model_v5.pkl")
    print("Model saved successfully!")











