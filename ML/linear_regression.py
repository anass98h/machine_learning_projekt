import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_validate
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import mlflow
import mlflow.sklearn
import custom_transformers
from lib.data_loader import load_data

#initiate mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("linear_regression")

df = load_data()


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

column_dropper = ColumnTransformer(
    [
        ('columns_to_drop', 'drop', irrelevant_columns)
    ],
    remainder="passthrough"
)

feature_combiner = custom_transformers.CombineCorrelatedFeatures(symmetricalColumns)
symmetrical_Columns = custom_transformers.symmetricalColumns(symmetricalColumns)
feature_weights = custom_transformers.FeatureWeights(feature_weights)
features_to_square = custom_transformers.SquareFeatures(columns=features_to_square, replace=False) 
features_to_cubic = custom_transformers.CubicFeatures(columns=features_to_cubic, replace=False)

pipeline = Pipeline(
    steps=[       
        #('feature_weights', feature_weights), 
        #('symmetrical_columns', symmetrical_Columns),
        ('features_to_cubic', features_to_cubic),
        ('features_to_square', features_to_square),
        ('combine_sym', feature_combiner),
        ('columndrop', column_dropper),
        ('normalize', StandardScaler()),
        ('model', LinearRegression())
    ]
)
corossvlidat_if_1 = 0
if corossvlidat_if_1 == 1 :
    # Set up 5-fold cross validation (as described in your lecture slides)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # Define scoring: note that sklearn's "neg_mean_squared_error" returns negative values
    scoring = {'mse': 'neg_mean_squared_error', 'r2': 'r2'}

    with mlflow.start_run():
        # Perform cross-validation on the entire dataset
        cv_results = cross_validate(pipeline, x, y, cv=cv, scoring=scoring)
        # Compute mean metrics; reverse sign for MSE
        mean_mse = -np.mean(cv_results['test_mse'])
        mean_r2 = np.mean(cv_results['test_r2'])
        
        # Log parameters and cross-validated metrics in mlflow
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("mse", mean_mse)
        mlflow.log_metric("r2", mean_r2)
        
        # Fit the final model on the entire dataset and log it in mlflow
        pipeline.fit(x, y)
        mlflow.sklearn.log_model(pipeline, "linear_regression_model_v5")
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="linear_regression_model_v5"
        )
        mlflow.register_model(
            model_uri=model_info.model_uri,
            name="linear_regression_model_v5"
        )
        
        print("Cross-Validated Mean Squared Error: ", mean_mse)
        print("Cross-Validated R2 Score: ", mean_r2)

else:
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
        

