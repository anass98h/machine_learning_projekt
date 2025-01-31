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


df = pd.read_excel("ML/data/AimoScore_WeakLink_big_scores.xls")


x = df.iloc[:,1:-1]
y = df.iloc[:,0]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

feature_weights = {
    7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2,  # FSM important features
    16: 2, 17: 2,22:2, 23: 4, 24: 4,25: 2,26: 2,27: 2, 28: 2, 29: 2, 29: 2, 33: 2, 34: 2, 35: 2, 36: 2,  # NASM key features
}


correlatedColumns = [
    (13, 14), (16, 17), (20, 21), (23, 24)  # NASM symmetry
]

class CombineCorrelatedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, correlatedColumns):
        self.correlatedColumns = correlatedColumns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_combined = pd.DataFrame(X.copy(), columns=X.columns)
        print(X_combined.head())
        
        drop_columns = ["No_15_NASM_Deviation", "No_2_Angle_Deviation", "No_13_NASM_Deviation", "No_19_NASM_Deviation", "No_4_Angle_Deviation", "No_5_Angle_Deviation", "No_9_Angle_Deviation", "No_10_Angle_Deviation" ]

        for col1_idx, col2_idx in self.correlatedColumns:
            if col1_idx < X_combined.shape[1] and col2_idx  < X_combined.shape[1]:
                col1_name = X_combined.columns[col1_idx]
                col2_name = X_combined.columns[col2_idx]
                
                new_col_name = f"combined_{col1_name}_{col2_name}"
                X_combined[new_col_name] = (X_combined[col1_name]+X_combined[col2_name]) /2

                drop_columns.extend([col1_name, col2_name])
                
            else:
                print(f"Columns {X_combined.iloc[:, col1_idx]} and {X_combined.iloc[:, col2_idx]} not found in the dataset")

                
        print(f"Columns {drop_columns} will be dropped")
        X_combined.drop(columns=drop_columns, inplace=True)
        print(X_combined.head())
        return X_combined


class FeatureWeights(BaseEstimator, TransformerMixin):
    def __init__(self, feature_weights):
        self.feature_weights = feature_weights

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        xWeighted = X.copy()
        for colIdx, weight in self.feature_weights.items():
            if colIdx < len(xWeighted.columns):
                colName = xWeighted.columns[colIdx]
                xWeighted[colName] *= weight
                #print(f"Column {colName} weighted by {weight}")
            else:
                print(f"Column {colIdx} not found in the dataset")

        return xWeighted

irrelevant_columns = [27, 1, 25, 31, 3, 4, 7, 8, 9]

column_dropper = ColumnTransformer(
    [
        ('columns_to_drop', 'drop', irrelevant_columns)
    ],
    remainder="passthrough"
)

pipeline = Pipeline(
    steps=[        
        ('feature_weights', FeatureWeights(feature_weights)),
        ('combine_corr', CombineCorrelatedFeatures(correlatedColumns)),
        #('columndrop', column_dropper),
        ('model', LinearRegression())
    ]
)

pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error: ", mse)
print("R2 Score: ", r2)

#plt.scatter(y_test, y_pred,alpha=0.5)
#plt.xlabel("real value y_test")
#plt.ylabel("predicted value y_pred")
#plt.title("Linear Regression Model")
#plt.show()

# Save the model
joblib.dump(pipeline, "ML/saved models/linear_regression_model_v4.pkl")
print("Model saved successfully!")