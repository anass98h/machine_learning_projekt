import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import joblib
import numpy as np


df = pd.read_excel("data/AimoScore_WeakLink_big_scores.xls")


x = df.iloc[:,1:-1]
y = df.iloc[:,0]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



class CombineCorrelatedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, correlatedColumns):
        self.correlatedColumns = correlatedColumns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_weighted = X
        
        drop_columns = []

        for col1_idx, col2_idx in self.correlatedColumns:
            if col1_idx < X_combined.shape[1] and col2_idx  < X_combined.shape[1]:
                col1_name = X_combined.columns[col1_idx]
                col2_name = X_combined.columns[col2_idx]
                
                new_col_name = f"combined_{col1_name}_{col2_name}"
                X_combined[new_col_name] = (X_combined[col1_name]+X_combined[col2_name]) /2

                drop_columns.extend([col1_name, col2_name])
                
            else:
                print(f"Columns {X_combined.iloc[:, col1_idx]} and {X_combined.iloc[:, col2_idx]} not found in the dataset")

                
                
        X_combined.drop(columns=drop_columns, inplace=True)
        return X_combined


feature_weights = {
    8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2,  # FSM important features
    17: 2, 18: 2, 24: 4, 25: 4, 26: 2, 27: 2,  # NASM key features
}

for col, weight in feature_weights.items():
    if col in x_train.columns:
        x_train[col]*= weight
        x_test[col]*=  weight

correlatedColumns = [
    (4, 6), (5, 7), (8, 11), (9, 12), (10, 13),  # FSM symmetry
    (14, 15), (17, 18), (21, 22), (24, 25), (26, 27), (28, 29), (31, 32)  # NASM symmetry
]

pipeline = Pipeline(
    steps=[
        ('preprocessing', CombineCorrelatedFeatures(correlatedColumns)),
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
joblib.dump(pipeline, "saved models/linear_regression_model_v3.pkl")
print("Model saved successfully!")