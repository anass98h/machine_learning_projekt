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



#feature_weights = {
#    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2,  # FSM important features
#    14:1,15: 1,16: 1,17: 2, 18: 2  ,19: 1,20: 1,21: 1,22: 1,23:2, 24: 4, 25: 4,26: 2,27: 2,28: 2, 29: 2, 30: 2,31: 1,32: 1,33: 1, 34: 2, 35: 2, 36: 2, 37: 2, 38: 2,  # NASM key features
#}
#for col_index, weight in feature_weights.items():
#    df.iloc[:, col_index] *= weight



#df.drop(df.columns[[28, 2, 26, 32, 4, 5, 8, 9, 10]], axis=1, inplace=True)


x = df.iloc[:,1:-1]
y = df.iloc[:,0]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Using corr() method
corr_matrix = df.corr()
y_corr = corr_matrix["AimoScore"].sort_values(ascending=False)
for col in y_corr.index:
    col_index = df.columns.get_loc(col)
    correlation_score = y_corr[col]
    print(f"Index: {col_index}, Column: {col}, Correlation Score: {correlation_score:.4f}")

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error: ", mse)
print("R2 Score: ", r2)