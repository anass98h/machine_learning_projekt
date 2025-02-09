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


feature_df = pd.read_excel("ML/data/classifier/AimoScore_WeakLink_big_scores.xls")
weaklink_df = pd.read_excel("ML/data/classifier/20190108 scores_and_weak_links.xlsx")

weakest = weaklink_df.iloc[:, 3:].idxmax(axis=1)
weaklink_df["Weakest"] = weakest
weaklink_df = weaklink_df[["ID", "Weakest"]]

feature_df.drop(columns=["AimoScore", "No_1_Time_Deviation", "No_2_Time_Deviation", "EstimatedScore"], inplace=True)

df = feature_df.merge(weaklink_df, "inner", "ID")
df.drop(columns=["ID"], inplace=True)

print(df.head())
