import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_excel("ML/data/AimoScore_WeakLink_big_scores.xls")

x = df.iloc[:,1:-1]
y = df.iloc[:,0]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lr_loaded = joblib.load("ML/saved models/linear_regression_model_v3.pkl")


y_pred = lr_loaded.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error: ", mse)
print("R2 Score: ", r2)

