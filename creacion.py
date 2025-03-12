import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import xgboost as xgb
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_csv("data-2025-03-10.csv")
print(df.head())

df["job_category"], mapC = pd.factorize(df["job_category"])
df["experience_level"], mapE = pd.factorize(df["experience_level"])
print(df.head())
print(dict(enumerate(mapC)))
print(dict(enumerate(mapE)))

df.columns=["year","experience_level","salary_in_usd","job_category"]

print(df.isna().sum())

X = df.drop('salary_in_usd', axis=1)
y = df['salary_in_usd']

X_train, X_test , y_train, y_test =  train_test_split(X,y,random_state=1)

model = XGBRegressor(max_depth = 3, learning_rate=0.5)

model.fit(X_train,y_train)

predictions = model.predict(X_test)
print(predictions[1:10])
print("Valor real")
print(X_test.iloc[1])
print("Resultado")
print(y_test.iloc[1])

pred_train = model.predict(X_train)
print(r2_score(y_train, pred_train))
print(mean_squared_error(y_train, pred_train))

print(model.score(X_test, y_test))
