import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv("promedio_salarios_por_categoria_anio_experiencia.csv")
print(df.head())

df["job_category"], mapC = pd.factorize(df["job_category"])
df["experience_level"], mapE = pd.factorize(df["experience_level"])
print(df.head())
print(dict(enumerate(mapC)))
print(dict(enumerate(mapE)))

df.columns=["job_category","work_year","experience_level","avg_salary_usd"]

print(df.isna().sum())

X = df.drop('avg_salary_usd', axis=1)
y = df['avg_salary_usd']

# Crear interacción personalizada entre work_year y experience_level
X['work_year_x_exp'] = X['work_year'] * X['experience_level']

# Agregar transformaciones no lineales para work_year
X['work_year_squared'] = X['work_year'] ** 2
X['work_year_log'] = np.log1p(X['work_year'])

# Visualizar la relación entre work_year y avg_salary_usd
sns.lineplot(x="work_year", y="avg_salary_usd", data=df)
plt.title("Evolución de los salarios por año")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Ajustar el modelo con constraints para reforzar la influencia de work_year
model = XGBRegressor(
    max_depth=5,
    learning_rate=0.1,
    interaction_constraints='[[0, 1, 2, 3, 4, 5]]',  # Fuerza interacciones
    random_state=1
)

# Dar más peso a work_year durante el entrenamiento
sample_weight = X_train['work_year'] / X_train['work_year'].mean()

model.fit(X_train, y_train, sample_weight=sample_weight)

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

xgb.plot_importance(model, importance_type="gain", ax=plt.gca())
plt.show()

model.save_model('modelo_xgb.json')