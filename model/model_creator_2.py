import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Cargar dataset
df = pd.read_csv('model/filtered_data.csv')

# Remover outliers por experiencia
def remove_outliers_by_experience(df, column='salary_in_usd', group_by='experience_level'):
    df_filtered = df.copy()
    for level in df[group_by].unique():
        subset = df[df[group_by] == level]
        Q1 = subset[column].quantile(0.25)
        Q3 = subset[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_filtered = df_filtered.drop(
            df_filtered[(df_filtered[group_by] == level) &
                        ((df_filtered[column] < lower_bound) | 
                         (df_filtered[column] > upper_bound))].index)
    return df_filtered

df = remove_outliers_by_experience(df)

# Agrupar profesiones
def agrupar_job_title(title):
    title = title.lower()
    if 'scientist' in title:
        return 'Scientist'
    elif 'engineer' in title:
        return 'Engineer'
    elif 'analyst' in title:
        return 'Analyst'
    elif 'manager' in title:
        return 'Manager'
    else:
        return 'Other'

df['job_group'] = df['job_title'].apply(agrupar_job_title)

# Inflación por año (4.5% acumulado)
inflation_map = {
    year: 1.045 ** (year - 2022)
    for year in range(2022, 2031)  # 👈 hasta 2030 incluido
}
print("📈 Mapa de inflación:", inflation_map)
df['inflation_factor'] = df['work_year'].map(inflation_map)

# Ajustar salario e incluir interacción
df['adjusted_salary'] = df['salary_in_usd'] / df['inflation_factor']
df['log_adjusted_salary'] = np.log1p(df['adjusted_salary'])
df['exp_year_combo'] = df['experience_level'] + "_" + df['work_year'].astype(str)

# Features y target
X = df[['work_year', 'experience_level', 'job_group', 'employee_residence',
        'remote_ratio', 'company_location', 'company_size', 'exp_year_combo']]
y = df['log_adjusted_salary']

# Separar train y test
train_df = df[df['work_year'] < 2025].copy()
test_df = df[df['work_year'] == 2025].copy()
X_train = train_df[X.columns]
y_train = train_df['log_adjusted_salary']
X_test = test_df[X.columns]
y_test = test_df['log_adjusted_salary']

# Codificación
categorical_cols = ['experience_level', 'job_group', 'employee_residence', 'company_location', 'company_size', 'exp_year_combo']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
X_test_encoded = encoder.transform(X_test[categorical_cols])

X_train_final = pd.concat([
    X_train.drop(columns=categorical_cols).reset_index(drop=True),
    pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(categorical_cols))
], axis=1)
X_test_final = pd.concat([
    X_test.drop(columns=categorical_cols).reset_index(drop=True),
    pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(categorical_cols))
], axis=1)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)

# Hiperparámetros y búsqueda
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2.0]
}

print("🔍 Ejecutando RandomizedSearchCV...")
xgb = XGBRegressor(random_state=42, n_jobs=-1)
search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=30,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=2,
    random_state=42
)
search.fit(X_train_scaled, y_train)
best_model = search.best_estimator_
print("✅ Mejores hiperparámetros encontrados:", search.best_params_)

# Evaluar con mejor modelo
model = best_model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Resultados del modelo optimizado:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# ----------- Predicciones futuras -----------

for target_year in [2026, 2027]:
    new_data = pd.DataFrame([{
        'work_year': target_year,
        'experience_level': 'MI',
        'job_group': 'Scientist',
        'employee_residence': 'US',
        'remote_ratio': 0,
        'company_location': 'US',
        'company_size': 'M',
        'exp_year_combo': 'MI_' + str(target_year)
    }])

    new_encoded = encoder.transform(new_data[categorical_cols])
    new_encoded_df = pd.DataFrame(new_encoded, columns=encoder.get_feature_names_out(categorical_cols))
    new_final = pd.concat([new_data.drop(columns=categorical_cols).reset_index(drop=True), new_encoded_df], axis=1)
    new_scaled = scaler.transform(new_final)

    predicted_adjusted_log = model.predict(new_scaled)
    predicted_adjusted = np.expm1(predicted_adjusted_log)
    predicted_final_salary = predicted_adjusted * inflation_map[target_year]

    print(f"Predicted Salary for {target_year}: ${predicted_final_salary[0]:,.2f}")

# ----------- Importancia de variables -----------

feature_names = X_train_final.columns
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
plt.title('Top 20 Feature Importances - Optimized XGBoost')
plt.tight_layout()
plt.show()

# ----------- Real vs Predicho -----------

plt.figure(figsize=(8, 6))
plt.scatter(np.expm1(y_test * inflation_map[2025]), np.expm1(y_pred * inflation_map[2025]), alpha=0.4)
plt.plot([0, 300_000], [0, 300_000], 'r--')
plt.xlabel("Real Salary (USD, 2025)")
plt.ylabel("Predicted Salary (USD, 2025)")
plt.title("Real vs Predicted Salaries (2025) - Optimized")
plt.grid(True)
plt.tight_layout()
plt.show()

joblib.dump(model, "model/xgb_salary_model2.pkl")
joblib.dump(encoder, "model/onehot_encoder2.pkl")
joblib.dump(scaler, "model/standard_scaler2.pkl")

print("✅ Modelo, encoder y scaler guardados correctamente.")