import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# Copy dataset
df = pd.read_csv('filtered_data.csv')

# Function to remove outliers based on IQR for each experience level
def remove_outliers_by_experience(df, column='salary_in_usd', group_by='experience_level'):
    df_filtered = df.copy()
    
    for level in df[group_by].unique():
        subset = df[df[group_by] == level]
        
        Q1 = subset[column].quantile(0.25)  # 25th percentile
        Q3 = subset[column].quantile(0.75)  # 75th percentile
        IQR = Q3 - Q1  # Interquartile range

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Remove outliers only within this experience level
        df_filtered = df_filtered.drop(df_filtered[(df_filtered[group_by] == level) & 
                                                   ((df_filtered[column] < lower_bound) | 
                                                    (df_filtered[column] > upper_bound))].index)

    return df_filtered

# Apply outlier removal
df = remove_outliers_by_experience(df)

# Apply log transformation to salary
df['salary_in_usd'] = np.log1p(df['salary_in_usd'])

# Select features and target variable
X = df[['work_year', 'experience_level', 'job_title', 'employee_residence', 
        'remote_ratio', 'company_location', 'company_size']]
y = df['salary_in_usd']

# Apply OneHotEncoding to categorical columns
categorical_cols = ['experience_level', 'job_title', 'employee_residence', 'company_location', 'company_size']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # Encoding categorical variables
X_encoded = encoder.fit_transform(X[categorical_cols])

# Convert to DataFrame and concatenate with numerical features
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols))
X_final = pd.concat([X.drop(columns=categorical_cols).reset_index(drop=True), X_encoded_df], axis=1)

# Apply StandardScaler to numerical features
scaler = StandardScaler()
X_final_scaled = scaler.fit_transform(X_final)

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_final_scaled, y, test_size=0.2, random_state=42)

# Train RandomForestRegressor (better for non-linear relationships)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5  # Square root to get RMSE
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R² Score: {r2}")

# Make a new prediction
new_data = np.array([[2026, 'MI', 'Data Scientist', 'US', 0, 'US', 'M']])  # Replace with real values

# Encode and scale new data
new_data_df = pd.DataFrame(new_data, columns=['work_year', 'experience_level', 'job_title', 
                                              'employee_residence', 'remote_ratio', 
                                              'company_location', 'company_size'])
new_data_encoded = encoder.transform(new_data_df[categorical_cols])
new_data_encoded_df = pd.DataFrame(new_data_encoded, columns=encoder.get_feature_names_out(categorical_cols))
new_data_final = pd.concat([new_data_df.drop(columns=categorical_cols).reset_index(drop=True), 
                            new_data_encoded_df], axis=1)
new_data_scaled = scaler.transform(new_data_final)

# Predict and reverse log transformation
predicted_salary = np.expm1(model.predict(new_data_scaled))
print(f"Predicted Salary: ${predicted_salary[0]:,.2f}")
import joblib  # Para guardar objetos como modelos, encoder, scaler

# Guardar el modelo entrenado
joblib.dump(model, 'trained_random_forest_model.pkl')

# Guardar el encoder y el scaler
joblib.dump(encoder, 'onehot_encoder.pkl')
joblib.dump(scaler, 'standard_scaler.pkl')