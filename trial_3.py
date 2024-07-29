import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt
import os
from datetime import timedelta
import os


os.makedirs('model_results', exist_ok=True)

# Load and prepare the data
df = pd.read_csv('processed_data/prepared_data.csv')
df['BUSINESS_DATE'] = pd.to_datetime(df['BUSINESS_DATE'])

features = ['STORE_KEY', 'DAY_OF_WEEK', 'MONTH', 'IS_WEEKEND', 'SALES_LAG_1', 'SALES_LAG_7', 'STORE_TYPE_NAME', 'STATE_NAME', 'PROMOTION_TYPE', 'PROMOTION_CHANNEL']
target = 'NET_SALES_FINAL_USD_AMOUNT'

# Split the data
train_df = df[df['BUSINESS_DATE'] < '2023-01-01']
test_df = df[df['BUSINESS_DATE'] >= '2023-01-01']

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# Train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on test data
test_predictions = rf_model.predict(X_test)

# Calculate metrics
test_mape = mean_absolute_percentage_error(y_test, test_predictions)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print(f"Test MAPE: {test_mape:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")

# Generate future dates for the next 3 months
last_date = df['BUSINESS_DATE'].max()
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=90)

# Create a DataFrame for future predictions
future_df = pd.DataFrame({'BUSINESS_DATE': future_dates})
future_df['DAY_OF_WEEK'] = future_df['BUSINESS_DATE'].dt.dayofweek
future_df['MONTH'] = future_df['BUSINESS_DATE'].dt.month
future_df['IS_WEEKEND'] = future_df['DAY_OF_WEEK'].isin([5, 6]).astype(int)

# Fill in other features with the most recent values or averages
for feature in features:
    if feature not in future_df.columns:
        if feature in ['SALES_LAG_1', 'SALES_LAG_7']:
            future_df[feature] = df[feature].mean()
        else:
            future_df[feature] = df[feature].iloc[-1]

# Make predictions for the future
future_predictions = rf_model.predict(future_df[features])

# Visualize actual, test predictions, and future predictions
plt.figure(figsize=(16, 8))
plt.plot(test_df['BUSINESS_DATE'], y_test, label='Actual', alpha=0.7)
plt.plot(test_df['BUSINESS_DATE'], test_predictions, label='Test Predictions', alpha=0.7)
plt.plot(future_df['BUSINESS_DATE'], future_predictions, label='Future Predictions', linestyle='--')

plt.title('Sales Forecast: Actual, Test Predictions, and 3-Month Forecast')
plt.xlabel('Date')
plt.ylabel('Sales (USD)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_results/sales_forecast_3_months.png')
plt.close()

# Feature importance
feature_importance = pd.DataFrame({'feature': features, 'importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_results/feature_importance.png')
plt.close()

feature_importance.to_csv('model_results/feature_importance.csv', index=False)

print("Model training, evaluation, and future forecasting completed. Check the 'model_results' folder for outputs.")