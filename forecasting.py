import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt
import os

os.makedirs('model_results', exist_ok=True)

df = pd.read_csv('processed_data/prepared_data.csv')
df['BUSINESS_DATE'] = pd.to_datetime(df['BUSINESS_DATE'])

features = ['STORE_KEY', 'DAY_OF_WEEK', 'MONTH', 'IS_WEEKEND', 'SALES_LAG_1', 'SALES_LAG_7', 
            'STORE_TYPE_NAME', 'STATE_NAME', 'PROMOTION_TYPE', 'PROMOTION_CHANNEL']
target = 'NET_SALES_FINAL_USD_AMOUNT'

train_df = df[df['BUSINESS_DATE'] < '2023-01-01']
test_df = df[df['BUSINESS_DATE'] >= '2023-01-01']

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

train_predictions = rf_model.predict(X_train)
test_predictions = rf_model.predict(X_test)

train_mape = mean_absolute_percentage_error(y_train, train_predictions)
test_mape = mean_absolute_percentage_error(y_test, test_predictions)
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

with open('model_results/model_metrics.txt', 'w') as f:
    f.write(f"Train MAPE: {train_mape:.2f}\n")
    f.write(f"Test MAPE: {test_mape:.2f}\n")
    f.write(f"Train RMSE: {train_rmse:.2f}\n")
    f.write(f"Test RMSE: {test_rmse:.2f}\n")

print(f"Train MAPE: {train_mape:.2f}")
print(f"Test MAPE: {test_mape:.2f}")
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(test_df['BUSINESS_DATE'], y_test, label='Actual')
plt.plot(test_df['BUSINESS_DATE'], test_predictions, label='Predicted')
plt.title('Actual vs Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Sales (USD)')
plt.legend()
plt.savefig('model_results/actual_vs_predicted.png')
plt.close()

feature_importance = pd.DataFrame({'feature': features, 'importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_results/feature_importance.png')
plt.close()

feature_importance.to_csv('model_results/feature_importance.csv', index=False)

print("Model training and evaluation completed. Check the 'model_results' folder for outputs.")



