import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns
from matplotlib.dates import DateFormatter

# Define the function to reduce memory usage
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df

# Define the function to calculate MAPE
def custom_mape(y_true, y_pred, epsilon=1e-10):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / (np.abs(y_true[mask]) + epsilon))) * 100

# Define the function to predict in batches
def predict_in_batches(model, X, batch_size=10000):
    predictions = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        pred_batch = model.predict(X_batch)
        predictions.append(pred_batch)
    return np.concatenate(predictions)

# Load trained models and imputer
rf_model = load('model_results/RandomForest_model.joblib')
lgbm_model = load('model_results/LightGBM_model.joblib')
imputer = load('model_results/imputer.joblib')

# Load data
sales_df = pd.read_csv('sales.csv')
store_df = pd.read_csv('store.csv')
promotions_df = pd.read_csv('promotions.csv')

store_df = reduce_mem_usage(store_df)
promotions_df = reduce_mem_usage(promotions_df)

# Process the data for months 16-18
sales_df['BUSINESS_DATE'] = pd.to_datetime(sales_df['BUSINESS_DATE'])
promotions_df['PROMOTION_START_DATE'] = pd.to_datetime(promotions_df['PROMOTION_START_DATE'])
promotions_df['PROMOTION_END_DATE'] = pd.to_datetime(promotions_df['PROMOTION_END_DATE'])

start_date = pd.Timestamp('2023-04-01')  # Assuming the dataset starts on 2022-01-01
end_date = start_date + pd.DateOffset(months=3)
mask = (sales_df['BUSINESS_DATE'] >= start_date) & (sales_df['BUSINESS_DATE'] < end_date)
eval_df = sales_df[mask]

# Merge with store and promotions data
eval_df = eval_df.merge(store_df, on='STORE_KEY', how='left')
eval_df = eval_df.merge(promotions_df, left_on='BUSINESS_DATE', right_on='PROMOTION_START_DATE', how='left')

# Feature engineering
eval_df['DAY_OF_WEEK'] = eval_df['BUSINESS_DATE'].dt.dayofweek
eval_df['MONTH'] = eval_df['BUSINESS_DATE'].dt.month
eval_df['YEAR'] = eval_df['BUSINESS_DATE'].dt.year
eval_df['IS_WEEKEND'] = eval_df['DAY_OF_WEEK'].isin([5, 6]).astype(int)

if 'OPEN_DATE' in eval_df.columns:
    eval_df['OPEN_DATE'] = pd.to_datetime(eval_df['OPEN_DATE'])
    eval_df['DAYS_SINCE_STORE_OPEN'] = (eval_df['BUSINESS_DATE'] - eval_df['OPEN_DATE']).dt.days
else:
    eval_df['FIRST_APPEARANCE'] = eval_df.groupby('STORE_KEY')['BUSINESS_DATE'].transform('min')
    eval_df['DAYS_SINCE_STORE_OPEN'] = (eval_df['BUSINESS_DATE'] - eval_df['FIRST_APPEARANCE']).dt.days

# Prepare features and target
features = ['STORE_KEY', 'DAY_OF_WEEK', 'MONTH', 'YEAR', 'IS_WEEKEND', 'DAYS_SINCE_STORE_OPEN']
X_eval = eval_df[features]
y_eval = eval_df['NET_SALES_FINAL_USD_AMOUNT']

# Impute missing values
X_eval_imputed = imputer.transform(X_eval)

# Make predictions
rf_predictions = predict_in_batches(rf_model, X_eval_imputed)
lgbm_predictions = predict_in_batches(lgbm_model, X_eval_imputed)

# Calculate metrics
rf_mape = custom_mape(y_eval, rf_predictions)
lgbm_mape = custom_mape(y_eval, lgbm_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_eval, rf_predictions))
lgbm_rmse = np.sqrt(mean_squared_error(y_eval, lgbm_predictions))

print(f"Random Forest MAPE: {rf_mape:.2f}")
print(f"LightGBM MAPE: {lgbm_mape:.2f}")
print(f"Random Forest RMSE: {rf_rmse:.2f}")
print(f"LightGBM RMSE: {lgbm_rmse:.2f}")

# Calculate rolling averages for smoothing
window = 7  # 7-day rolling average
y_eval_smooth = y_eval.rolling(window=window, center=True).mean()
rf_predictions_smooth = pd.Series(rf_predictions).rolling(window=window, center=True).mean()
lgbm_predictions_smooth = pd.Series(lgbm_predictions).rolling(window=window, center=True).mean()

# Set up the plot style
plt.style.use('seaborn')
sns.set_palette("deep")

# Create the plot
fig, ax = plt.subplots(figsize=(16, 8))

# Plot the data
ax.plot(eval_df['BUSINESS_DATE'], y_eval_smooth, label='Actual', linewidth=2)
ax.plot(eval_df['BUSINESS_DATE'], rf_predictions_smooth, label='Random Forest', linewidth=2)
ax.plot(eval_df['BUSINESS_DATE'], lgbm_predictions_smooth, label='LightGBM', linewidth=2)

# Customize the plot
ax.set_title('Actual vs Predicted Sales (7-day Rolling Average)', fontsize=20, pad=20)
ax.set_xlabel('Date', fontsize=14, labelpad=10)
ax.set_ylabel('Sales (USD)', fontsize=14, labelpad=10)
ax.legend(fontsize=12, loc='upper left')

# Format x-axis ticks
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45, ha='right')

# Add grid
ax.grid(True, linestyle='--', alpha=0.7)

# Add metrics to the plot
metrics_text = f"Random Forest MAPE: {rf_mape:.2f}%\nLightGBM MAPE: {lgbm_mape:.2f}%\n"
metrics_text += f"Random Forest RMSE: {rf_rmse:.2f}\nLightGBM RMSE: {lgbm_rmse:.2f}"
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# Adjust layout and save
plt.tight_layout()
plt.savefig('model_results/improved_actual_vs_predicted_16_18_months.png', dpi=300)
plt.show()