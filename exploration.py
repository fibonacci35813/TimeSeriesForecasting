import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

os.makedirs('processed_data', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

sales_df = pd.read_csv('sales.csv')
store_df = pd.read_csv('store.csv')
promotions_df = pd.read_csv('promotions.csv')

sales_df['BUSINESS_DATE'] = pd.to_datetime(sales_df['BUSINESS_DATE'])
store_df['OPEN_DATE'] = pd.to_datetime(store_df['OPEN_DATE'])
promotions_df['PROMOTION_START_DATE'] = pd.to_datetime(promotions_df['PROMOTION_START_DATE'])
promotions_df['PROMOTION_END_DATE'] = pd.to_datetime(promotions_df['PROMOTION_END_DATE'])

merged_df = sales_df.merge(store_df, on='STORE_KEY', how='left')
merged_df = merged_df.merge(promotions_df, left_on='BUSINESS_DATE', right_on='PROMOTION_START_DATE', how='left')

# Feature engineering
merged_df['DAY_OF_WEEK'] = merged_df['BUSINESS_DATE'].dt.dayofweek
merged_df['MONTH'] = merged_df['BUSINESS_DATE'].dt.month
merged_df['IS_WEEKEND'] = merged_df['DAY_OF_WEEK'].isin([5, 6]).astype(int)

# Create lagged features
merged_df['SALES_LAG_1'] = merged_df.groupby('STORE_KEY')['NET_SALES_FINAL_USD_AMOUNT'].shift(1)
merged_df['SALES_LAG_7'] = merged_df.groupby('STORE_KEY')['NET_SALES_FINAL_USD_AMOUNT'].shift(7)

# Handle missing values
merged_df['SALES_LAG_1'].fillna(merged_df['SALES_LAG_1'].mean(), inplace=True)
merged_df['SALES_LAG_7'].fillna(merged_df['SALES_LAG_7'].mean(), inplace=True)

# Encode categorical variables
le = LabelEncoder()
categorical_cols = ['STORE_TYPE_NAME', 'STATE_NAME', 'PROMOTION_TYPE', 'PROMOTION_CHANNEL']
for col in categorical_cols:
    merged_df[col] = le.fit_transform(merged_df[col].astype(str))

# Save the prepared dataset
merged_df.to_csv('processed_data/prepared_data.csv', index=False)
print("Prepared data saved to 'processed_data/prepared_data.csv'")

# Visualize sales trends
plt.figure(figsize=(12, 6))
merged_df.groupby('BUSINESS_DATE')['NET_SALES_FINAL_USD_AMOUNT'].mean().plot()
plt.title('Average Daily Sales Trend')
plt.xlabel('Date')
plt.ylabel('Average Sales (USD)')
plt.savefig('visualizations/sales_trend.png')
plt.close()

# Analyze sales by day of week
plt.figure(figsize=(10, 5))
sns.boxplot(x='DAY_OF_WEEK', y='NET_SALES_FINAL_USD_AMOUNT', data=merged_df)
plt.title('Sales Distribution by Day of Week')
plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
plt.ylabel('Sales (USD)')
plt.savefig('visualizations/sales_by_day.png')
plt.close()

# Correlation heatmap
corr_matrix = merged_df[['NET_SALES_FINAL_USD_AMOUNT', 'TRANSACTION_FINAL_COUNT', 'SALES_LAG_1', 'SALES_LAG_7', 'DAY_OF_WEEK', 'MONTH']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('visualizations/correlation_heatmap.png')
plt.close()

# Print and save summary statistics
summary_stats = merged_df.describe()
print(summary_stats)
summary_stats.to_csv('processed_data/summary_statistics.csv')

print("Data preparation and exploration completed. Check the 'processed_data' and 'visualizations' folders for outputs.")