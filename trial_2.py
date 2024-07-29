# Import required libraries
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

# Create a directory to save plots
def create_plots_directory():
    plots_dir = 'saved_plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return plots_dir

# Load and prepare the data
def load_and_prepare_data(store_file, sales_file, promotions_file):
    store_df = pd.read_csv(store_file)
    sales_df = pd.read_csv(sales_file)
    promotions_df = pd.read_csv(promotions_file)
    
    print("Columns in store_df:", store_df.columns)
    print("Columns in sales_df:", sales_df.columns)
    
    # Rename conflicting columns
    store_df = store_df.rename(columns={'OPEN_DATE': 'STORE_OPEN_DATE'})
    sales_df = sales_df.rename(columns={'OPEN_DATE': 'SALES_OPEN_DATE'})
    
    # Merge store and sales data
    df = pd.merge(sales_df, store_df, on=['STORE_KEY', 'STORE_NUMBER'], how='left')
    
    print("Columns in merged df:", df.columns)
    
    # Convert date columns to datetime
    date_columns = ['BUSINESS_DATE', 'SALES_OPEN_DATE', 'STORE_OPEN_DATE', 'CLOSE_DATE']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%m/%d/%Y', errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found in dataframe")
    
    # Calculate store age based on SALES_OPEN_DATE
    if 'SALES_OPEN_DATE' in df.columns and 'BUSINESS_DATE' in df.columns:
        df['STORE_AGE'] = (df['BUSINESS_DATE'] - df['SALES_OPEN_DATE']).dt.days
    else:
        print("Warning: Unable to calculate store age due to missing columns")
    
    # Convert promotion date columns to datetime if they exist
    promo_date_columns = ['PROMOTION_START_DATE', 'PROMOTION_END_DATE']
    for col in promo_date_columns:
        if col in promotions_df.columns:
            promotions_df[col] = pd.to_datetime(promotions_df[col], format='%m/%d/%Y', errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found in promotions dataframe")
    
    return df, promotions_df

# Feature engineering
def engineer_features(df, promotions_df):
    # Create lag features
    df['SALES_LAG_7'] = df.groupby('STORE_KEY')['NET_SALES_FINAL_USD_AMOUNT'].shift(7)
    df['SALES_LAG_14'] = df.groupby('STORE_KEY')['NET_SALES_FINAL_USD_AMOUNT'].shift(14)
    
    # Create rolling average features
    df['SALES_ROLLING_7'] = df.groupby('STORE_KEY')['NET_SALES_FINAL_USD_AMOUNT'].rolling(7).mean().reset_index(0, drop=True)
    df['SALES_ROLLING_30'] = df.groupby('STORE_KEY')['NET_SALES_FINAL_USD_AMOUNT'].rolling(30).mean().reset_index(0, drop=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_columns = ['STORE_TYPE_NAME', 'REGION_NAME', 'STATE_NAME', 'CITY_NAME']
    for col in categorical_columns:
        if col in df.columns:
            df[f'{col}_ENCODED'] = le.fit_transform(df[col].astype(str))
    
    # Add promotion features
    df = add_promotion_features(df, promotions_df)
    
    return df

# Add promotion features
def add_promotion_features(df, promotions_df):
    # Create a date range for each promotion
    promo_dates = []
    for _, promo in promotions_df.iterrows():
        dates = pd.date_range(start=promo['PROMOTION_START_DATE'], end=promo['PROMOTION_END_DATE'])
        promo_dates.extend([(date, promo['PROMOTION_TYPE'], promo['PLATFORM']) for date in dates])
    
    promo_df = pd.DataFrame(promo_dates, columns=['DATE', 'PROMOTION_TYPE', 'PLATFORM'])
    
    # Merge promotions with main dataframe
    df = pd.merge(df, promo_df, left_on='BUSINESS_DATE', right_on='DATE', how='left')
    
    # Create binary features for promotion types and platforms
    df['HAS_PROMOTION'] = df['PROMOTION_TYPE'].notna().astype(int)
    df['IS_DELIVERY_PROMO'] = (df['PROMOTION_TYPE'] == 'Delivery Fee').astype(int)
    df['IS_INHOUSE_PLATFORM'] = (df['PLATFORM'] == 'Inhouse').astype(int)
    df['IS_DD_PLATFORM'] = (df['PLATFORM'] == 'DD').astype(int)
    
    return df

# Feature selection with handling missing values
def select_features(df, target_col):
    # List of columns to exclude
    exclude_cols = [target_col, 'BUSINESS_DATE', 'SALES_OPEN_DATE', 'STORE_OPEN_DATE', 'CLOSE_DATE', 
                    'ROW_EFFECTIVE_TIMESTAMP', 'ROW_EXPIRATION_TIMESTAMP', 'EDW_CREATE_TIMESTAMP', 'EDW_MODIFY_TIMESTAMP']
    
    # Drop non-numeric columns and excluded columns if they exist
    cols_to_drop = [col for col in exclude_cols if col in df.columns]
    X = df.select_dtypes(include=[np.number]).drop(columns=cols_to_drop, errors='ignore')
    
    # Drop columns with all NaN values
    X = X.dropna(axis=1, how='all')
    
    y = df[target_col]
    
    # Impute missing values with the mean (or other strategy)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_imputed, y)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select top 15 features
    selected_features = feature_importance.head(15)['feature'].tolist()
    
    return selected_features, feature_importance

# Prepare data for Prophet
def prepare_prophet_data(df, store_key, features):
    store_data = df[df['STORE_KEY'] == store_key].sort_values('BUSINESS_DATE')
    
    prophet_df = pd.DataFrame({
        'ds': store_data['BUSINESS_DATE'],
        'y': store_data['NET_SALES_FINAL_USD_AMOUNT']
    })
    
    for feature in features:
        prophet_df[feature] = store_data[feature]
    
    return prophet_df

# Train Prophet model and make predictions
def train_and_predict(train_df, test_df, features, future_periods=90):
    model = Prophet()
    for feature in features:
        model.add_regressor(feature)
    
    model.fit(train_df)
    
    # Create future dataframe including test data and additional 90 days
    last_date = max(train_df['ds'].max(), test_df['ds'].max())
    future = model.make_future_dataframe(periods=len(test_df) + future_periods)
    
    # Fill known features for test period
    for feature in features:
        future.loc[:len(train_df) + len(test_df) - 1, feature] = pd.concat([train_df[feature], test_df[feature]]).reset_index(drop=True)
    
    # For the additional 90 days, use the last known value for each feature
    for feature in features:
        future.loc[len(train_df) + len(test_df):, feature] = future.loc[len(train_df) + len(test_df) - 1, feature]
    
    forecast = model.predict(future)
    
    return model, forecast

# Evaluate model performance
def evaluate_model(test_df, forecast):
    y_true = test_df['y'].values
    y_pred = forecast.tail(len(test_df))['yhat'].values
    
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return mape

# Plot feature importance
def plot_feature_importance(feature_importance, plots_dir):
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
    plt.close()

# Plot sales distribution
def plot_sales_distribution(df, plots_dir):
    plt.figure(figsize=(10, 6))
    plt.hist(df['NET_SALES_FINAL_USD_AMOUNT'].dropna(), bins=50, edgecolor='k')
    plt.xlabel('Net Sales (USD)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Net Sales')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'sales_distribution.png'))
    plt.close()

# Plot historical sales data
def plot_sales_data(df, store_key, plots_dir):
    store_data = df[df['STORE_KEY'] == store_key]
    plt.figure(figsize=(12, 6))
    plt.plot(store_data['BUSINESS_DATE'], store_data['NET_SALES_FINAL_USD_AMOUNT'], marker='o', linestyle='-')
    plt.xlabel('Date')
    plt.ylabel('Net Sales (USD)')
    plt.title(f'Historical Sales Data for Store {store_key}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'historical_sales_store_{store_key}.png'))
    plt.close()

# Plot forecast with important relationships
def plot_forecast_with_relationships(model, forecast, store_data, features, future_periods, plots_dir):
    plt.figure(figsize=(16, 12))
    
    # Plot historical data and forecast
    plt.subplot(3, 1, 1)
    plt.plot(store_data['ds'], store_data['y'], label='Historical Data', color='blue')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='red')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.2)
    plt.xlabel('Date')
    plt.ylabel('Net Sales (USD)')
    plt.title('Sales Forecast with 3 Months Projection')
    plt.legend()
    
    # Plot important features
    num_features = min(4, len(features))  # Plot top 4 features or less if fewer are available
    for i, feature in enumerate(features[:num_features], 1):
        plt.subplot(3, 2, i+2)
        plt.plot(forecast['ds'], forecast[feature], label=feature)
        plt.xlabel('Date')
        plt.ylabel(feature)
        plt.title(f'Feature: {feature}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'forecast_with_relationships.png'))
    plt.close()

# Check for columns with all NaN values
def check_all_missing_columns(df):
    all_missing = df.columns[df.isna().all()].tolist()
    if all_missing:
        print(f"Columns with all missing values: {all_missing}")
    return all_missing

# Main function
def main():
    # Create plots directory
    plots_dir = create_plots_directory()

    # Load and prepare data
    df, promotions_df = load_and_prepare_data('store.csv', 'sales.csv', 'promotions.csv')
    
    print(f"Shape of merged dataframe: {df.shape}")
    print(f"Columns in merged dataframe: {df.columns}")
    print(df.head())
    
    # Engineer features
    df = engineer_features(df, promotions_df)
    
    # Check for columns with all NaN values
    all_missing_cols = check_all_missing_columns(df)
    if all_missing_cols:
        df = df.drop(columns=all_missing_cols)
    
    # Select features
    selected_features, feature_importance = select_features(df, 'NET_SALES_FINAL_USD_AMOUNT')
    print("Selected features:", selected_features)
    
    # Plot and save feature importance
    plot_feature_importance(feature_importance, plots_dir)
    
    # Plot and save sales distribution
    plot_sales_distribution(df, plots_dir)
    
    # Select a store for demonstration
    store_key = df['STORE_KEY'].unique()[0]
    
    # Plot and save historical sales data
    plot_sales_data(df, store_key, plots_dir)
    
    # Prepare data for the selected store
    store_data = prepare_prophet_data(df, store_key, selected_features)
    
    # Split data into train and test sets
    train_df, test_df = train_test_split(store_data, test_size=0.2, shuffle=False)
    
    # Train model and make predictions for next 3 months
    model, forecast = train_and_predict(train_df, test_df, selected_features, future_periods=90)
    
    # Evaluate model
    mape = evaluate_model(test_df, forecast.iloc[:len(test_df)])
    print(f"MAPE for store {store_key}: {mape:.2%}")
    
    # Plot and save forecast results with important relationships
    plot_forecast_with_relationships(model, forecast, store_data, selected_features, future_periods=90, plots_dir=plots_dir)

    print(f"All plots have been saved in the '{plots_dir}' directory.")

if __name__ == "__main__":
    main()