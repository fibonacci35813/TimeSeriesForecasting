import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
import os
import gc
from joblib import dump, load

# Create directory for saving models and results
os.makedirs('model_results', exist_ok=True)

# Function to reduce memory usage of dataframes
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

# Custom MAPE function
def custom_mape(y_true, y_pred, epsilon=1e-10):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / (np.abs(y_true[mask]) + epsilon))) * 100

# Function to process chunks of data
def process_chunk(chunk, store_df, promotions_df):
    try:
        chunk = reduce_mem_usage(chunk)
        chunk['BUSINESS_DATE'] = pd.to_datetime(chunk['BUSINESS_DATE'])
        chunk = chunk.merge(store_df, on='STORE_KEY', how='left')
        chunk = chunk.merge(promotions_df, left_on='BUSINESS_DATE', right_on='PROMOTION_START_DATE', how='left')
        
        chunk['DAY_OF_WEEK'] = chunk['BUSINESS_DATE'].dt.dayofweek
        chunk['MONTH'] = chunk['BUSINESS_DATE'].dt.month
        chunk['YEAR'] = chunk['BUSINESS_DATE'].dt.year
        chunk['IS_WEEKEND'] = chunk['DAY_OF_WEEK'].isin([5, 6]).astype(int)
        
        if 'OPEN_DATE' in chunk.columns:
            chunk['DAYS_SINCE_STORE_OPEN'] = (chunk['BUSINESS_DATE'] - chunk['OPEN_DATE']).dt.days
        else:
            chunk['FIRST_APPEARANCE'] = chunk.groupby('STORE_KEY')['BUSINESS_DATE'].transform('min')
            chunk['DAYS_SINCE_STORE_OPEN'] = (chunk['BUSINESS_DATE'] - chunk['FIRST_APPEARANCE']).dt.days
        
        print(f"Processed chunk shape: {chunk.shape}")
        print(f"Processed chunk columns: {chunk.columns}")
        
        return chunk
    except Exception as e:
        print(f"Error in process_chunk: {str(e)}")
        return None
   
# Function to train a model with batching
def train_model(model, X, y, batch_size=10000):
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        if hasattr(model, 'partial_fit'):
            model.partial_fit(X_batch, y_batch)
        else:
            model.fit(X_batch, y_batch)
    return model

# Function to predict in batches
def predict_in_batches(model, X, batch_size=10000):
    predictions = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        pred_batch = model.predict(X_batch)
        predictions.append(pred_batch)
    return np.concatenate(predictions)

try:
    # Load and process data
    chunk_size = 100000
    sales_chunks = pd.read_csv('sales.csv', chunksize=chunk_size)
    store_df = pd.read_csv('store.csv')
    promotions_df = pd.read_csv('promotions.csv')

    store_df = reduce_mem_usage(store_df)
    promotions_df = reduce_mem_usage(promotions_df)

    if 'OPEN_DATE' in store_df.columns:
        store_df['OPEN_DATE'] = pd.to_datetime(store_df['OPEN_DATE'])
    promotions_df['PROMOTION_START_DATE'] = pd.to_datetime(promotions_df['PROMOTION_START_DATE'])
    promotions_df['PROMOTION_END_DATE'] = pd.to_datetime(promotions_df['PROMOTION_END_DATE'])

    # Process data in chunks
    features = ['STORE_KEY', 'DAY_OF_WEEK', 'MONTH', 'YEAR', 'IS_WEEKEND', 'DAYS_SINCE_STORE_OPEN', 'BUSINESS_DATE']
    target = 'NET_SALES_FINAL_USD_AMOUNT'

    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []

    for chunk in sales_chunks:
        processed_chunk = process_chunk(chunk, store_df, promotions_df)
        if processed_chunk is not None:
            train_mask = processed_chunk['BUSINESS_DATE'] < '2023-01-01'
            
            X_train_chunk = processed_chunk[train_mask][features]
            y_train_chunk = processed_chunk[train_mask][target]
            X_test_chunk = processed_chunk[~train_mask][features]
            y_test_chunk = processed_chunk[~train_mask][target]
            
            X_train_list.append(X_train_chunk)
            y_train_list.append(y_train_chunk)
            X_test_list.append(X_test_chunk)
            y_test_list.append(y_test_chunk)
            
            del processed_chunk, X_train_chunk, y_train_chunk, X_test_chunk, y_test_chunk
            gc.collect()

    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = pd.concat(y_train_list, ignore_index=True)
    X_test = pd.concat(X_test_list, ignore_index=True)
    y_test = pd.concat(y_test_list, ignore_index=True)

    del X_train_list, y_train_list, X_test_list, y_test_list
    gc.collect()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_train columns: {X_train.columns}")
    print(f"X_test shape: {X_test.shape}")
    print(f"X_test columns: {X_test.columns}")

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train.drop('BUSINESS_DATE', axis=1))
    X_test_imputed = imputer.transform(X_test.drop('BUSINESS_DATE', axis=1))

    # Train models
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, warm_start=True),
        'LightGBM': LGBMRegressor(
            n_estimators=100,
            num_leaves=31,
            max_depth=10,
            min_data_in_leaf=20,
            random_state=42,
            n_jobs=-1,
            force_col_wise=True
        )
    }

    results = {}

    for name, model in models.items():
        print(f"Training {name} model...")
        print(f"X_train_imputed shape: {X_train_imputed.shape}")
        print(f"y_train shape: {y_train.shape}")
        
        trained_model = train_model(model, X_train_imputed, y_train)
        
        print("Predicting on train set...")
        train_predictions = predict_in_batches(trained_model, X_train_imputed)
        print("Predicting on test set...")
        test_predictions = predict_in_batches(trained_model, X_test_imputed)
        
        print("Calculating metrics...")
        train_mape = custom_mape(y_train, train_predictions)
        test_mape = custom_mape(y_test, test_predictions)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        
        results[name] = {
            'train_mape': train_mape,
            'test_mape': test_mape,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_predictions': test_predictions
        }
        
        # Save the model
        dump(trained_model, f'model_results/{name}_model.joblib')
        
        print(f"Finished training {name} model")
        
        # Clear memory after each model
        del trained_model, train_predictions, test_predictions
        gc.collect()

    # Save results to file
    with open('model_results/results.txt', 'w') as f:
        for name, metrics in results.items():
            f.write(f"Model: {name}\n")
            f.write(f"Train MAPE: {metrics['train_mape']:.2f}\n")
            f.write(f"Test MAPE: {metrics['test_mape']:.2f}\n")
            f.write(f"Train RMSE: {metrics['train_rmse']:.2f}\n")
            f.write(f"Test RMSE: {metrics['test_rmse']:.2f}\n\n")

    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    try:
        plt.plot(X_test['BUSINESS_DATE'], y_test, label='Actual')
        for name, metrics in results.items():
            plt.plot(X_test['BUSINESS_DATE'], metrics['test_predictions'], label=f'{name} Predicted')
        plt.title('Actual vs Predicted Sales')
        plt.xlabel('Date')
        plt.ylabel('Sales (USD)')
        plt.legend()
        plt.savefig('model_results/actual_vs_predicted.png')
    except Exception as e:
        print(f"Error occurred while plotting actual vs predicted: {str(e)}")
        print("Skipping this plot and continuing with the rest of the script.")
    finally:
        plt.close()

    # Feature importance (for RandomForest only)
    rf_model = load('model_results/RandomForest_model.joblib')
    feature_importance = pd.DataFrame({'feature': X_train.columns.drop('BUSINESS_DATE'), 'importance': rf_model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importance (RandomForest)')
    plt.tight_layout()
    plt.savefig('model_results/feature_importance.png')
    plt.close()

    # Forecast next 3 months using forecasting_2 approach
    print("Generating 3-month forecast using forecasting_2...")
    last_date = X_test['BUSINESS_DATE'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=90)
    future_df = pd.DataFrame({'BUSINESS_DATE': future_dates})
    future_df['DAY_OF_WEEK'] = future_df['BUSINESS_DATE'].dt.dayofweek
    future_df['MONTH'] = future_df['BUSINESS_DATE'].dt.month
    future_df['YEAR'] = future_df['BUSINESS_DATE'].dt.year
    future_df['IS_WEEKEND'] = future_df['DAY_OF_WEEK'].isin([5, 6]).astype(int)

    # Prepare future features
    store_keys = X_train['STORE_KEY'].unique()
    future_features = []

    for store in store_keys:
        store_future = future_df.copy()
        store_future['STORE_KEY'] = store
        store_future['DAYS_SINCE_STORE_OPEN'] = (future_dates - pd.Timestamp.now()).days + X_train[X_train['STORE_KEY'] == store]['DAYS_SINCE_STORE_OPEN'].iloc[0]
        future_features.append(store_future[features[:-1]])  # Exclude 'BUSINESS_DATE'

    future_features = pd.concat(future_features, ignore_index=True)

    # Handle missing values in future features
    future_features_imputed = imputer.transform(future_features)

    # Make predictions using the best model
    best_model_name = min(results, key=lambda x: results[x]['test_mape'])
    best_model = load(f'model_results/{best_model_name}_model.joblib')
    future_predictions = predict_in_batches(best_model, future_features_imputed)

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'BUSINESS_DATE': np.tile(future_dates, len(store_keys)),
        'STORE_KEY': np.repeat(store_keys, 90),
        'PREDICTED_SALES': future_predictions
    })

    # Plot forecast
    plt.figure(figsize=(12, 6))
    for store in forecast_df['STORE_KEY'].unique()[:5]:  # Plot only first 5 stores for clarity
        store_data = forecast_df[forecast_df['STORE_KEY'] == store]
        plt.plot(store_data['BUSINESS_DATE'], store_data['PREDICTED_SALES'], label=f'Store {store}')
    plt.title('3-Month Sales Forecast (Sample of 5 Stores)')
    plt.xlabel('Date')
    plt.ylabel('Predicted Sales (USD)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('model_results/3_month_forecast.png')
    plt.close()

    # Save forecast
    forecast_df.to_csv('model_results/3_month_forecast.csv', index=False)

    print("Script completed successfully!")

except Exception as e:
    print(f"An error occurred: {str(e)}")
