# Sales Forecasting Dashboard

This project provides a dashboard for store-wise and overall sales forecasting using Prophet. The dashboard is built with Dash and Plotly, and forecasts sales for the next 90 days.

## Table of Contents

- [Introduction](#introduction)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Approach](#approach)
- [Model Validation](#model-validation)
## Introduction

The Sales Forecasting Dashboard provides insights into historical sales trends and forecasts future sales for individual stores and overall sales. The model incorporates US holidays to improve forecast accuracy.

## Data

- `sales.csv`: Contains sales data with columns `BUSINESS_DATE`, `STORE_KEY`, and `NET_SALES_FINAL_USD_AMOUNT`.
- `store.csv`: Contains store information with columns `STORE_KEY` and `STORE_NAME`.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/fibonacci35813/TimeSeriesForecasting.git
    cd TimeSeriesForecasting
    ```

## Usage

1. Place your `sales.csv` and `store.csv` files in the project directory.

2. Run the Dash app:
    ```sh
    python app.py
    ```

## Approach

### Data Exploration

Sales data is aggregated by date, and holiday information is added. The store names are mapped using an additional dataset.

### Model Selection

Prophet is chosen for its ability to handle seasonality and holidays. The model is trained on 80% of the data, and the remaining 20% is used for validation.

### Feature Importance

US holidays are added as an external regressor, given their significant impact on sales patterns.

### Insights

The model provides a 90-day sales forecast. Forecast accuracy is measured using Mean Absolute Percentage Error (MAPE) and Median Absolute Percentage Error (MdAPE).

### Data Flow and Architecture

- **Data Joins**: Sales data is joined with store information using `STORE_KEY`, and holiday data is joined based on dates.
- **Architecture**: A Dash app visualizes historical sales trends and forecasts for individual stores and overall sales.
- **Model Complexity**: Prophet’s complexity lies in its decomposition approach, handling yearly, weekly, and holiday effects.

## Model Validation

To validate the model, we use the following metrics:

- **Mean Absolute Percentage Error (MAPE)**: Measures the average percentage error.
- **Median Absolute Percentage Error (MdAPE)**: Measures the median percentage error.
- **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.
