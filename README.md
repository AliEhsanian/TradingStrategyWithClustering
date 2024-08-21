# Trading Strategy with Clustering Stocks Data

This repository contains a Jupyter Notebook outlining a trading strategy based on clustering stock data and optimizing portfolio allocations. The strategy leverages clustering to categorize similar assets and applies portfolio optimization techniques to maximize returns.

## Requirements

Ensure that you have installed the Python packages listed in the `environment.yml` file.

## Project Structure

### Data Collection and Preprocessing

#### Downloading Data
- Retrieves S&P 500 stock symbols from Wikipedia.
- Downloads historical stock data from Yahoo Finance using the symbols.

#### Feature Calculation and Technical Indicators
- **Garman-Klass Volatility**: Estimates historical volatility considering open, close, high, and low prices.
- **Relative Strength Index (RSI)**: Identifies overbought or oversold conditions.
- **Bollinger Bands**: Determines market volatility and overbought/oversold conditions.
- **Average True Range (ATR)**: Measures market volatility.
- **Moving Average Convergence Divergence (MACD)**: Tracks momentum and trend following.
- **Dollar Volume**: Calculates the total value traded for each stock.

### Monthly Data Processing
- Aggregates daily data into month-end frequencies.
- Selects the top 150 most liquid stocks based on a 5-year rolling average of dollar volume.

### Feature Generation
- Calculates monthly returns across various time periods using `.pct_change(lag)` to analyze time series momentum.

### Download Fama-French Factors and Calculate Rolling Factor Betas
- Uses Fama-French data to assess asset exposure to common risk factors using rolling linear regression.

### Monthly K-Means Clustering
- Groups similar assets using K-Means clustering with optimal initial centroids to enhance clustering effectiveness.

### Monthly Asset Selection and Portfolio Optimization
- Selects stocks from specific clusters based on momentum and optimizes portfolio weights using the EfficientFrontier optimizer.
- Applies constraints on stock weights to ensure portfolio diversification.
- Visualizes monthly returns and compares them with the S&P 500 index.

## Usage

- Open the Jupyter Notebook (`ClusteringStocksTrading.ipynb`) and execute the cells sequentially to replicate the analysis and view the results.
