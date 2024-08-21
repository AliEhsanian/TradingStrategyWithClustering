import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import yfinance as yf
import talib
import pandas_datareader.data as web
from sklearn.cluster import KMeans
from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier


def get_symbols_list(url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'):
    """
    Fetch and return a list of S&P 500 company symbols from Wikipedia.

    Parameters:
    url (str): URL to fetch the S&P 500 list from. Defaults to Wikipedia's S&P 500 page.

    Returns:
    list: A list of S&P 500 company symbols formatted for yfinance.
    """
    try:
        # Fetch the table from the provided URL
        sp500 = pd.read_html(url)[0]

        # Ensure the 'Symbol' column exists
        if 'Symbol' not in sp500.columns:
            raise ValueError("The 'Symbol' column is missing from the fetched data.")

        # Replace '.' with '-' in symbols for yfinance compatibility
        sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')

        # Get unique symbols and convert to list
        symbols_list = sp500['Symbol'].unique().tolist()

        return symbols_list

    except Exception as e:
        raise RuntimeError(f"Error fetching S&P 500 symbols: {e}")

def download_stocks_data(symbols_list, start_date, end_date, chunk_size = 100):
    """
    Download stock data in chunks to avoid overloading the API.

    Parameters:
    symbols_list (list): List of stock symbols to fetch.
    start_date (str or pd.Timestamp): Start date for data download.
    end_date (str or pd.Timestamp): End date for data download.
    chunk_size (int): Number of symbols to fetch per chunk.

    Returns:
    pd.DataFrame: Combined DataFrame with stock data.
    """
    try:
        all_data = []

        # Iterate over chunks of symbols
        for i in range(0, len(symbols_list), chunk_size):
            chunk = symbols_list[i: i + chunk_size]

            try:
                # Download data for the current chunk
                data = yf.download(tickers = chunk, start = start_date, end = end_date).stack(future_stack = True)
                data.index.names = ['date', 'ticker']
                data.columns = data.columns.str.lower()
                all_data.append(data)
            except Exception as e:
                raise RuntimeError(f"Error fetching data for chunk {chunk}: {e}")

        # Combine all chunks into a single DataFrame
        if all_data:
            df = pd.concat(all_data)
        else:
            raise RuntimeError("No data was fetched. Please check your symbols list and dates.")

        return df

    except Exception as e:
        raise RuntimeError(f"Error: {e}")


# Garman-Klass Volatility Calculation
def calculate_gk_volatility(df):
    """
    Calculate Garman-Klass volatility for a given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'high', 'low', 'adj close', and 'open' columns.

    Returns:
    pd.Series: Series with the Garman-Klass volatility.
    """
    try:
        # Ensure necessary columns are present
        #required_columns = ['high', 'low', 'adj close', 'open']
        #if not all(col in df.columns for col in required_columns):
        #    raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

        # Calculate Garman-Klass volatility
        return (((np.log(df['high']) - np.log(df['low'])) ** 2) / 2 - 
                (2 * np.log(2) - 1) * ((np.log(df['adj close']) - np.log(df['open'])) ** 2))

    except Exception as e:
        raise RuntimeError(f"Error calculating Garman-Klass volatility: {e}")


# RSI Calculation
def calculate_rsi(series, window = 14):
    """
    Calculate the Relative Strength Index (RSI) for a given series.

    Parameters:
    series (pd.Series): Price series to calculate RSI.
    window (int): The window size for RSI calculation.

    Returns:
    pd.Series: RSI values.
    """
    try:
        # Calculate RSI using TA-Lib
        return talib.RSI(series, timeperiod = window)
    except Exception as e:
        raise RuntimeError(f"Error calculating RSI: {e}")


# Bollinger Bands Calculation
def calculate_bbands(series, window = 20):
    """
    Calculate the Bollinger Bands for a given series.

    Parameters:
    series (pd.Series): Price series to calculate Bollinger Bands.
    window (int): The window size for Bollinger Bands calculation.

    Returns:
    tuple(pd.Series, pd.Series, pd.Series): Lower band, middle band, and upper band.
    """
    try:
        # Calculate Bollinger Bands using TA-Lib
        upper, middle, lower = talib.BBANDS(series, timeperiod = window, nbdevup = 2, nbdevdn = 2, matype = 0)
        return lower, middle, upper
    except Exception as e:
        raise RuntimeError(f"An error occurred while calculating Bollinger Bands: {e}")


# ATR Calculation
def calculate_atr(dframe, window = 14):
    """
    Calculate the Average True Range for a given DataFrame.

    Parameters:
    dframe (pd.DataFrame): DataFrame containing the 'high', 'low', and 'close' columns.
    window (int): The window size for ATR calculation.

    Returns:
    pd.Series: ATR values.
    """
    # Check for required columns
    required_columns = ['high', 'low', 'close']
    if not all(col in dframe.columns for col in required_columns):
        raise ValueError(f"DataFrame is missing required columns: {required_columns}")

    # Check that columns contain numeric data
    try:
        # Calculate ATR using TA-Lib
        atr = talib.ATR(dframe['high'], dframe['low'], dframe['close'], timeperiod = window)

        return (atr - atr.mean()) / atr.std()
    except Exception as e:
        raise RuntimeError(f"An error occurred while calculating ATR: {e}")


# MACD Calculation
def calculate_macd(close, fastperiod = 12, slowperiod = 26, signalperiod = 9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) for a given DataFrame.

    Parameters:
    close (pd.Series): DataFrame containing the 'close' column.
    fastperiod (int, optional): The fast period for MACD calculation. Defaults to 12.
    slowperiod (int, optional): The slow period for MACD calculation. Defaults to 26.
    signalperiod (int, optional): The signal period for MACD calculation. Defaults to 9.

    Returns:
    pd.Series: MACD values.
    """
    try:
        # Calculate MACD using TA-Lib
        macd, macdsignal, macdhist = talib.MACD(close, fastperiod = fastperiod, slowperiod = slowperiod, signalperiod = signalperiod)

        return (macd - macd.mean()) / macd.std()
    except Exception as e:
        raise RuntimeError(f"An error occurred while calculating MACD: {e}")


def dollar_volume_monthly_aggregation(df):
    """
    Aggregate monthly dollar volume data to reflect month-end frequencies.

    Parameters:
    df (pd.DataFrame): DataFrame containing the 'dollar_volume' column.

    Returns:
    pd.DataFrame: DataFrame containing the monthly dollar volume data.
    """
    try:
        # Validate that 'ticker' is in the index and 'dollar_volume' is a column
        if 'ticker' not in df.index.names:
            raise ValueError("'ticker' must be in the DataFrame index.")
        if 'dollar_volume' not in df.columns:
            raise ValueError("'dollar_volume' must be a column in the DataFrame.")

        # Define columns to exclude
        exclude_cols = {'dollar_volume', 'volume', 'open', 'high', 'low', 'close'}

        # Filter columns to keep
        indicator_cols = [c for c in df.columns if c not in exclude_cols]

        # Check if there are columns left after exclusion
        if not indicator_cols:
            raise ValueError("No columns left after exclusion.")

        # Resample and process 'dollar_volume'
        dollar_volume_resampled = df.unstack('ticker')['dollar_volume'].resample('ME').mean().stack('ticker', future_stack = True).to_frame('dollar_volume')

        # Resample and process remaining columns
        remaining_data_resampled = df.unstack('ticker')[indicator_cols].resample('ME').last().stack('ticker', future_stack = True)

        # Concatenate dataframes and drop NA values
        data = pd.concat([dollar_volume_resampled, remaining_data_resampled], axis=1).dropna()

        return data

    except (ValueError, KeyError) as e:
        raise RuntimeError(f"An error occurred during data processing: {e}")


def select_top_liquid_stocks(data):
    """
    Select the 150 most liquid stocks based on 5-year rolling average of dollar volume.

    Parameters:
    data (pd.DataFrame): DataFrame containing the monthly dollar volume data.

    Returns:
    pd.DataFrame: DataFrame containing the 150 most liquid stocks.
    """
    try:
        # Validate required columns
        if 'dollar_volume' not in data.columns:
            raise ValueError("'dollar_volume' must be a column in the DataFrame.")

        if 'ticker' not in data.index.names:
            raise ValueError("'ticker' must be in the DataFrame index.")

        if 'date' not in data.index.names:
            raise ValueError("'date' must be in the DataFrame index.")

        # Perform rolling mean, ranking, filtering, and column dropping
        data = (
            data
            .assign(
                dollar_volume = lambda df: df['dollar_volume']
                .unstack(level = 'ticker')
                .rolling(window = 5*12, min_periods = 12)
                .mean()
                .stack(level = 'ticker')
                )
                .assign(
                    dollar_volume_rank = lambda df: df
                    .groupby(level = 'date')['dollar_volume']
                    .rank(ascending = False)
                    )
                    .query('dollar_volume_rank < 150')
                    .drop(columns = ['dollar_volume', 'dollar_volume_rank'], errors = 'ignore')
            )

        return data

    except (KeyError, ValueError) as e:
        raise RuntimeError(f"An error occurred during data processing: {e}")


def calculate_returns(df):
    """
    Calculate returns for various lags and clip outliers.

    Parameters:
    df (pd.DataFrame): DataFrame with 'adj close' column.

    Returns:
    pd.DataFrame: DataFrame with additional return columns for different lags.
    """
    try:
        # Validate presence of 'adj close' column
        if 'adj close' not in df.columns:
            raise ValueError("'adj close' column is missing from the DataFrame.")

        outlier_cutoff = 0.005
        lags = [1, 2, 3, 6, 9, 12]

        # Function to calculate and clip returns
        def calculate_return(lag):
            return (
                df['adj close']
                .pct_change(lag)
                .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                       upper=x.quantile(1 - outlier_cutoff)))
                .add(1)
                .pow(1 / lag)
                .sub(1)
            )

        # Calculate returns for each lag and add to DataFrame
        for lag in lags:
            df[f'return_{lag}m'] = calculate_return(lag)

        return df

    except ValueError as e:
        raise RuntimeError(f"An error occurred during data processing: {e}")


def get_fama_french_factors(data, start_date):
    """
    Fetch and process Fama-French factor data.

    Parameters:
    data (pd.DataFrame): DataFrame containing stock return data.
    start_date (str): Start date for fetching data.

    Returns:
    pd.DataFrame: Processed factor data joined with stock returns.
    """
    try:
        # Fetch Fama-French factor data
        factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start = start_date)[0]

        # Drop the 'RF' column
        if 'RF' not in factor_data.columns:
            raise KeyError("'RF' column is missing from the fetched factor data.")

        factor_data = factor_data.drop('RF', axis = 1)


        # Convert index to timestamp and set the index name
        factor_data.index = factor_data.index.to_timestamp()
        factor_data.index.name = 'date'

        # Resample data to end-of-month frequency, adjust units, and join with stock returns
        if 'return_1m' not in data.columns:
            raise KeyError("'return_1m' column is missing from the 'data' DataFrame.")

        factor_data = (
            factor_data
            .resample('ME')
            .last()
            .div(100)
            .join(data[['return_1m']], how = 'inner')
            .sort_index()
            )

        # Filter out stocks with less than 10 months of data
        if 'ticker' not in data.index.names:
            raise ValueError("'ticker' must be an index level in the stock return data.")

        observations = factor_data.groupby(level = 'ticker').size()
        valid_stocks = observations[observations >= 10]

        # Filter factor data based on valid stocks
        factor_data = factor_data[factor_data.index.get_level_values(level = 'ticker').isin(valid_stocks.index)]

        return factor_data

    except KeyError as e:
        raise KeyError(f"Data error: {e}")
    except Exception as e:
        raise RuntimeError(f"An error occurred during data processing: {e}")


def calculate_betas(factor_data):
    """
    Calculate rolling betas for each ticker using RollingOLS.

    Parameters:
    factor_data (pd.DataFrame): DataFrame with factor and return data.

    Returns:
    pd.DataFrame: DataFrame with rolling betas.
    """
    try:
        # Ensure required columns are present
        if 'return_1m' not in factor_data.columns:
            raise ValueError("'return_1m' column is missing from the factor data.")

        # Calculate rolling betas for each ticker
        def compute_rolling_betas(df):
            # Determine window size
            window_size = min(24, len(df))
            # Fit RollingOLS model
            model = RollingOLS(endog = df['return_1m'],
                               exog = sm.add_constant(df.drop(columns = 'return_1m')),
                               window = window_size,
                               min_nobs = len(df.columns) + 1)

            betas = model.fit(params_only = True).params.drop('const', axis = 1)

            return betas

        # Apply the function to each ticker group
        betas = factor_data.groupby(level = 'ticker', group_keys = False).apply(compute_rolling_betas)
        return betas

    except KeyError as e:
        raise KeyError(f"DataFrame column error: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


def join_betas_data(data, betas):
    """
    Process and join betas with the main data, fill missing values, and clean the data.

    Parameters:
    data (pd.DataFrame): DataFrame with stock returns and factors.
    betas (pd.DataFrame): DataFrame with rolling betas.

    Returns:
    pd.DataFrame: Processed data with betas and cleaned factors.
    """
    try:
        # Join rolling betas with the main data
        data = data.join(betas.groupby(level='ticker').shift())

        # Define the list of factors
        factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

        # Ensure the factors exist in the DataFrame
        if not all(factor in data.columns for factor in factors):
            raise ValueError("One or more required factors are missing from the data.")

        # Fill missing values in factors with the mean of the respective factor
        data[factors] = data.groupby(level = 'ticker', group_keys = False)[factors].apply(lambda x: x.fillna(x.mean()))

        # Drop unnecessary columns and rows with missing values
        if 'adj close' in data.columns:
            data.drop(columns = 'adj close', inplace = True)
        data.dropna(inplace = True)

        return data

    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


def kmeans_clustering(df, n_clusters = 4, init = 'random'):
    """
    Apply KMeans clustering to the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame to cluster.
    n_clusters (int): Number of clusters for KMeans.
    init (str): Method for initialization. Defaults to 'random'.

    Returns:
    pd.DataFrame: DataFrame with a new 'cluster' column.
    """
    try:
        # Ensure the DataFrame is not empty and contains more rows than clusters
        if df.empty or len(df) <= n_clusters:
            raise ValueError("DataFrame is empty or contains fewer rows than the number of clusters.")

        # Select numerical features for clustering
        features = df.select_dtypes(include=['float64', 'int64'])
        if features.empty:
            raise ValueError("No numerical features available for clustering.")

        # Fit KMeans
        df['cluster'] = KMeans(n_clusters = n_clusters, random_state = 0, init = init).fit_predict(df)

        return df

    except Exception as e:
        raise RuntimeError(f"Error during KMeans clustering: {e}")


def plot_clusters(ax, data, title):
    """
    Plot clusters on a given axis.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to plot on.
    data (pd.DataFrame): DataFrame with cluster labels.
    title (str): Title for the plot.
    """

    try:
        # Check if 'cluster' column exists
        if 'cluster' not in data.columns:
            raise ValueError("The DataFrame must contain a 'cluster' column.")

        # Get unique clusters
        clusters = data['cluster'].unique()
        clusters.sort()
        colors = ['red', 'green', 'blue', 'black']

        for cluster in clusters:
            cluster_data = data[data['cluster'] == cluster]
            ax.scatter(cluster_data.iloc[:, 5], cluster_data.iloc[:, 1],
                       color = colors[cluster % len(colors)],
                       label = f'cluster {cluster}')

        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    except Exception as e:
        raise RuntimeError(f'Error during plotting: {e}')


def plot_all_clusters(data):
    """
    Plot clusters for multiple dates in a grid of 10 rows x 3 columns.

    Parameters:
    data (pd.DataFrame): DataFrame with cluster labels and dates as index.
    """
    plt.style.use('ggplot')

    unique_dates = data.index.get_level_values('date').unique().tolist()
    num_dates = len(unique_dates)
    num_rows = 10
    num_cols = 3

    fig, axes = plt.subplots(num_rows, num_cols, figsize = (15, 45))

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    for i, date in enumerate(unique_dates):
        if i >= len(axes):
            break  # Avoid plotting more than the available axes

        g = data.xs(date, level = 'date', drop_level = False)
        plot_clusters(axes[i], g, f'Date {date}')

    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def initialize_centroids(target_rsi_values, num_features):
    """
    Initialize centroids for KMeans based on target RSI values.

    Parameters:
    target_rsi_values (list): List of target RSI values for centroid initialization.
    num_features (int): Number of features in the data.

    Returns:
    np.ndarray: Array of initial centroids.
    """
    centroids = np.zeros((len(target_rsi_values), num_features))
    centroids[:, 1] = target_rsi_values
    return centroids


def get_cluster_dates(data, target_cluster = 3):
    """
    Process the DataFrame to filter by a specific cluster and adjust date indexing.

    Parameters:
    data (pd.DataFrame): DataFrame containing cluster information.
    target_cluster (int): The cluster to filter by.

    Returns:
    dict: Dictionary with dates as keys and lists of tickers as values.
    """
    try:
        # Filter DataFrame by target cluster and adjust indexing
        filter_df = data[data['cluster'] == target_cluster].copy()
        if filter_df.empty:
            raise ValueError(f"No data found for cluster {target_cluster}.")

        filter_df = filter_df.reset_index(level = 'ticker')
        filter_df.index = filter_df.index + pd.DateOffset(1)
        filter_df = filter_df.reset_index().set_index(['date', 'ticker'])

        # Get unique dates and prepare the dictionary with fixed dates
        dates = filter_df.index.get_level_values('date').unique()
        fixed_dates = {d.strftime('%Y-%m-%d'): filter_df.xs(d, level = 'date').index.tolist() for d in dates}

        return fixed_dates

    except Exception as e:
        raise RuntimeError(f"Error processing filtered data: {e}")


def optimize_weights(prices, lower_bound = 0, upper_bound = 0.1):
    """
    Optimize portfolio weights to maximize the Sharpe ratio.

    Parameters:
    prices (pd.DataFrame): Historical price data for assets.
    lower_bound (float): Lower bound for weights (default is 0).
    upper_bound (float): Upper bound for weights (default is 0.1).

    Returns:
    dict: Dictionary of optimized asset weights.
    """
    try:
        # Ensure prices DataFrame is not empty
        if prices.empty:
            raise ValueError("The prices DataFrame is empty.")

        # Calculate expected returns and covariance matrix
        returns = expected_returns.mean_historical_return(prices = prices, frequency = 252)
        cov_matrix = risk_models.sample_cov(prices = prices, frequency = 252)

        # Initialize Efficient Frontier with bounds
        ef = EfficientFrontier(expected_returns = returns, cov_matrix = cov_matrix, 
                               weight_bounds = (lower_bound, upper_bound), 
                               solver='SCS')

        # Optimize for maximum Sharpe ratio
        optimal_weights = ef.max_sharpe()

        # Clean and return the weights
        return ef.clean_weights()

    except Exception as e:
        raise RuntimeError(f"Error optimizing weights: {e}")


# Download Fresh Daily Prices Data only for short listed stocks.
def download_stock_data(data):
    """
    Download fresh daily price data for a list of stocks based on the provided DataFrame.

    Parameters:
    data (pd.DataFrame): DataFrame containing stock tickers and dates.

    Returns:
    pd.DataFrame: DataFrame with the downloaded stock price data.
    """
    try:
        # Ensure 'ticker' and 'date' are available and non-empty
        if 'ticker' not in data.index.names or 'date' not in data.index.names:
            raise ValueError("The DataFrame must have 'ticker' and 'date' as index levels.")

        # Extract unique tickers and date range
        tickers = data.index.get_level_values('ticker').unique().tolist()
        dates = data.index.get_level_values('date').unique()

        if not tickers or dates.empty:
            raise ValueError("No 'tickers' or 'dates' found in the DataFrame.")

        start_date = dates.min() - pd.DateOffset(months = 12)
        end_date = dates.max()

        # Download stock price data
        new_df = yf.download(tickers = tickers, start = start_date, end = end_date)

        # Check if download was successful
        if new_df.empty:
            raise ValueError("The downloaded DataFrame is empty. Please check the ticker symbols and date range.")

        return new_df

    except Exception as e:
        raise RuntimeError(f"Error downloading stock data: {e}")


# Calculate daily returns for each potential stock in our portfolio.
# At the start of each month, use the function to select stocks and calculate their optimal weights for the upcoming month.
# In cases where the maximum Sharpe ratio optimization fails, default to equally-weighted weights for that month.

def calculate_monthly_returns(fresh_data, fixed_dates):
    """
    Calculate the monthly returns for each potential stock in the portfolio and optimize weights.

    Parameters:
    fresh_data (pd.DataFrame): Historical stock price data.
    fixed_dates (dict): Dictionary with dates as keys and lists of tickers as values.

    Returns:
    pd.DataFrame: DataFrame with portfolio returns for each month.
    """
    try:
        # Calculate daily log returns
        returns_dataframe = np.log(fresh_data['Adj Close']).diff()

        # Prepare an empty DataFrame to store the portfolio returns
        portfolio_df = pd.DataFrame()

        # Iterate over each start date in fixed_dates
        for start_date in fixed_dates.keys():

            if pd.to_datetime(start_date) > fresh_data.index.get_level_values('Date').max():
                break;
            try:
                end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
                optimization_start_date = (pd.to_datetime(start_date) - pd.DateOffset(months = 12)).strftime('%Y-%m-%d')
                optimization_end_date = (pd.to_datetime(start_date) - pd.DateOffset(days = 1)).strftime('%Y-%m-%d')

                # Extract the data for optimization
                cols = fixed_dates[start_date]
                optimization_df = fresh_data[optimization_start_date: optimization_end_date]['Adj Close'][cols]

                # Optimize weights and handle exceptions
                success = False
                try:
                    weights = optimize_weights(prices = optimization_df, 
                                               lower_bound = round(1 / (len(optimization_df.columns) * 2), 3))
                    weights = pd.DataFrame(weights, index = pd.Series(0))
                    success = True
                except Exception as e:
                    print(f'Max Sharpe Optimization failed for {start_date}, Continuing with Equal-Weights')
                if not success:
                    # If optimization failed, use equal weights
                    weights = pd.DataFrame([1 / len(optimization_df.columns) for i in range(len(optimization_df.columns))], 
                                        index = optimization_df.columns.tolist(), 
                                        columns = pd.Series(0)).T

                # Calculate returns for the current month
                temp_df = returns_dataframe[start_date: end_date]
                temp_df = (temp_df
                        .stack()
                        .to_frame('return')
                        .reset_index(level = 0)
                        .merge(weights.stack()
                                .to_frame('weight')
                                .reset_index(level = 0, drop = True), left_index = True, right_index = True)
                        .reset_index()
                        .set_index(['Date', 'Ticker'])
                        .unstack()
                        .stack(future_stack = True)
                        )
                temp_df.index.names = ['date', 'ticker']
                temp_df['weighted_return'] = temp_df['return'] * temp_df['weight']
                temp_df = temp_df.groupby(level = 0)['weighted_return'].sum().to_frame('Strategy Return')
                portfolio_df = pd.concat([portfolio_df, temp_df], axis = 0)
            except Exception as e:
                raise RuntimeError(f'Error in {start_date}: {e}')

        return portfolio_df.drop_duplicates()

    except Exception as e:
        raise RuntimeError(f"Error in calculate_monthly_returns: {e}")


def download_and_process_spy_data(start_date, end_date, portfolio_df):
    """
    Download SPY data, calculate returns, and merge with the portfolio DataFrame.

    Parameters:
    start_date (str): Start date for downloading SPY data.
    end_date (str): End date for downloading SPY data.
    portfolio_df (pd.DataFrame): Existing portfolio DataFrame to merge with.

    Returns:
    pd.DataFrame: Updated portfolio DataFrame with SPY returns.
    """
    try:
        # Download SPY data
        SPY500 = yf.download(tickers = 'SPY', start = start_date, end = end_date)

        if SPY500.empty:
            raise ValueError("Downloaded SPY data is empty. Check the ticker symbol and date range.")

        # Calculate daily returns and rename column
        SPY500_returns = np.log(SPY500[['Adj Close']]).diff().dropna().rename(columns = {'Adj Close': 'S&P500 Buy&Hold'})

        # Merge with portfolio_df
        updated_portfolio_df = portfolio_df.merge(SPY500_returns, left_index = True, right_index = True)

        return updated_portfolio_df

    except Exception as e:
        raise RuntimeError(f"Error processing SPY data: {e}")


def plot_cumulative_returns(portfolio_df, end_date = '2024-08-02'):
    """
    Plot the cumulative returns of the optimized portfolio and compare with the S&P 500.

    Parameters:
    portfolio_df (pd.DataFrame): DataFrame containing portfolio returns.
    end_date (str): End date for plotting the cumulative returns.
    """
    try:
        # Check if the DataFrame contains required data
        if portfolio_df.empty:
            raise ValueError("The portfolio DataFrame is empty. Cannot plot cumulative returns.")

        # Calculate cumulative returns
        portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum()) - 1

        # Filter data up to the specified end date
        end_date = pd.to_datetime(end_date)
        filtered_data = portfolio_cumulative_return.loc[: end_date]

        # Plotting
        plt.style.use('ggplot')
        plt.figure(figsize = (21, 9))
        filtered_data.plot(figsize = (21, 9))

        plt.title('Cumulative Returns of Optimized Portfolio vs. S&P 500')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1.0))
        plt.grid(True)
        plt.legend(['Optimized Portfolio', 'S&P 500'], loc = 'best')
        #plt.tight_layout()

        plt.show()

    except Exception as e:
        raise RuntimeError(f"Error plotting cumulative returns: {e}")


