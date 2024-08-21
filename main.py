import numpy as np
import pandas as pd
import datetime as dt
import functions as fn

def main():
    symbols_list = fn.get_symbols_list()

    end_date = '2024-08-03'
    start_date = pd.to_datetime(end_date) - pd.DateOffset(years = 8)

    df = fn.download_stocks_data(symbols_list, start_date, end_date)

    df['garman_klass_vol'] = df.apply(fn.calculate_gk_volatility, axis = 1)

    df['rsi'] = df.groupby(level = 1)['adj close'].transform(lambda x: fn.calculate_rsi(x, window = 14))

    df['bb_low'] =  df.groupby(level = 1)['adj close'].transform(lambda x: fn.calculate_bbands(np.log1p(x), window = 20)[0])
    df['bb_mid'] = df.groupby(level = 1)['adj close'].transform(lambda x: fn.calculate_bbands(np.log1p(x), window = 20)[1])
    df['bb_high'] = df.groupby(level = 1)['adj close'].transform(lambda x: fn.calculate_bbands(np.log1p(x), window = 20)[2])

    df['atr'] = df.groupby(level = 1, group_keys = False).apply(fn.calculate_atr, window = 14)

    df['macd'] = df.groupby(level = 1, group_keys = False)['adj close'].apply(fn.calculate_macd)

    df['dollar_volume'] = (df['adj close'] * df['volume']) / 1e6

    data = fn.dollar_volume_monthly_aggregation(df)

    data = fn.select_top_liquid_stocks(data)

    data = data.groupby(level = 'ticker', group_keys = False).apply(fn.calculate_returns).dropna()

    factor_data = fn.get_fama_french_factors(data = data, start_date = '2016-01-01')

    betas = fn.calculate_betas(factor_data)

    data = fn.join_betas_data(data, betas)

    data = data.groupby(level = 'date', group_keys = False).apply(fn.kmeans_clustering)

    #fn.plot_all_clusters(data)

    data.drop(columns = 'cluster', axis = 1, inplace = True)

    initial_centroids = fn.initialize_centroids(target_rsi_values = [32.5, 47.5, 56, 72.5], num_features = len(data.columns))
    data = data.groupby(level = 'date', group_keys = False).apply(fn.kmeans_clustering, init = initial_centroids)

    #fn.plot_all_clusters(data)

    fixed_dates = fn.get_cluster_dates(data)

    fresh_data = fn.download_stock_data(data)

    portfolio_df = fn.calculate_monthly_returns(fresh_data, fixed_dates)

    start_date = '2016-01-01'
    end_date = dt.date.today().strftime('%Y-%m-%d')
    portfolio_df = fn.download_and_process_spy_data(start_date, end_date, portfolio_df)

    fn.plot_cumulative_returns(portfolio_df)

if __name__ == "__main__":
    main()