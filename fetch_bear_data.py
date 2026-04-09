import ccxt
import pandas as pd
import pandas_ta as ta
import os
import numpy as np
from datetime import datetime

def fetch_historical_data(symbol='BTC/USDT', timeframe='1h', since=None, limit=3000, exchange_id='binance'):
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class()
    print(f"Fetching {limit} candles for {symbol} ({timeframe}) starting from {since}...")
    
    all_ohlcv = []
    current_since = since
    while len(all_ohlcv) < limit:
        remaining = limit - len(all_ohlcv)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, current_since, min(remaining, 1000))
        if not ohlcv: break
        all_ohlcv.extend(ohlcv)
        current_since = ohlcv[-1][0] + 1
        if len(ohlcv) < 100: break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Feature Engineering (MUST MATCH MAIN PIPELINE)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['vol_10'] = df['log_return'].rolling(window=10).std()
    df['vol_30'] = df['log_return'].rolling(window=30).std()
    df['high_30'] = df['high'].rolling(window=30).max()
    df['low_30'] = df['low'].rolling(window=30).min()
    df['rel_high_low'] = (df['close'] - df['low_30']) / (df['high_30'] - df['low_30'] + 1e-9)
    df['vol_delta'] = df['volume'].diff() / (df['volume'].shift(1) + 1e-9)
    
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    # Start: Jan 1, 2022 (The start of the 2022 Bear Market)
    start_ts = int(datetime(2022, 1, 1, 0, 0).timestamp() * 1000)
    
    bear_data = fetch_historical_data(since=start_ts, limit=3000)
    bear_data.to_csv('data/bear_2022.csv', index=False)
    print(f"\n[SUCCESS] 2022 BEAR MARKET DATA SAVED TO data/bear_2022.csv")
    print(f"Total Candles: {len(bear_data)}")
    print(f"Date Range:    {bear_data['timestamp'].iloc[0]} to {bear_data['timestamp'].iloc[-1]}")
