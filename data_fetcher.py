import ccxt
import pandas as pd
import pandas_ta as ta
import os
import numpy as np
from datetime import datetime

def fetch_historical_data(symbol='BTC/USDT', timeframe='1h', since=None, limit=5000, exchange_id='binance'):
    """
    Fetch historical OHLCV data with pagination to ensure limit is met.
    """
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class()
    
    print(f"Fetching {limit} candles for {symbol} ({timeframe}) from {exchange_id}...")
    
    all_ohlcv = []
    current_since = since
    
    while len(all_ohlcv) < limit:
        # Fetch a batch
        remaining = limit - len(all_ohlcv)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, current_since, min(remaining, 1000))
        
        if not ohlcv:
            break
            
        all_ohlcv.extend(ohlcv)
        # Move 'since' to the last candle + 1ms to get the next page
        current_since = ohlcv[-1][0] + 1
        
        if len(ohlcv) < 100: # No more data available
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # 1. Log Returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. Technical Indicators
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # 3. Rolling Volatility
    df['vol_10'] = df['log_return'].rolling(window=10).std()
    df['vol_30'] = df['log_return'].rolling(window=30).std()
    
    # 4. Price Relative to High/Low (30 periods)
    df['high_30'] = df['high'].rolling(window=30).max()
    df['low_30'] = df['low'].rolling(window=30).min()
    df['rel_high_low'] = (df['close'] - df['low_30']) / (df['high_30'] - df['low_30'] + 1e-9)
    
    # 5. Volume Delta
    df['vol_delta'] = df['volume'].diff() / (df['volume'].shift(1) + 1e-9)
    
    # Cleanup
    df.dropna(inplace=True)
    return df

def split_data(df, train_ratio=0.8):
    """
    Split data into non-overlapping train and validation sets (Walk-forward).
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    return train_df, val_df

def save_prepared_data(df, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename} (Shape: {df.shape})")

if __name__ == "__main__":
    data = fetch_historical_data(limit=2000)
    train, val = split_data(data)
    save_prepared_data(train, 'data/train.csv')
    save_prepared_data(val, 'data/val.csv')
