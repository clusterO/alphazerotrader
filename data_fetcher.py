import ccxt
import pandas as pd
import os
from datetime import datetime

def fetch_historical_data(symbol='BTC/USDT', timeframe='1h', since=None, limit=1000, exchange_id='binance'):
    """
    Fetch historical OHLCV data from an exchange using CCXT.
    """
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class()
    
    print(f"Fetching {limit} candles for {symbol} ({timeframe}) from {exchange_id}...")
    
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    return df

def save_to_csv(df, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    # Example usage
    data = fetch_historical_data(symbol='BTC/USDT', timeframe='1h', limit=500)
    save_to_csv(data, 'data/btc_usdt_1h.csv')
