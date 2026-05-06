import ccxt
import pandas as pd
import pandas_ta as ta
import os
import numpy as np
from datetime import datetime, timedelta
import argparse

class DataManager:
    def __init__(self, exchange_id='binance'):
        self.exchange = getattr(ccxt, exchange_id)()
        self.base_dir = os.path.join('data', 'stable')
        
        # New structure: data/stable/{training, val, backtest}
        self.dirs = {
            'training': os.path.join(self.base_dir, 'training'),
            'val': os.path.join(self.base_dir, 'val'),
            'backtest': os.path.join(self.base_dir, 'backtest')
        }
        
        # Ensure directories exist
        for d in self.dirs.values():
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

    def fetch_ohlcv(self, symbol='BTC/USDT', timeframe='1h', limit=5000, start_date=None, end_date=None):
        """
        Fetch OHLCV data. If start_date (YYYY-MM-DD) is provided, fetch forward.
        If end_date (YYYY-MM-DD) is provided, stop fetching after that date.
        """
        since = None
        end_ts = None
        if start_date:
            since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        
        if end_date:
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        if not since and not end_date:
            # Calculate 'since' based on limit and timeframe for backward fetch
            now = datetime.now()
            # Approximation for pagination
            mins_per_tf = {
                '1m': 1, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '4h': 240, '1d': 1440
            }
            m = mins_per_tf.get(timeframe, 60)
            delta = timedelta(minutes=(limit + 200) * m) # extra buffer
            since = int((now - delta).timestamp() * 1000)

        print(f"Fetching candles for {symbol} ({timeframe}) starting since {datetime.fromtimestamp(since/1000) if since else 'exchange start'}...")
        
        all_ohlcv = []
        current_since = since
        
        while (limit is None or len(all_ohlcv) < limit):
            fetch_limit = 1000
            if limit:
                fetch_limit = min(limit - len(all_ohlcv), 1000)
            
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, current_since, fetch_limit)
                if not ohlcv:
                    break
                
                # Check for end_date
                if end_ts:
                    filtered = [c for c in ohlcv if c[0] <= end_ts]
                    all_ohlcv.extend(filtered)
                    if len(filtered) < len(ohlcv):
                        print(f"  Reached end_date: {end_date}")
                        break
                else:
                    all_ohlcv.extend(ohlcv)
                
                current_since = ohlcv[-1][0] + 1
                print(f"  Received {len(ohlcv)} candles... Total: {len(all_ohlcv)}")
                
                if len(ohlcv) < 100:
                    break
            except Exception as e:
                print(f"Error fetching: {e}")
                break

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def add_features(self, df):
        """
        Feature Engineering (STRICTLY ALIGNED WITH game.py)
        """
        print("Adding technical indicators and regime-aware features...")
        df = df.copy()
        # 1. Log Returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # 2. Technical Indicators
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14) / (df['close'] + 1e-9)
        
        # 3. Rolling Volatility
        df['vol_10'] = df['log_return'].rolling(window=10).std()
        df['vol_30'] = df['log_return'].rolling(window=30).std()
        
        # 4. Price Relative to High/Low (30 periods)
        df['high_30'] = df['high'].rolling(window=30).max()
        df['low_30'] = df['low'].rolling(window=30).min()
        df['rel_high_low'] = (df['close'] - df['low_30']) / (df['high_30'] - df['low_30'] + 1e-9)
        
        # 5. Volume Delta
        df['vol_delta'] = df['volume'].diff() / (df['volume'].shift(1) + 1e-9)

        # --- REGIME AWARE FEATURES ---
        close = df['close']
        returns = df['log_return'] # Using log_return for consistency
        realized_vol = returns.rolling(20).std()

        # 1. Trend direction — normalized by vol so it's transferable across regimes
        df['trend_20'] = close.pct_change(20) / (realized_vol + 1e-8)
        df['trend_50'] = close.pct_change(50) / (realized_vol + 1e-8)

        # 2. Distance from long-term MA — normalized
        df['dist_ma_200'] = (close - close.rolling(200).mean()) / (close.rolling(20).std() + 1e-8)

        # 3. Downside vs upside volatility ratio
        neg_returns = returns.clip(upper=0)
        pos_returns = returns.clip(lower=0)
        df['vol_skew'] = (neg_returns.rolling(20).std() / (pos_returns.rolling(20).std() + 1e-8))

        # 4. Momentum persistence — is the trend accelerating or decelerating?
        df['momentum_5'] = close.pct_change(5) / (realized_vol + 1e-8)
        df['momentum_10'] = close.pct_change(10) / (realized_vol + 1e-8)

        # 5. Volatility regime — is vol expanding (fear) or contracting (calm)?
        vol_ma = realized_vol.rolling(50).mean()
        df['vol_regime'] = (realized_vol - vol_ma) / (vol_ma + 1e-8)
        
        df.dropna(inplace=True)
        return df

    def save_regime(self, df, purpose, regime):
        """
        Saves a dataframe to the specified purpose and regime slot.
        purpose: training, val, backtest
        regime: bull, bear, range
        """
        if purpose not in self.dirs:
            print(f"Error: Invalid purpose '{purpose}'. Must be one of {list(self.dirs.keys())}")
            return
            
        valid_regimes = ['bull', 'bear', 'range']
        if regime not in valid_regimes:
            print(f"Error: Invalid regime '{regime}'. Must be one of {valid_regimes}")
            return
            
        path = os.path.join(self.dirs[purpose], f"{regime}.csv")
        df.to_csv(path, index=False)
        print(f"\n[SUCCESS] Data saved to {path} ({len(df)} rows)")

def main():
    parser = argparse.ArgumentParser(description='Unified Data Manager for AlphaZero Trader')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Instrument symbol')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe (1h, 5m, 1d, etc.)')
    parser.add_argument('--limit', type=int, default=None, help='Number of candles')
    parser.add_argument('--start_date', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--purpose', type=str, choices=['training', 'val', 'backtest'], help='Target folder')
    parser.add_argument('--regime', type=str, choices=['bull', 'bear', 'range'], help='Regime filename')
    
    args = parser.parse_args()
    
    dm = DataManager()
    
    # Set default limits for best system performance
    limit = args.limit
    if limit is None and not args.end_date:
        limit = 5000
        
    df_raw = dm.fetch_ohlcv(args.symbol, args.timeframe, limit, args.start_date, args.end_date)
    df = dm.add_features(df_raw)
    
    if args.purpose and args.regime:
        dm.save_regime(df, args.purpose, args.regime)
    else:
        # Fallback to old behavior for description if not using purpose/regime
        sym_clean = args.symbol.replace('/', '_')
        filename = f"{sym_clean}_{args.timeframe}_raw_{len(df)}.csv"
        path = os.path.join(dm.dirs['backtest'], filename)
        df.to_csv(path, index=False)
        print(f"\n[INFO] No purpose/regime specified. Saved raw data to {path}")

if __name__ == "__main__":
    main()
