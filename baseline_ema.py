import pandas as pd
import numpy as np
import pandas_ta as ta

def run_ema_baseline(data_path='data/val.csv', fast=9, slow=21):
    print(f"--- EMA CROSSOVER BASELINE ({fast}/{slow}) ---")
    df = pd.read_csv(data_path)
    
    # 1. Indicators
    df['ema_fast'] = ta.ema(df['close'], length=fast)
    df['ema_slow'] = ta.ema(df['close'], length=slow)
    df.dropna(inplace=True)
    
    # 2. Strategy Logic
    # 1: Long (fast > slow), 0: Flat
    df['signal'] = np.where(df['ema_fast'] > df['ema_slow'], 1, 0)
    
    # 3. PnL Calculation
    # We shift signal by 1 because we enter at the NEXT candle open
    df['pnl_pct'] = df['close'].pct_change().fillna(0)
    df['strategy_returns'] = df['signal'].shift(1) * df['pnl_pct']
    
    # Subtract fees (0.0005) only on signal changes
    df['trades'] = df['signal'].diff().abs().fillna(0)
    df['strategy_returns'] -= df['trades'] * 0.0005
    
    # Cumulative
    df['cum_return'] = (1 + df['strategy_returns']).cumprod()
    
    initial_bal = 1000.0
    final_bal = initial_bal * df['cum_return'].iloc[-1]
    total_return = (final_bal / initial_bal) - 1
    
    # Annualized Sharpe
    mu = df['strategy_returns'].mean()
    sigma = df['strategy_returns'].std()
    sharpe = (mu / (sigma + 1e-9)) * np.sqrt(252 * 24)
    
    print(f"Data: {data_path}")
    print(f"Initial Balance: {initial_bal:.2f}")
    print(f"Final Balance:   {final_bal:.2f}")
    print(f"Total Return:    {total_return*100:.2f}%")
    print(f"Sharpe (ANN):    {sharpe:.4f}")
    
    return total_return, sharpe

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/test.csv'
    run_ema_baseline(path)
