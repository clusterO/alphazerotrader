import numpy as np
import pandas as pd
import yaml
from games.trading.game import TradingGame

def verify_pnl_connection():
    print("\n--- Verifying PnL Connection to Price Movement (Phase 3.2) ---")
    
    # 1. Create mock data where price increases exactly 1%
    mock_data = pd.DataFrame({
        'timestamp': pd.to_datetime([0, 1, 2], unit='s'),
        'open': [100, 100, 101],
        'high': [100, 100, 101],
        'low': [100, 100, 101],
        'close': [100, 100, 101], # 1% move from tick 1 to 2
        'volume': [1000, 1000, 1000],
        'log_return': [0, 0, 0.00995],
        'rsi': [50, 50, 50],
        'atr': [1, 1, 1],
        'vol_10': [0.01, 0.01, 0.01],
        'vol_30': [0.01, 0.01, 0.01],
        'rel_high_low': [0.5, 0.5, 0.5],
        'vol_delta': [0, 0, 0]
    })
    
    config = {
        'trading': {
            'window_size': 1,
            'initial_balance': 1000.0,
            'fee': 0.0, # Disable fees for pure PnL check
            'tanh_scale': 10.0,
            'episode_length_range': [2, 2]
        }
    }
    
    game = TradingGame(mock_data, config)
    state = game.reset()
    game.gameState.current_tick = 1 # Start at tick 1
    
    print(f"Initial Price: {mock_data.iloc[1]['close']}")
    print(f"Initial Balance: {state.portfolio['balance']}")
    
    # Take LONG action
    print("Action: LONG (1)")
    next_state, reward, done, _ = game.step(1)
    
    print(f"Next Price: {mock_data.iloc[2]['close']} (+1.0%)")
    print(f"Next Balance: {next_state.portfolio['balance']:.2f}")
    print(f"Calculated Reward: {reward:.6f}")
    
    # Assertions
    expected_balance = 1000.0 * 1.01
    assert abs(next_state.portfolio['balance'] - expected_balance) < 1e-6, \
        f"PnL Mismatch! Expected {expected_balance}, got {next_state.portfolio['balance']}"
    assert abs(reward - 0.01) < 1e-4, \
        f"Reward Mismatch! Expected 0.01, got {reward}"
        
    print("\n[SUCCESS] PnL calculation is correctly connected to price movement.")

if __name__ == "__main__":
    try:
        verify_pnl_connection()
    except Exception as e:
        print(f"\n[FAILURE] {e}")
