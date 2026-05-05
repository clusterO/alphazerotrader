import numpy as np
import pandas as pd
import yaml
from game import TradingGame

def verify_pnl_connection():
    print("\n--- Verifying PnL Fixes: No Lag & Liquidation ---")
    
    n_rows = 300
    np.random.seed(42)
    noise = np.random.normal(0, 0.01, n_rows)
    close_prices = 100.0 + np.cumsum(noise)
    
    mock_data = pd.DataFrame({
        'timestamp': pd.to_datetime(np.arange(n_rows), unit='s'),
        'open': close_prices,
        'high': close_prices + 0.1,
        'low': close_prices - 0.1,
        'close': close_prices,
        'volume': np.random.randint(1000, 2000, n_rows)
    })
    
    # Set a 1% price increase at index 150
    base_price = mock_data.iloc[150]['close']
    mock_data.iloc[151, mock_data.columns.get_loc('close')] = base_price * 1.01
    
    config = {
        'trading': {
            'window_size': 1,
            'initial_balance': 1000.0,
            'fee': 0.0, 
            'tanh_scale': 10.0,
            'episode_length_range': [200, 200],
            'drawdown_threshold': 0.9,
            'idling_penalty': 0.0
        }
    }
    
    game = TradingGame(mock_data, config)
    game.reset(start_tick=10, end_tick=250)
    
    move_tick = -1
    for i in range(len(game.close_prices) - 1):
        p1 = game.close_prices[i]
        p2 = game.close_prices[i+1]
        if abs((p2 - p1) / p1 - 0.01) < 1e-4:
            move_tick = i
            break
            
    print(f"Found move at tick {move_tick}: {game.close_prices[move_tick]:.4f} -> {game.close_prices[move_tick+1]:.4f}")
    
    # --- TEST 1: NO LAG ---
    print("\n--- TEST 1: No Lag ---")
    game.gameState.current_tick = move_tick
    game.gameState.portfolio['position'] = 0 # Start FLAT
    game.gameState.portfolio['balance'] = 1000.0
    
    print(f"Taking action LONG (1) at move_tick")
    next_state, reward, done, _ = game.step(1)
    
    print(f"New Balance: {next_state.portfolio['balance']:.2f}")
    print(f"Reward: {reward:.6f}")
    
    if abs(reward - 0.01) < 1e-4:
        print("[SUCCESS] NO LAG: Action immediately applied to the move.")
    else:
        print("[FAILURE] LAG still exists or return calculation is wrong.")

    # --- TEST 2: LIQUIDATION ---
    print("\n--- TEST 2: Liquidation ---")
    game.gameState.current_tick = move_tick
    game.gameState.portfolio['position'] = 0
    game.gameState.portfolio['balance'] = 60.0 # Just above 5% threshold (50.0)
    
    # Force a 20% drop for a LONG position
    p_now = game.close_prices[move_tick]
    game.close_prices[move_tick+1] = p_now * 0.8
    
    print(f"Initial Balance: 60.0. Taking LONG (1) into a 20% drop.")
    next_state, reward, done, _ = game.step(1)
    
    print(f"New Balance: {next_state.portfolio['balance']:.2f}")
    print(f"Reward: {reward:.6f}")
    print(f"Done: {done}")
    
    if next_state.portfolio['balance'] == 0.0 and done:
        print("[SUCCESS] Liquidation triggered and episode terminated.")
    else:
        print("[FAILURE] Liquidation did not trigger or balance is wrong.")

if __name__ == "__main__":
    try:
        verify_pnl_connection()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[FAILURE] {e}")
