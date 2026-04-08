import numpy as np
import pandas as pd
import yaml
import pickle
import os
from games.trading.game import TradingGame

def analyze_existing_memory(memory_path):
    if not os.path.exists(memory_path):
        print("\n[!] No existing memory file found to analyze.")
        return
    
    print(f"\n--- Analyzing Existing Memory: {memory_path} ---")
    with open(memory_path, "rb") as f:
        memory = pickle.load(f)
    
    ltm = memory.ltmemory
    if len(ltm) == 0:
        print("Memory is empty.")
        return
        
    z_values = [mem['value'] for mem in ltm]
    print(f"Sample size: {len(z_values)}")
    print(f"z-values: mean={np.mean(z_values):.4f}, std={np.std(z_values):.4f}, "
          f"min={np.min(z_values):.4f}, max={np.max(z_values):.4f}")
    
    # Check action distribution from MCTS Search (AV)
    av_values = np.array([mem['AV'] for mem in ltm])
    avg_av = np.mean(av_values, axis=0)
    print(f"Avg MCTS Policy (Search): Flat={avg_av[0]:.4f}, Long={avg_av[1]:.4f}, Short={avg_av[2]:.4f}")

def run_diagnostic_episode(config_path, data_path, use_random=True):
    print(f"\n--- Running Instrumented Episode (Random={use_random}) ---")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    data = pd.read_csv(data_path)
    env = TradingGame(data, cfg)
    state = env.reset()
    
    actions_taken = []
    rewards = []
    balances = [state.portfolio['balance']]
    positions = []
    
    done = False
    step_count = 0
    
    while not done:
        if use_random:
            action = np.random.randint(0, 3)
        else:
            # Placeholder for model-based action if needed later
            action = np.random.randint(0, 3)
            
        new_state, reward, done, _ = env.step(action)
        
        actions_taken.append(action)
        rewards.append(reward)
        balances.append(new_state.portfolio['balance'])
        positions.append(new_state.portfolio['position'])
        
        step_count += 1
        if step_count >= 100: # Limit diagnostic to 100 steps
            break

    print(f"Steps: {step_count}")
    print(f"Action distribution: Flat={actions_taken.count(0)}, "
          f"Long={actions_taken.count(1)}, Short={actions_taken.count(2)}")
    print(f"Rewards: mean={np.mean(rewards):.6f}, std={np.std(rewards):.6f}, "
          f"min={np.min(rewards):.6f}, max={np.max(rewards):.6f}")
    
    final_balance = balances[-1]
    initial_balance = balances[0]
    total_pnl_pct = (final_balance / initial_balance) - 1.0
    
    print(f"Initial Balance: {initial_balance:.2f}")
    print(f"Final Balance: {final_balance:.2f}")
    print(f"Total PnL %: {total_pnl_pct*100:.2f}%")
    print(f"Terminal z (tanh): {np.tanh(total_pnl_pct):.6f}")
    
    if np.std(rewards) == 0:
        print("[CRITICAL] REWARD VARIANCE IS ZERO. The reward function is not sensing price changes.")
    if final_balance == initial_balance and actions_taken.count(0) < step_count:
        print("[CRITICAL] BALANCE UNCHANGED despite non-flat actions. PnL calculation is broken.")

if __name__ == "__main__":
    # 1. Analyze existing corrupted/new memory
    SYM = "BTC_USDT"
    TF = "1h"
    analyze_existing_memory(f"run/memory/memory_{SYM}_{TF}.p")
    
    # 2. Run fresh diagnostic
    run_diagnostic_episode("config.yaml", "data/train.csv")
