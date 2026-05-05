import numpy as np
import pandas as pd
import yaml
import pickle
import os
from game import TradingGame

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

def run_diagnostic_episode(config_path, data_path, use_random=True):
    print(f"\n--- Running FULL RISK-AWARE Episode (Random={use_random}) ---")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    data = pd.read_csv(data_path)
    env = TradingGame(data, cfg)
    state = env.reset()
    
    actions_taken = []
    rewards = []
    balances = [state.portfolio['balance']]
    
    done = False
    step_count = 0
    
    while not done:
        action = np.random.randint(0, 3) if use_random else 1 # Force long if not random
        new_state, reward, done, _ = env.step(action)
        
        actions_taken.append(action)
        rewards.append(reward)
        balances.append(new_state.portfolio['balance'])
        step_count += 1

    print(f"Steps: {step_count}")
    print(f"Action distribution: Flat={actions_taken.count(0)}, Long={actions_taken.count(1)}, Short={actions_taken.count(2)}")
    
    # Calculate Terminal Z manually to see the penalty impact
    pnl_pct = (new_state.portfolio['balance'] / cfg['trading']['initial_balance']) - 1.0
    max_dd = new_state.portfolio['max_drawdown']
    
    risk_penalty = 0.0
    if max_dd > cfg['trading']['drawdown_threshold']:
        risk_penalty = (max_dd - cfg['trading']['drawdown_threshold']) * 2.0
    
    z_raw = np.tanh(cfg['trading']['tanh_scale'] * pnl_pct)
    z_final = np.clip(z_raw - risk_penalty, -1.0, 1.0)
    
    print(f"\n--- RISK RESULTS ---")
    print(f"Total PnL %:      {pnl_pct*100:.2f}%")
    print(f"Max Drawdown:      {max_dd*100:.2f}%")
    print(f"Threshold:         {cfg['trading']['drawdown_threshold']*100:.1f}%")
    print(f"Raw Tanh(PnL):     {z_raw:.4f}")
    print(f"Risk Penalty:     -{risk_penalty:.4f}")
    print(f"Final Terminal z:  {z_final:.4f}")
    
    if risk_penalty > 0:
        print("[SUCCESS] Penalty logic is active and reducing terminal value.")
    else:
        print("[INFO] Drawdown was within safety limits. No terminal penalty.")

if __name__ == "__main__":
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    data_path = os.path.join(cfg['trading'].get('data_path', 'data/training/BTC_USDT_1h'), 'train.csv')
    run_diagnostic_episode("config.yaml", data_path, use_random=True)
