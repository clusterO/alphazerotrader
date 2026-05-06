import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
import os
from game import TradingGame
from model import Residual_CNN, TransformerBlock, softmax_cross_entropy_with_logits
from agent import Agent
from settings import run_folder

def run_diagnostic(model_path, data_path, label):
    # 1. Load Config
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 2. Load Environment
    data = pd.read_csv(data_path)
    env = TradingGame(data, cfg)
    
    # 3. Load Model
    # The new model should match the environment (19 features)
    nn = Residual_CNN(cfg['rl']['learning_rate'], cfg['rl']['learning_rate'], env.input_shape, env.action_size)
    m_tmp = tf.keras.models.load_model(model_path, custom_objects={
        'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits,
        'TransformerBlock': TransformerBlock
    })
    nn.model.set_weights(m_tmp.get_weights())
    
    # 4. Initialize Agent in FAIR MODE (MCTS=0)
    agent = Agent('diagnostic_agent', env.state_size, env.action_size, 0, cfg['rl']['cpuct'], nn)
    
    # 5. Inference Loop
    state = env.reset()
    probs_list = []
    vals_list = []
    
    print(f"\n--- DIAGNOSTIC: {label} ---")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Data: {data_path} ({len(data)} rows)")
    
    while not env.gameState.isEndGame:
        # Get raw predictions
        val, probs, allowed = agent.get_preds(env.gameState)
        probs_list.append(probs)
        vals_list.append(val)
        
        if env.gameState.current_tick % 50 == 0:
            print(f"Tick {env.gameState.current_tick:4d} | P: [Flat: {probs[0]:.4f}, Long: {probs[1]:.4f}, Short: {probs[2]:.4f}] | V: {val:+.4f}")
        
        # Take deterministic best action according to policy head to progress
        action = np.argmax(probs)
        state, reward, done, _ = env.step(action)

    avg_probs = np.mean(probs_list, axis=0)
    avg_val = np.mean(vals_list)
    return avg_probs, avg_val

def main():
    model_path = "run/models/BTC_USDT_1h_v0020.keras"
    bear_data = "data/stable/backtest/bear_2018.csv"
    bull_data = "data/stable/backtest/bull_2019.csv"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    results = {}
    
    results['Bear 2018'] = run_diagnostic(model_path, bear_data, "Bear 2018")
    results['Bull 2019'] = run_diagnostic(model_path, bull_data, "Bull 2019")
    
    print("\n" + "="*65)
    print("FINAL SUMMARY: RAW POLICY & VALUE")
    print("="*65)
    print(f"{'Regime':<15} | {'Flat':<8} | {'Long':<8} | {'Short':<8} | {'Value':<8}")
    print("-"*65)
    for regime, (avg_p, avg_v) in results.items():
        print(f"{regime:<15} | {avg_p[0]:.4f}   | {avg_p[1]:.4f}   | {avg_p[2]:.4f}   | {avg_v:+.4f}")
    print("="*65)

if __name__ == "__main__":
    main()
