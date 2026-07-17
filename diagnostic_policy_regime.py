import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
import os
from game import TradingGame
from model import Residual_CNN, TransformerBlock, softmax_cross_entropy_with_logits
from agent import Agent

def run_diagnostic(model_path, data_path, label):
    # 1. Load Config
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 2. Load Environment
    data = pd.read_csv(data_path)
    env = TradingGame(data, cfg)
    
    # 3. Load Model
    nn = Residual_CNN(cfg['rl']['learning_rate'], cfg['rl']['learning_rate'], env.input_shape, env.action_size)
    m_tmp = tf.keras.models.load_model(model_path, custom_objects={
        'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits,
        'TransformerBlock': TransformerBlock
    })
    nn.model.set_weights(m_tmp.get_weights())
    
    # 4. Initialize Agent in FAIR MODE (MCTS=0)
    agent = Agent('diagnostic_agent', env.state_size, env.action_size, 0, cfg['rl']['cpuct'], nn, cfg)
    
    # 5. Inference Loop
    env.reset()
    probs_list = []
    vals_list = []
    
    print(f"\nEvaluating {label}...")
    
    while not env.gameState.isEndGame:
        # Get raw predictions
        val, probs, allowed = agent.get_preds(env.gameState)
        probs_list.append(probs)
        vals_list.append(val)
        
        # Progress state
        action = np.argmax(probs)
        env.step(action)

    avg_probs = np.mean(probs_list, axis=0)
    avg_val = np.mean(vals_list)
    return avg_probs, avg_val

def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    SYM = cfg['trading']['symbol'].replace("/", "_")
    TF = cfg['trading']['timeframe']
    run_folder = "./run/"
    model_dir = run_folder + 'models/'
    
    # Find latest model
    diag_name = f"{SYM}_{TF}_diagnostic.keras"
    if os.path.exists(os.path.join(model_dir, diag_name)):
        model_path = os.path.join(model_dir, diag_name)
        print(f"USING DIAGNOSTIC MODEL: {diag_name}")
    else:
        existing_models = [f for f in os.listdir(model_dir) if f.startswith(f"{SYM}_{TF}_v") and f.endswith('.keras')]
        if not existing_models:
            print("No models found in " + model_dir)
            return
        existing_models.sort()
        model_path = os.path.join(model_dir, existing_models[-1])
        print(f"USING LATEST BEST MODEL: {os.path.basename(model_path)}")
    
    bear_data = "data/stable/backtest/bear_2018.csv"
    bull_data = "data/stable/backtest/bull_2019.csv"
    
    print(f"GATE 3 DIAGNOSTIC: 5-Iteration Test Results")
    print(f"Latest Model: {os.path.basename(model_path)}")
    
    results = {}
    results['Bear 2018'] = run_diagnostic(model_path, bear_data, "Bear 2018")
    results['Bull 2019'] = run_diagnostic(model_path, bull_data, "Bull 2019")
    
    print("\n" + "="*75)
    print(f"{'Regime':<15} | {'Flat':<8} | {'Long':<8} | {'Short':<8} | {'Value':<8}")
    print("-"*75)
    for regime, (avg_p, avg_v) in results.items():
        print(f"{regime:<15} | {avg_p[0]:.4f}   | {avg_p[1]:.4f}   | {avg_p[2]:.4f}   | {avg_v:+.4f}")
    print("="*75)
    
    # --- OUTCOME CATEGORIZATION ---
    bear_p, bear_v = results['Bear 2018']
    bull_p, bull_v = results['Bull 2019']
    
    short_diff = (bear_p[2] - bull_p[2]) * 100 # Bear Short - Bull Short
    long_diff = (bull_p[1] - bear_p[1]) * 100  # Bull Long - Bear Long
    
    print("\nGate 3 Metrics:")
    print(f"  Short Difference (Bear - Bull): {short_diff:+.1f}%")
    print(f"  Long Difference (Bull - Bear):  {long_diff:+.1f}%")
    print(f"  Value Separation: Bear={bear_v:+.4f}, Bull={bull_v:+.4f}")
    
    # Criteria (Aggressive Pivot - Iteration 5 Gate)
    # Success: Bear value negative, Bull value >= Bear value
    # Short diff > 8%, Long diff > 8%
    # Flat % < 60%
    v_separated = (bear_v < 0 and bull_v >= bear_v)
    p_separated = (short_diff > 8.0 and long_diff > 8.0)
    flat_check = (bear_p[0] < 0.60 and bull_p[0] < 0.60)
    
    print("\nFINAL VERDICT:")
    if v_separated and p_separated and flat_check:
        print(">>> [OUTCOME A] SUCCESS: Value head separates, policy diverges, and flat bias is contained.")
        print("    Proceed to full training.")
    elif not v_separated:
        print(">>> [OUTCOME B] FAILURE: Value head uniform or inverted.")
        print("    World model is too noisy or rewards are misaligned.")
    elif not flat_check:
        print(">>> [OUTCOME C] FAILURE: Flat bias too high (>60%).")
        print("    Agent is paralyzed. Increase action floor or reduce idling penalty.")
    elif v_separated and not p_separated:
        print(">>> [OUTCOME D] FAILURE: Value separates but policy head ignores it.")
        print("    Check entropy/temperature or use explicit regime conditioning.")

if __name__ == "__main__":
    main()
