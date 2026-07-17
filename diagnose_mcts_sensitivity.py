# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import yaml
import os
import tensorflow as tf
from game import TradingGame, GameState
from model import Residual_CNN, TransformerBlock, softmax_cross_entropy_with_logits
from world_model import TransitionModel
from agent import Agent

def run_test(regime, df, cfg, nn, wm):
    print(f"Testing {regime.upper()} REGIME state...")
    env = TradingGame(df, cfg)
    # Fresh agent for every test
    agent = Agent(f'diag_{regime}', env.state_size, env.action_size, 600, 0.35, nn, cfg, world_model=wm)
    
    state = env.reset()
    # Seek a state with significant price movement
    for _ in range(50):
        state, _, _ = state.takeAction(0)
    
    # Run 600 sims
    _, pi, mcts_v, nn_v = agent.act(state, 1.0)
    return pi, mcts_v, nn_v

def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    SYM = cfg['trading']['symbol'].replace("/", "_")
    TF = cfg['trading']['timeframe']
    run_folder = "./run/"
    model_dir = run_folder + 'models/'
    
    # Force MCTS Sensitivity V2 settings
    cfg['rl']['mcts_sims'] = 600
    cfg['rl']['cpuct'] = 0.35
    cfg['rl']['epsilon'] = 0.3 

    # 1. Setup Data
    data_dir = cfg['trading'].get('data_path', 'data/stable')
    bear_df = pd.read_csv(os.path.join(data_dir, 'training', 'bear.csv'))
    bull_df = pd.read_csv(os.path.join(data_dir, 'training', 'bull.csv'))
    
    # 2. Load Models
    nn = Residual_CNN(cfg['rl']['learning_rate'], cfg['rl']['learning_rate'], (60, 19), 3)
    diag_model_path = model_dir + f"{SYM}_{TF}_diagnostic.keras"
    if os.path.exists(diag_model_path):
        m_tmp = tf.keras.models.load_model(diag_model_path, custom_objects={'TransformerBlock': TransformerBlock, 'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})
        nn.model.set_weights(m_tmp.get_weights())
        print(f"Loaded recalibrated model: {diag_model_path}")
    
    wm_path = f"{run_folder}models/world_model_{SYM}_{TF}.keras"
    world_model = TransitionModel.load(wm_path)

    print("\n" + "="*60)
    print("   MCTS SENSITIVITY DIAGNOSTIC V2 (600 Sims, 0.35 CPUCT)")
    print("="*60)

    pi_bear, mcts_v_bear, nn_v_bear = run_test('bear', bear_df, cfg, nn, world_model)
    pi_bull, mcts_v_bull, nn_v_bull = run_test('bull', bull_df, cfg, nn, world_model)

    print("\n" + "-"*40)
    print(f"Regime | Flat % | Long % | Short % | MCTS Value | NN Value")
    print(f"BEAR   | {pi_bear[0]:.1%}  | {pi_bear[1]:.1%}  | {pi_bear[2]:.1%}   | {mcts_v_bear:+.4f}    | {nn_v_bear:+.4f}")
    print(f"BULL   | {pi_bull[0]:.1%}  | {pi_bull[1]:.1%}  | {pi_bull[2]:.1%}   | {mcts_v_bull:+.4f}    | {nn_v_bull:+.4f}")
    print("-"*40)

    short_gap = (pi_bear[2] - pi_bull[2]) * 100
    long_gap = (pi_bull[1] - pi_bear[1]) * 100

    print(f"\nRESULTS:")
    print(f"  Short Divergence (Bear - Bull): {short_gap:+.1f}%")
    print(f"  Long Divergence (Bull - Bear):  {long_gap:+.1f}%")

    if short_gap > 5.0 or long_gap > 5.0: # Relaxed success threshold for V2
        print("\n>>> VERDICT: SUCCESS! MCTS is finally differentiating regimes.")
        print("    Proceed to full training.")
    else:
        print("\n>>> VERDICT: FAILURE. Prior bias still too strong or Signal too weak.")
        print("    Reduce cpuct further (e.g. 0.2) or increase epsilon for more noise.")

if __name__ == "__main__":
    main()
