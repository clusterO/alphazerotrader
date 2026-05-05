# -*- coding: utf-8 -*-
import numpy as np
import yaml
import pandas as pd
import os
import tensorflow as tf
import pickle
import psutil
import gc
import time
from game import TradingGame
from model import Residual_CNN, TransformerBlock, softmax_cross_entropy_with_logits
from agent import Agent
import loggers as lg
from funcs import playMatchesSingle

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Global Settings
SYM = cfg['trading']['symbol'].replace("/", "_")
TF = cfg['trading']['timeframe']
run_folder = "./run/"
memory_path = f"{run_folder}memory/memory_{SYM}_{TF}.p"

# Ensure dirs
if not os.path.exists(run_folder + 'models'): os.makedirs(run_folder + 'models')
if not os.path.exists(run_folder + 'memory'): os.makedirs(run_folder + 'memory')

# Reusable function for Sharpe
def get_sharpe(rets, label=""):
    if not rets or len(rets) < 10: 
        if label: print(f"  {label} -> Insufficient returns ({len(rets) if rets else 0})")
        return 0.0
    mu = np.mean(rets)
    sigma = np.std(rets) + 1e-9
    ann_factor = 1.0 # Use raw episode Sharpe for consistent 5m ranking
    sharpe = (mu / sigma) * ann_factor
    if label:
        print(f"  {label} -> Sharpe: {sharpe:.4f} (Mean: {mu:.6f}, Std: {sigma:.6f}, N: {len(rets)})")
    return sharpe

# Environment setup
# Determine data paths (Phase 5: Parameterized Data)
# PHASE 6: MIXED-REGIME TRAINING DATA (with 80/20 Time-Series Split)
data_dir = cfg['trading'].get('data_path', 'data/stable')
regime_names = ['bear.csv', 'bull.csv', 'range.csv']

train_files = [os.path.join(data_dir, 'training', f) for f in regime_names]
val_files = [os.path.join(data_dir, 'val', f) for f in regime_names]

print(f"Loading mixed-regime environment from: {data_dir}")
train_data_list = [pd.read_csv(f) for f in train_files]
val_data_list = [pd.read_csv(f) for f in val_files]

print(f"Data loading complete. Train size: {[len(d) for d in train_data_list]}, Val size: {[len(d) for d in val_data_list]}")

env = TradingGame(train_data_list, cfg)
# We also keep the original unseen val.csv as the final holdout "Wall of Truth"
holdout_file = 'data/val.csv'

# Load existing state
current_NN = Residual_CNN(cfg['rl']['learning_rate'], cfg['rl']['learning_rate'], env.input_shape, env.action_size)
best_NN = Residual_CNN(cfg['rl']['learning_rate'], cfg['rl']['learning_rate'], env.input_shape, env.action_size)

best_player_version = 0
model_dir = run_folder + 'models/'
existing_models = [f for f in os.listdir(model_dir) if f.startswith(f"{SYM}_{TF}_v") and f.endswith('.keras')]
if existing_models:
    existing_models.sort()
    latest_model = existing_models[-1]
    best_player_version = int(latest_model.split('_v')[-1].split('.')[0])
    from model import TransformerBlock, softmax_cross_entropy_with_logits
    m_tmp = tf.keras.models.load_model(model_dir + latest_model, custom_objects={'TransformerBlock': TransformerBlock, 'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})
    best_NN.model.set_weights(m_tmp.get_weights())
    current_NN.model.set_weights(m_tmp.get_weights())
    print(f"RESUMING FROM VERSION {best_player_version}")

current_player = Agent('current_player', env.state_size, env.action_size, cfg['rl']['mcts_sims'], cfg['rl']['cpuct'], current_NN)
best_player = Agent('best_player', env.state_size, env.action_size, cfg['rl']['mcts_sims'], cfg['rl']['cpuct'], best_NN)

# Persistent Memory
if os.path.exists(memory_path):
    with open(memory_path, "rb") as f:
        memory = pickle.load(f)
else:
    from memory import Memory
    memory = Memory(cfg['rl']['memory_size'])

iteration = getattr(memory, 'metadata', {}).get('iteration', 0)
if not existing_models:
    print("NEW INSTRUMENT/FRESH RUN DETECTED: RESETTING ITERATION TO 0")
    iteration = 0

# --- MAIN LOOP (Phase 4.0: Zero Bias) ---
while 1:
    iteration += 1
    print(f'\n--- ITERATION {iteration} (Version {best_player_version}) ---')
    
    # Adaptive Learning Rate
    warmup_iters = cfg['rl'].get('warmup_iterations', 0)
    if iteration <= warmup_iters:
        lr = cfg['rl']['learning_rate']
    else:
        # Reset decay logic to start after warmup
        decay_step = iteration - warmup_iters
        lr = cfg['rl']['learning_rate'] * (cfg['rl']['lr_decay'] ** decay_step)
        
    print(f"ADAPTIVE LR: {lr:.7f}")
    current_NN.set_lr(lr)

    # 1. SELF PLAY
    is_warmup = (iteration <= warmup_iters)
    
    if is_warmup:
        print(f"WARMUP PHASE ({iteration}/{warmup_iters}): FORCING UNIFORM EXPLORATION")

    print(f'SELF PLAYING {cfg["rl"]["episodes"]} EPISODES...')
    from funcs import playMatches
    _, memory, _, _ = playMatches(env, best_player, best_player, cfg['rl']['episodes'], lg.logger_main, turns_until_tau0=0, memory=memory, is_warmup=is_warmup)
    
    memory.metadata = {'iteration': iteration}
    memory.save(memory_path)

    # --- TRAINING DIAGNOSTICS (Restored) ---
    z_values = [mem['value'] for mem in memory.ltmemory]
    z_mean = np.mean(z_values) if z_values else 0
    z_std = np.std(z_values) if z_values else 0

    actions = [np.argmax(mem['AV']) for mem in memory.ltmemory]
    flat_pct = actions.count(0) / len(actions) if actions else 0
    long_pct = actions.count(1) / len(actions) if actions else 0
    short_pct = actions.count(2) / len(actions) if actions else 0

    ram_usage = psutil.Process().memory_info().rss / (1024 * 1024 * 1024) # GB

    print(f'MEMORY SIZE: {len(memory.ltmemory)} / {cfg["rl"]["memory_size"]}')
    print(f'Memory z: mean={z_mean:.3f}, std={z_std:.3f}')
    print(f'Action Dist: Flat={flat_pct:.1%}, Long={long_pct:.1%}, Short={short_pct:.1%}')
    print(f'RAM Usage: {ram_usage:.2f} GB')
    # ---------------------------------------

    # 2. RETRAINING (if buffer 10% full and NOT in warmup)

    if not is_warmup and len(memory.ltmemory) >= (cfg['rl']['memory_size'] * 0.1): # Lowered threshold to start learning earlier
        print(f'RETRAINING ({len(memory.ltmemory)} memories)...')
        current_player.replay(memory.ltmemory)

        # 3. TOURNAMENT (PHASE 6: MULTI-REGIME)
        print('TOURNAMENT (Multi-Regime Validation)...')
        
        regime_sharpes = []
        regime_wins = 0
        
        for r_idx, v_file in enumerate(val_files):
            print(f"Testing Regime {r_idx}: {v_file}")
            v_env = TradingGame(val_data_list[r_idx], cfg)
            
            # Use smaller eval batches per regime for speed
            rets = playMatchesSingle(v_env, current_player, cfg['evaluation']['eval_episodes'] // 3, lg.logger_tourney)
            sharpe = get_sharpe(rets, label=f"REGIME {r_idx}")
            
            regime_sharpes.append(sharpe)
            if sharpe > 0:
                regime_wins += 1

        avg_sharpe = np.mean(regime_sharpes)
        print(f"\nOverall Multi-Regime Sharpe (Validation Slices): {avg_sharpe:.4f}")
        print(f"Regimes with positive Sharpe: {regime_wins} / {len(val_files)}")

        # 4. PROMOTION (2-of-3 Rule)
        # Model must have a positive Sharpe on at least 2 of the 3 hidden market slices
        if avg_sharpe > 0.02 and regime_wins >= 2:
            print('NEW BEST PLAYER PROMOTED! Passed 2-of-3 Multi-Regime Validation.')
            best_player_version += 1
            best_NN.model.set_weights(current_NN.model.get_weights())
            model_name = f"{SYM}_{TF}_v{str(best_player_version).zfill(4)}.keras"
            best_NN.model.save(model_dir + model_name)
            # Synchronize best_player
            best_player.model.model.set_weights(current_NN.model.get_weights())
    
    # Memory Management
    gc.collect()
