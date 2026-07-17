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
import subprocess
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
trans_memory_path = f"{run_folder}memory/trans_memory_{SYM}_{TF}.p"

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
data_dir = cfg['trading'].get('data_path', 'data/stable')
regime_names = ['bear.csv', 'bull.csv', 'range.csv']
train_files = [os.path.join(data_dir, 'training', f) for f in regime_names]
val_files = [os.path.join(data_dir, 'val', f) for f in regime_names]

print(f"Loading mixed-regime environment from: {data_dir}")
train_data_list = [pd.read_csv(f) for f in train_files]
val_data_list = [pd.read_csv(f) for f in val_files]

env = TradingGame(train_data_list, cfg)

# Load existing state
current_NN = Residual_CNN(cfg['rl']['learning_rate'], cfg['rl']['learning_rate'], env.input_shape, env.action_size)
best_NN = Residual_CNN(cfg['rl']['learning_rate'], cfg['rl']['learning_rate'], env.input_shape, env.action_size)

best_player_version = 0
model_dir = run_folder + 'models/'
existing_models = [f for f in os.listdir(model_dir) if f.startswith(f"{SYM}_{TF}_v") and f.endswith('.keras')]
diag_model_path = model_dir + f"{SYM}_{TF}_diagnostic.keras"

if existing_models:
    existing_models.sort()
    latest_model = existing_models[-1]
    best_player_version = int(latest_model.split('_v')[-1].split('.')[0])
    print(f"LOADING LATEST VERSIONED MODEL: {latest_model}")
    from model import TransformerBlock, softmax_cross_entropy_with_logits
    m_tmp = tf.keras.models.load_model(model_dir + latest_model, custom_objects={'TransformerBlock': TransformerBlock, 'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})
    best_NN.model.set_weights(m_tmp.get_weights())
    current_NN.model.set_weights(m_tmp.get_weights())
elif os.path.exists(diag_model_path):
    print(f"NO VERSIONED MODELS FOUND. LOADING DIAGNOSTIC MODEL: {diag_model_path}")
    from model import TransformerBlock, softmax_cross_entropy_with_logits
    m_tmp = tf.keras.models.load_model(diag_model_path, custom_objects={'TransformerBlock': TransformerBlock, 'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})
    best_NN.model.set_weights(m_tmp.get_weights())
    current_NN.model.set_weights(m_tmp.get_weights())

# Persistent Memory
if os.path.exists(memory_path):
    with open(memory_path, "rb") as f:
        memory = pickle.load(f)
    print(f"LOADED PERSISTENT MEMORY FROM {memory_path}")
else:
    from memory import Memory
    memory = Memory(cfg['rl']['memory_size'])

# Iteration logic
iteration = getattr(memory, 'metadata', {}).get('iteration', 0)
if iteration > 0:
    print(f"RESUMING FROM ITERATION {iteration}")
else:
    print("STARTING FRESH RUN (Iteration 0)")

# Load World Model
world_model = None
if cfg['rl'].get('use_world_model', False):
    from world_model import TransitionModel
    wm_path = f"{run_folder}models/world_model_{SYM}_{TF}.keras"
    if os.path.exists(wm_path):
        print(f"LOADING WORLD MODEL FROM {wm_path}")
        world_model = TransitionModel.load(wm_path)
    else:
        print(f"WARNING: World Model enabled but not found at {wm_path}. Proceeding with historical data.")

# Initialization: Agents
current_player = Agent('current_player', env.state_size, env.action_size, cfg['rl']['mcts_sims'], cfg['rl']['cpuct'], current_NN, cfg, world_model=world_model)
best_player = Agent('best_player', env.state_size, env.action_size, cfg['rl']['mcts_sims'], cfg['rl']['cpuct'], best_NN, cfg, world_model=world_model)

# Persistent Transition Memory
from transition_memory import TransitionMemory
trans_memory = TransitionMemory(capacity=100000)
if os.path.exists(trans_memory_path):
    trans_memory.load(trans_memory_path)

# --- MAIN LOOP ---
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
    print(f'SELF PLAYING {cfg["rl"]["episodes"]} EPISODES...')
    from funcs import playMatches
    _, memory, _, _ = playMatches(env, best_player, best_player, cfg['rl']['episodes'], lg.logger_main, turns_until_tau0=0, memory=memory, is_warmup=is_warmup, transition_memory=trans_memory)
    
    memory.metadata = {'iteration': iteration}
    memory.save(memory_path)
    trans_memory.save(trans_memory_path)

    # --- MAINTENANCE & REPORTS ---
    
    # 3. MCTS Visit Count Divergence Report (Every 2 iterations for faster feedback)
    if iteration % 2 == 0:
        print(f"\n--- MCTS VISIT DIVERGENCE REPORT (Iteration {iteration}) ---")
        for regime in ['bear', 'bull', 'range']:
            # Aggregate stats from both agents
            stats = current_player.visit_stats[regime] + best_player.visit_stats[regime]
            if stats:
                avg_dist = np.mean(stats, axis=0)
                total = np.sum(avg_dist) + 1e-9
                avg_pct = (avg_dist / total) * 100
                print(f"Regime {regime.upper():<5} | Flat: {avg_pct[0]:.1f}% | Long: {avg_pct[1]:.1f}% | Short: {avg_pct[2]:.1f}% | (N={len(stats)})")
            else:
                print(f"Regime {regime.upper():<5} | No data collected.")
        
        # Clear buffers
        current_player.visit_stats = {'bear': [], 'bull': [], 'range': []}
        best_player.visit_stats = {'bear': [], 'bull': [], 'range': []}


    # B. World Model Retraining (Every 2 iterations)
    if world_model is not None and iteration % 2 == 0:
        print(f"\n--- SCHEDULED WORLD MODEL RETRAINING (Iteration {iteration}) ---")
        if len(trans_memory) >= 500:
            states, actions, next_states = trans_memory.sample(10000)
            world_model.set_trainable(True)
            world_model.fit(states, actions, next_states, epochs=5, batch_size=128)
            world_model.set_trainable(False)
            wm_path = f"{run_folder}models/world_model_{SYM}_{TF}.keras"
            world_model.save(wm_path)
            print(f"World Model adapted and saved to {wm_path}")

    # C. Automated Diagnostic Monitoring (Every 5 iterations)
    if iteration % 5 == 0:
        print(f"\n--- SCHEDULED GATE 3 DIAGNOSTIC (Iteration {iteration}) ---")
        try:
            result = subprocess.run(['python', 'diagnostic_policy_regime.py'], capture_output=True, text=True)
            print(result.stdout)
        except Exception as e:
            print(f"Failed to run diagnostic: {e}")

    # --- TRAINING DIAGNOSTICS (In-Memory Stats) ---
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

    # 2. RETRAINING
    if len(memory.ltmemory) >= (cfg['rl']['memory_size'] * 0.1):
        print(f'RETRAINING ({len(memory.ltmemory)} memories)...')
        current_player.replay(memory.ltmemory)
        
        # Save current state for diagnostics
        diag_model_path = model_dir + f"{SYM}_{TF}_diagnostic.keras"
        current_NN.model.save(diag_model_path)
        print(f"DIAGNOSTIC MODEL SAVED TO {diag_model_path}")

        # 3. TOURNAMENT
        if not is_warmup and not cfg['rl'].get('skip_tournament', False):
            print('TOURNAMENT (Multi-Regime Validation)...')
            regime_sharpes = []
            regime_wins = 0
            for r_idx, v_file in enumerate(val_files):
                v_env = TradingGame(val_data_list[r_idx], cfg)
                rets = playMatchesSingle(v_env, current_player, cfg['evaluation']['eval_episodes'] // 3, lg.logger_tourney)
                sharpe = get_sharpe(rets, label=f"REGIME {r_idx}")
                regime_sharpes.append(sharpe)
                if sharpe > 0: regime_wins += 1

            avg_sharpe = np.mean(regime_sharpes)
            print(f"\nOverall Multi-Regime Sharpe: {avg_sharpe:.4f}")
            print(f"Regimes with positive Sharpe: {regime_wins} / {len(val_files)}")

            # 4. PROMOTION
            if avg_sharpe > 0.02 and regime_wins >= 2:
                print('NEW BEST PLAYER PROMOTED!')
                best_player_version += 1
                best_NN.model.set_weights(current_NN.model.get_weights())
                model_name = f"{SYM}_{TF}_v{str(best_player_version).zfill(4)}.keras"
                best_NN.model.save(model_dir + model_name)
                best_player.model.model.set_weights(current_NN.model.get_weights())
        else:
            print("SKIPPING TOURNAMENT (Warmup or Skip Mode)")

    gc.collect()
