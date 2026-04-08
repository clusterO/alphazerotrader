# -*- coding: utf-8 -*-
# %matplotlib inline

import numpy as np
np.set_printoptions(suppress=True)

import yaml
import pandas as pd
import random
import os
import config
from shutil import copyfile
from keras.utils import plot_model
import pickle
import psutil
import tensorflow as tf
from loss import softmax_cross_entropy_with_logits

with open("config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

# Symbol and TF for naming
SYM = cfg['trading']['symbol'].replace('/', '_')
TF = cfg['trading']['timeframe']

if hasattr(config, 'SYMBOL') or 'trading' in cfg:
    from games.trading.game import TradingGame as Game
    data = pd.read_csv('data/train.csv')
    env = Game(data, cfg)
else:
    from game import Game, GameState
    env = Game()

from agent import Agent
from memory import Memory
from model import Residual_CNN, TransformerBlock
from funcs import playMatches, playMatchesBetweenVersions

import loggers as lg
from settings import run_folder, run_archive_folder
import initialise

######## LOAD MEMORIES IF NECESSARY ########
memory_path = run_folder + f"memory/memory_{SYM}_{TF}.p"
memory = Memory(cfg['rl']['memory_size'])
if os.path.exists(memory_path):
    print(f'LOADING EXISTING MEMORY FROM {memory_path}...')
    try:
        memory = memory.load(memory_path)
        print(f'Successfully loaded {len(memory.ltmemory)} memories.')
    except Exception as e:
        print(f"Error loading memory: {e}. Starting fresh.")
else:
    print("No existing memory found. Starting fresh.")

######## LOAD MODEL IF NECESSARY ########
# create an untrained neural network objects
current_NN = Residual_CNN(cfg['rl']['learning_rate'], cfg['rl']['learning_rate'], env.input_shape,   env.action_size)
best_NN = Residual_CNN(cfg['rl']['learning_rate'], cfg['rl']['learning_rate'], env.input_shape,   env.action_size)

# Version management
best_player_version = 0
# Check if any model already exists to resume
model_dir = run_folder + 'models/'
existing_models = [f for f in os.listdir(model_dir) if f.startswith(f"{SYM}_{TF}_v") and f.endswith('.keras')]
if existing_models:
    existing_models.sort()
    latest_model = existing_models[-1]
    best_player_version = int(latest_model.split('_v')[-1].split('.')[0])
    print(f"LOADING LATEST MODEL VERSION {best_player_version} ({latest_model})...")
    m_tmp = tf.keras.models.load_model(model_dir + latest_model, custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits, 'TransformerBlock': TransformerBlock})
    current_NN.model.set_weights(m_tmp.get_weights())
    best_NN.model.set_weights(m_tmp.get_weights())

######## CREATE THE PLAYERS ########
current_player = Agent('current_player', env.state_size, env.action_size, cfg['rl']['mcts_sims'], cfg['rl']['cpuct'], current_NN)
best_player = Agent('best_player', env.state_size, env.action_size, cfg['rl']['mcts_sims'], cfg['rl']['cpuct'], best_NN)

# Load persistent iteration counter
iteration = 0
if hasattr(memory, 'metadata'):
    iteration = memory.metadata.get('iteration', 0)

while 1:
    iteration += 1
    print(f'ITERATION NUMBER {iteration}')
    print(f'BEST PLAYER VERSION {best_player_version}')

    ######## SELF PLAY ########
    print(f'SELF PLAYING {cfg["rl"]["episodes"]} EPISODES...')
    _, memory, _, _ = playMatches(env, best_player, best_player, cfg['rl']['episodes'], lg.logger_main, turns_until_tau0 = cfg['rl']['turns_until_tau0'], memory = memory)

    # SAVE MEMORY BACKUP
    if not hasattr(memory, 'metadata'):
        memory.metadata = {}
    memory.metadata['iteration'] = iteration
    memory.save(memory_path)

    # --- TRAINING DIAGNOSTICS (Phase 4.5) ---
    z_values = [mem['value'] for mem in memory.ltmemory]
    z_mean = np.mean(z_values) if z_values else 0
    z_std = np.std(z_values) if z_values else 0

    # Action distribution in memory
    actions = [np.argmax(mem['AV']) for mem in memory.ltmemory]
    flat_pct = actions.count(0) / len(actions) if actions else 0
    long_pct = actions.count(1) / len(actions) if actions else 0
    short_pct = actions.count(2) / len(actions) if actions else 0

    # RAM Monitor
    ram_usage = psutil.Process().memory_info().rss / (1024 * 1024 * 1024) # GB

    print(f'MEMORY SIZE: {len(memory.ltmemory)} / {cfg["rl"]["memory_size"]}')
    print(f'Memory z: mean={z_mean:.3f}, std={z_std:.3f}')
    print(f'Action Dist: Flat={flat_pct:.1%}, Long={long_pct:.1%}, Short={short_pct:.1%}')
    print(f'RAM Usage: {ram_usage:.2f} GB')
    # ----------------------------------------

    # Phase 4.2: don't start training until the buffer is at least 50% full
    if len(memory.ltmemory) >= (cfg['rl']['memory_size'] * 0.5):
        ######## RETRAINING ########
        print(f'RETRAINING (Buffer {len(memory.ltmemory)/cfg["rl"]["memory_size"]:.1%} full)...')
        current_player.replay(memory.ltmemory)

        ######## TOURNAMENT ########
        print('TOURNAMENT (UNSEEN DATA)...')
        val_data = pd.read_csv('data/val.csv')
        val_env = Game(val_data, cfg)
        scores, _, points, _ = playMatches(val_env, best_player, current_player, cfg['evaluation']['eval_episodes'], lg.logger_tourney, turns_until_tau0 = 0, memory = None)

        # Calculate Sharpe (Phase 4.3)
        def get_sharpe(pts):
            if len(pts) < 2: return 0.0
            mu = np.mean(pts)
            sigma = np.std(pts)
            if sigma < 1e-8: return 0.0
            # Annualization: sqrt(252*24) for 1h candles
            return (mu / sigma) * np.sqrt(252 * 24)

        best_sharpe = get_sharpe(points[best_player.name])
        curr_sharpe = get_sharpe(points[current_player.name])

        print(f'BEST PLAYER SHARPE (ANN): {best_sharpe:.4f}')
        print(f'CURR PLAYER SHARPE (ANN): {curr_sharpe:.4f}')

        # Phase 4.3: Promotion floor & First-Save Logic
        PROMOTION_THRESHOLD = -1.0

        should_promote = False
        if curr_sharpe > best_sharpe * cfg['evaluation']['scoring_threshold']:
            if curr_sharpe > PROMOTION_THRESHOLD:
                should_promote = True
            elif best_player_version == 0:
                print(f'INITIAL PROMOTION: Promoting v1 despite Sharpe {curr_sharpe:.2f} < {PROMOTION_THRESHOLD} floor.')
                should_promote = True
            else:
                print(f'PROMOTION DENIED: Current Sharpe {curr_sharpe:.4f} below floor of {PROMOTION_THRESHOLD}')

        if should_promote:
            print('NEW BEST PLAYER PROMOTED!')
            best_player_version += 1
            best_NN.model.set_weights(current_NN.model.get_weights())

            model_name = f"{SYM}_{TF}_v{str(best_player_version).zfill(4)}.keras"
            best_NN.model.save(run_folder + 'models/' + model_name)
            print(f"Saved: {model_name}")
        else:
            print(f'STAYING WITH VERSION {best_player_version}')
