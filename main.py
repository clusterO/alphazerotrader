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

iteration = 0

while 1:
    iteration += 1
    print(f'ITERATION NUMBER {iteration}')
    print(f'BEST PLAYER VERSION {best_player_version}')

    ######## SELF PLAY ########
    print(f'SELF PLAYING {cfg["rl"]["episodes"]} EPISODES...')
    _, memory, _, _ = playMatches(env, best_player, best_player, cfg['rl']['episodes'], lg.logger_main, turns_until_tau0 = cfg['rl']['turns_until_tau0'], memory = memory)

    # SAVE MEMORY BACKUP
    memory.save(memory_path)
    print(f'CURRENT MEMORY SIZE: {len(memory.ltmemory)} / {cfg["rl"]["memory_size"]}')

    if len(memory.ltmemory) >= cfg['rl']['memory_size']:
        ######## RETRAINING ########
        print('RETRAINING...')
        current_player.replay(memory.ltmemory)

        ######## TOURNAMENT ########
        print('TOURNAMENT (UNSEEN DATA)...')
        val_data = pd.read_csv('data/val.csv')
        val_env = Game(val_data, cfg)
        scores, _, points, _ = playMatches(val_env, best_player, current_player, cfg['evaluation']['eval_episodes'], lg.logger_tourney, turns_until_tau0 = 0, memory = None)

        # Calculate Sharpe
        def get_sharpe(pts):
            if len(pts) < 2: return -1.0
            return np.mean(pts) / (np.std(pts) + 1e-9)

        best_sharpe = get_sharpe(points[best_player.name])
        curr_sharpe = get_sharpe(points[current_player.name])

        print(f'BEST PLAYER SHARPE: {best_sharpe:.4f}')
        print(f'CURR PLAYER SHARPE: {curr_sharpe:.4f}')

        if curr_sharpe > best_sharpe * cfg['evaluation']['scoring_threshold']:
            print('NEW BEST PLAYER PROMOTED!')
            best_player_version += 1
            best_NN.model.set_weights(current_NN.model.get_weights())

            model_name = f"{SYM}_{TF}_v{str(best_player_version).zfill(4)}.keras"
            best_NN.model.save(run_folder + 'models/' + model_name)
            print(f"Saved: {model_name}")
        else:
            print(f'STAYING WITH VERSION {best_player_version}')
