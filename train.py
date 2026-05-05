import numpy as np
import os
import random
import pickle
import yaml
import time
import gc
from shutil import copyfile
from importlib import reload

import loggers as lg
from game import TradingGame
from agent import Agent
from memory import Memory
from model import Residual_CNN
from funcs import playMatches
from settings import run_folder, run_archive_folder
import initialise

def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    
    lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
    lg.logger_main.info('=*=*=*=*=*=.      STABLE RUN      =*=*=*=*=*')
    lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')

    # PHASE 6: MIXED-REGIME TRAINING DATA
    import pandas as pd
    data_dir = cfg['trading'].get('data_path', 'data/stable')
    regime_names = ['bear.csv', 'bull.csv', 'range.csv']
    
    train_files = [os.path.join(data_dir, 'training', f) for f in regime_names]
    val_files = [os.path.join(data_dir, 'val', f) for f in regime_names]

    print(f"Loading mixed-regime training data from: {os.path.join(data_dir, 'training')}")
    data_list = [pd.read_csv(f) for f in train_files]
    env = TradingGame(data_list, cfg)

    # Validation regimes (Bull, Bear, Range)
    val_data_list = [pd.read_csv(f) for f in val_files]

    ######## LOAD MEMORIES IF NECESSARY ########
    if initialise.INITIAL_MEMORY_VERSION == None:
        memory = Memory(cfg['rl']['memory_size'])
    else:
        print('LOADING MEMORY VERSION ' + str(initialise.INITIAL_MEMORY_VERSION) + '...')
        m_path = run_archive_folder + env.name + '/run' + str(initialise.INITIAL_RUN_NUMBER).zfill(4) + "/memory/memory" + str(initialise.INITIAL_MEMORY_VERSION).zfill(4) + ".p"
        with open(m_path, 'rb') as f:
            memory = pickle.load(f)

    ######## LOAD MODEL IF NECESSARY ########
    input_dim = env.input_shape
    current_NN = Residual_CNN(cfg['rl']['learning_rate'], cfg['rl']['learning_rate'], input_dim, env.action_size)
    best_NN = Residual_CNN(cfg['rl']['learning_rate'], cfg['rl']['learning_rate'], input_dim, env.action_size)

    if initialise.INITIAL_MODEL_VERSION != None:
        best_player_version = initialise.INITIAL_MODEL_VERSION
        print('LOADING MODEL VERSION ' + str(initialise.INITIAL_MODEL_VERSION) + '...')
        m_tmp = best_NN.read(env.name, initialise.INITIAL_RUN_NUMBER, best_player_version)
        current_NN.model.set_weights(m_tmp.get_weights())
        best_NN.model.set_weights(m_tmp.get_weights())
    else:
        best_player_version = 0
        best_NN.model.set_weights(current_NN.model.get_weights())

    # Save initial config
    copyfile('./config.yaml', run_folder + 'config.yaml')

    ######## CREATE THE PLAYERS ########
    current_player = Agent('current_player', env.state_size, env.action_size, cfg['rl']['mcts_sims'], cfg['rl']['cpuct'], current_NN)
    best_player = Agent('best_player', env.state_size, env.action_size, cfg['rl']['mcts_sims'], cfg['rl']['cpuct'], best_NN)
    
    iteration = 0
    consecutive_high_z = 0

    while True:
        # Check for stop signal
        if os.path.exists('stop_train.flag'):
            print("Stop signal detected. Exiting...")
            os.remove('stop_train.flag')
            break

        iteration += 1
        cfg = load_config() # Reload config every iteration
        
        print(f'\nITERATION NUMBER {iteration}')
        lg.logger_main.info('BEST PLAYER VERSION: %d', best_player_version)
        print(f'BEST PLAYER VERSION {best_player_version}')

        ######## SELF PLAY ########
        print(f'SELF PLAYING {cfg["rl"]["episodes"]} EPISODES...')
        _, memory, _, _ = playMatches(env, best_player, best_player, cfg['rl']['episodes'], lg.logger_main, turns_until_tau0 = cfg['rl'].get('turns_until_tau0', 10), memory = memory)
        print('\n')
        
        # PHASE 6: Z-MEAN MONITORING
        z_values = [m['value'] for m in memory.stmemory]
        if z_values:
            avg_z = np.mean(z_values)
            abs_z = abs(avg_z)
            print(f"Iteration Avg Z-Value: {avg_z:.4f}")
            if abs_z > 0.85:
                consecutive_high_z += 1
                print(f"WARNING: High directional saturation detected ({consecutive_high_z} consecutive)")
                if consecutive_high_z >= 5:
                    print("CRITICAL: Directional Monoculture confirmed. Consider adversarial injection.")
            else:
                consecutive_high_z = 0

        memory.clear_stmemory()
        
        if len(memory.ltmemory) >= cfg['rl']['batch_size']:
            ######## RETRAINING ########
            print('RETRAINING...')
            current_player.replay(memory.ltmemory)
            print('')

            if iteration % 5 == 0:
                m_save_path = run_folder + "memory/memory" + str(iteration).zfill(4) + ".p"
                with open(m_save_path, 'wb') as f:
                    pickle.dump(memory, f)

            ######## TOURNAMENT (PHASE 6: MULTI-REGIME) ########
            print('TOURNAMENT (Multi-Regime Validation)...')
            
            regime_sharpes = []
            regime_wins = 0
            
            for r_idx, v_file in enumerate(val_files):
                print(f"Testing Regime {r_idx}: {v_file}")
                # Create specialized env for this regime
                v_env = TradingGame(val_data_list[r_idx], cfg)
                
                scores, _, points, _ = playMatches(v_env, best_player, current_player, cfg['evaluation']['eval_episodes'] // 4, lg.logger_tourney, turns_until_tau0 = 0, memory = None)
                
                # Calculate Sharpe for this regime
                rets = np.array(points['current_player'])
                if len(rets) > 1:
                    mu = np.mean(rets)
                    sigma = np.std(rets) + 1e-9
                    sharpe = mu / sigma
                else:
                    sharpe = -1.0
                
                regime_sharpes.append(sharpe)
                if sharpe > 0:
                    regime_wins += 1
                
                print(f"Regime {r_idx} Sharpe: {sharpe:.4f} | Wins: {scores['current_player']} vs {scores['best_player']}")

            avg_sharpe = np.mean(regime_sharpes)
            print(f"\nOverall Multi-Regime Sharpe: {avg_sharpe:.4f}")
            print(f"Regimes with positive Sharpe: {regime_wins} / {len(val_files)}")

            # PROMOTION RULE: Improve Avg Sharpe AND pass at least 2 regimes
            if regime_wins >= 2 and avg_sharpe > 0.02:
                best_player_version = best_player_version + 1
                print(f'UPDATING BEST PLAYER TO VERSION {best_player_version}')
                best_NN.model.set_weights(current_NN.model.get_weights())
                best_NN.write(env.name, best_player_version)
                best_player.model.model.set_weights(current_NN.model.get_weights())
        else:
            print(f'MEMORY SIZE: {len(memory.ltmemory)} / {cfg["rl"]["batch_size"]}')

        gc.collect()

if __name__ == "__main__":
    main()
