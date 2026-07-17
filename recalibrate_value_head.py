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
from agent import RandomAgent, Agent
import loggers as lg
from funcs import playMatches
from memory import Memory

def main():
    # Load config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    SYM = cfg['trading']['symbol'].replace("/", "_")
    TF = cfg['trading']['timeframe']
    run_folder = "./run/"
    model_dir = run_folder + 'models/'

    # 1. CLEAR REPLAY BUFFER
    print("STEP 1: CLEARING REPLAY BUFFER")
    memory = Memory(cfg['rl']['memory_size'])
    
    # 2. GENERATE CALIBRATION DATASET
    print("STEP 2: GENERATING CALIBRATION DATASET (30 Episodes, Random Actions)")
    
    # Setup data
    data_dir = cfg['trading'].get('data_path', 'data/stable')
    regime_names = ['bear.csv', 'bull.csv', 'range.csv']
    train_files = [os.path.join(data_dir, 'training', f) for f in regime_names]
    train_data_list = [pd.read_csv(f) for f in train_files]
    
    env = TradingGame(train_data_list, cfg)
    
    # Initialize Random Agent (Fixed: Added cfg)
    random_player = RandomAgent('random_player', env.state_size, env.action_size, cfg)
    
    # Run episodes
    _, memory, _, _ = playMatches(env, random_player, random_player, 30, lg.logger_main, turns_until_tau0=0, memory=memory)
    
    print(f"Generated {len(memory.ltmemory)} transitions.")

    # 3. TRAIN VALUE HEAD ONLY
    print("STEP 3: SURGICAL VALUE HEAD TRAINING (10 Epochs)")
    
    # Load existing model to recalibrate
    nn = Residual_CNN(cfg['rl']['learning_rate'], cfg['rl']['learning_rate'], env.input_shape, env.action_size)
    
    existing_models = [f for f in os.listdir(model_dir) if f.startswith(f"{SYM}_{TF}_v") and f.endswith('.keras')]
    if existing_models:
        existing_models.sort()
        latest = existing_models[-1]
        print(f"Recalibrating existing model: {latest}")
        m_tmp = tf.keras.models.load_model(model_dir + latest, custom_objects={'TransformerBlock': TransformerBlock, 'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})
        nn.model.set_weights(m_tmp.get_weights())
    else:
        print("Starting recalibration from fresh weights.")

    # FREEZE Policy Head
    nn.set_policy_head_trainable(False)
    
    # Train only on Value targets
    training_states = np.array([row['state'].binary for row in memory.ltmemory])
    z_values = np.array([row['value'] for row in memory.ltmemory])
    
    # Dummy policy targets (not used since loss weight is 0.0)
    dummy_pi = np.zeros((len(z_values), env.action_size))
    
    targets = {
        'value_head': z_values,
        'policy_head': dummy_pi
    }
    
    nn.model.fit(training_states, targets, epochs=10, batch_size=256, verbose=1)
    
    # SAVE Recalibrated Model
    diag_name = f"{SYM}_{TF}_diagnostic.keras"
    nn.model.save(model_dir + diag_name)
    print(f"RECALIBRATED MODEL SAVED TO {model_dir + diag_name}")
    
    # 4. VERIFICATION READY
    print("\nSTEP 4: VERIFICATION READY")
    print("Please run: python diagnostic_policy_regime.py")
    
    # 5. RESTORATION NOTE
    print("\nSTEP 5: RESTORATION")
    print("After verification, restart main.py. It will load the recalibrated model and resume.")

if __name__ == "__main__":
    main()
