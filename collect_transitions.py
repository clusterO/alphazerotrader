import numpy as np
import pandas as pd
import yaml
import os
from game import TradingGame
from transition_memory import TransitionMemory

def collect_transitions(limit_per_regime=5000):
    """
    Populate TransitionMemory by stepping through training regimes.
    This generates ground-truth (S, A, S') triples.
    """
    # 1. Load Config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    SYM = cfg['trading']['symbol'].replace("/", "_")
    TF = cfg['trading']['timeframe']
    run_folder = "./run/"
    trans_memory_path = f"{run_folder}memory/trans_memory_{SYM}_{TF}.p"
    
    data_dir = cfg['trading'].get('data_path', 'data/stable')
    regime_names = ['bear.csv', 'bull.csv', 'range.csv']
    train_files = [os.path.join(data_dir, 'training', f) for f in regime_names]

    memory = TransitionMemory(capacity=100000)
    if os.path.exists(trans_memory_path):
        memory.load(trans_memory_path)
        print(f"Loaded existing memory with {len(memory)} transitions.")

    # 2. Collect from each regime
    for f_path in train_files:
        if not os.path.exists(f_path):
            print(f"Skipping missing file: {f_path}")
            continue
            
        print(f"Collecting transitions from {f_path}...")
        df = pd.read_csv(f_path)
        # Force a single regime environment
        env = TradingGame(df, cfg)
        
        count = 0
        while count < limit_per_regime:
            state = env.reset()
            while not env.gameState.isEndGame and count < limit_per_regime:
                # Store current state tensor
                s_tensor = env.gameState.binary.copy()
                
                # Pick a random action to explore all transition types
                action = np.random.randint(0, env.action_size)
                
                # Step
                next_state_obj, reward, done, _ = env.step(action)
                
                # Store transition
                memory.append(s_tensor, action, next_state_obj.binary.copy())
                
                count += 1
                if count % 1000 == 0:
                    print(f"  Collected {count} transitions...")

        print(f"Completed collection for {f_path}.")

    # 3. Save
    if not os.path.exists(os.path.dirname(trans_memory_path)):
        os.makedirs(os.path.dirname(trans_memory_path))
    memory.save(trans_memory_path)
    print(f"Successfully saved {len(memory)} transitions to {trans_memory_path}")

if __name__ == "__main__":
    collect_transitions()
