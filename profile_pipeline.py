import cProfile
import pstats
import io
import pandas as pd
import yaml
import os
import tensorflow as tf
from game import TradingGame
from agent import Agent
from model import Residual_CNN, TransformerBlock
from loss import softmax_cross_entropy_with_logits
from funcs import playMatches
from settings import run_folder
import loggers as lg

def run_profiled_session():
    # 1. Setup Environment (Same as main.py)
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    data_path = os.path.join(cfg['trading'].get('data_path', 'data/training/BTC_USDT_1h'), 'train.csv')
    data = pd.read_csv(data_path)
    env = TradingGame(data, cfg)
    
    # 2. Load Latest Model
    model_dir = run_folder + 'models/'
    existing_models = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
    existing_models.sort()
    latest_model = existing_models[-1] if existing_models else None
    
    nn = Residual_CNN(cfg['rl']['learning_rate'], cfg['rl']['learning_rate'], env.input_shape, env.action_size)
    if latest_model:
        print(f"Profiling with Model: {latest_model}")
        m_tmp = tf.keras.models.load_model(model_dir + latest_model, custom_objects={
            'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits,
            'TransformerBlock': TransformerBlock
        })
        nn.model.set_weights(m_tmp.get_weights())
    
    player = Agent('profiler_agent', env.state_size, env.action_size, cfg['rl']['mcts_sims'], cfg['rl']['cpuct'], nn)

    # 3. Profiling Execution
    print("\n--- STARTING PROFILED EPISODE ---")
    pr = cProfile.Profile()
    pr.enable()
    
    # Run exactly 1 episode
    playMatches(env, player, player, 1, lg.logger_main, turns_until_tau0=cfg['rl']['turns_until_tau0'], memory=None)
    
    pr.disable()
    print("--- PROFILING COMPLETE ---\n")

    # 4. Generate Report
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(30) # Print top 30 for more detail
    print(s.getvalue())

    # 5. Summary Analysis
    # We look specifically for the heaviest hitters in our logic
    stats = pstats.Stats(pr)
    print("\n--- ARCHITECTURAL HOTSPOTS ---")
    print("Function Name                                     | Total Time | Call Count")
    print("-" * 75)
    
    hotspots = [
        ('model.py', 'call'),           # Transformer / CNN math
        ('agent.py', 'act'),            # Core MCTS control
        ('MCTS.py', 'moveToLeaf'),      # Tree traversal
        ('game.py', 'takeAction'),      # Environment logic
        ('game.py', '_generate_state_tensor'), # Data preprocessing
    ]
    
    for file, func in hotspots:
        for (f_file, line, f_func), (cc, nc, tt, ct, callers) in stats.stats.items():
            if func in f_func and file in f_file:
                print(f"{f_func:<45} | {ct:>8.3f}s | {nc:>10}")

if __name__ == "__main__":
    run_profiled_session()
