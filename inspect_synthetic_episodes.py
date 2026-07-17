import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
import os
import random
from world_model import TransitionModel
from agent import Agent
from model import Residual_CNN, TransformerBlock, softmax_cross_entropy_with_logits
from game import TradingGame, GameState

# Try to import arch for Criterion 2
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

def inspect_synthetic_episodes():
    # 1. Load Config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    SYM = cfg['trading']['symbol'].replace("/", "_")
    TF = cfg['trading']['timeframe']
    run_folder = "./run/"
    model_path = f"{run_folder}models/world_model_{SYM}_{TF}.keras"
    
    data_dir = cfg['trading'].get('data_path', 'data/stable')
    bear_file = os.path.join(data_dir, 'training', 'bear.csv')
    bull_file = os.path.join(data_dir, 'training', 'bull.csv')

    if not os.path.exists(model_path):
        print(f"Error: Transition Model not found at {model_path}")
        return

    # 2. Load Models
    print("Loading models...")
    # Transition Model
    window_size = cfg['trading']['window_size']
    n_features = 19
    
    wm = TransitionModel((window_size, n_features), 3)
    wm.model = tf.keras.models.load_model(model_path)
    
    # Agent (Policy for rollouts)
    dummy_env = TradingGame([pd.read_csv(bear_file)], cfg)
    nn = Residual_CNN(cfg['rl']['learning_rate'], cfg['rl']['learning_rate'], dummy_env.input_shape, dummy_env.action_size)
    
    # Load latest best model for the agent
    model_dir = run_folder + 'models/'
    existing_models = [f for f in os.listdir(model_dir) if f.startswith(f"{SYM}_{TF}_v") and f.endswith('.keras')]
    if existing_models:
        existing_models.sort()
        latest_model = existing_models[-1]
        print(f"Loading agent policy from {latest_model}")
        m_tmp = tf.keras.models.load_model(model_dir + latest_model, custom_objects={'TransformerBlock': TransformerBlock, 'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})
        nn.model.set_weights(m_tmp.get_weights())
    
    agent = Agent('inspector', dummy_env.state_size, dummy_env.action_size, 0, cfg['rl']['cpuct'], nn)

    # 3. Compute High Volatility Threshold (Criterion 4)
    print("Computing volatility distribution...")
    all_returns = []
    for f in [bear_file, bull_file]:
        df = pd.read_csv(f)
        all_returns.extend(df.iloc[:, 1].tolist()) # Assuming log_return is col 1 (index 0 in tensor)
    
    # Realized vol in 60-step windows
    vol_dist = []
    for i in range(len(all_returns) - 60):
        window = all_returns[i:i+60]
        vol_dist.append(np.std(window))
    vol_threshold = np.percentile(vol_dist, 80)
    print(f"80th Percentile Volatility Threshold: {vol_threshold:.4f}")

    # 4. Perform Rollouts
    def run_rollouts(file_path, regime_name, n=10, steps=50, force_action=None):
        df = pd.read_csv(file_path)
        env = TradingGame(df, cfg)
        
        # Sample starts from the usable range of the processed data
        total_usable = len(env.market_data_array)
        possible_starts = list(range(env.window_size, total_usable - steps - 60))
        sampled_starts = random.sample(possible_starts, n)
        
        episodes = []

        for start_idx in sampled_starts:
            # Reconstruct the starting GameState correctly at the sampled tick
            state = env.reset(start_tick=start_idx)
            
            traj_returns = []
            traj_prices = [] # To check for negative/crazy moves
            start_vol = np.std(state.binary[:, 0])
            
            curr_state_tensor = state.binary.copy()
            
            # Simple mock state for the agent's policy head
            class MockState:
                def __init__(self, tensor):
                    self.binary = tensor
                    self.allowedActions = [0, 1, 2]
            
            for s in range(steps):
                # 1. Get action (Policy or Forced)
                if force_action is not None:
                    action = force_action
                else:
                    action, _, _, _ = agent.act(MockState(curr_state_tensor), tau=0)
                
                # 2. Predict next state
                next_state_tensor = wm.predict(curr_state_tensor, action)
                
                # 3. Extract results (last row of predicted window)
                ret = next_state_tensor[-1, 0]
                price = next_state_tensor[-1, 1] 
                
                traj_returns.append(ret)
                traj_prices.append(price)
                
                # 4. Update for next step
                curr_state_tensor = next_state_tensor
            
            episodes.append({
                'returns': np.array(traj_returns),
                'prices': np.array(traj_prices),
                'start_vol': start_vol
            })
        return episodes

    print("Generating synthetic Bear episodes (Forced Short)...")
    bear_episodes = run_rollouts(bear_file, "Bear", force_action=2)
    print("Generating synthetic Bull episodes (Forced Long)...")
    bull_episodes = run_rollouts(bull_file, "Bull", force_action=1)

    # 5. Evaluate Criteria
    results = {
        'C1_Bear': 0, 'C1_Bull': 0,
        'C2_Count': 0,
        'C3_Bear_DD': 0,
        'C4_Violations': 0
    }

    print("\n" + "="*60)
    print("GATE 2: SYNTHETIC EPISODE INSPECTION")
    print("="*60)

    # Criterion 1 & 3
    for ep in bear_episodes:
        if np.mean(ep['returns']) < 0: results['C1_Bear'] += 1
        
        # Max Drawdown
        cum_ret = np.cumsum(ep['returns'])
        cum_prices = np.exp(cum_ret) # Reconstruct relative price path
        peak = np.maximum.accumulate(cum_prices)
        drawdown = (peak - cum_prices) / peak
        if np.max(drawdown) >= 0.05: results['C3_Bear_DD'] += 1

    for ep in bull_episodes:
        if np.mean(ep['returns']) > 0: results['C1_Bull'] += 1

    # Criterion 2 (Volatility Clustering)
    all_episodes = bear_episodes + bull_episodes
    for ep in all_episodes:
        if HAS_ARCH:
            try:
                res_scaled = ep['returns'] * 100.0
                am = arch_model(res_scaled, vol='Garch', p=1, q=1, dist='Normal', rescale=False)
                res_garch = am.fit(disp='off')
                persistence = res_garch.params['alpha[1]'] + res_garch.params['beta[1]']
                if persistence > 0.8: results['C2_Count'] += 1
            except:
                pass
        else:
            # Fallback: AC of squared returns
            sq_ret = ep['returns']**2
            ac1 = np.corrcoef(sq_ret[:-1], sq_ret[1:])[0,1]
            if ac1 > 0.2: results['C2_Count'] += 1

    # Criterion 4 (Impossible Paths)
    for ep in all_episodes:
        # No negative prices (normalized/raw)
        # Usually prices in state are normalized > 0, but we check if they go crazy negative
        if np.any(ep['prices'] < -5.0): # Threshold for normalized price floor
             results['C4_Violations'] += 1
        
        # No single-step move > 15.0 (Normalized Threshold)
        # Features are divided by mean absolute return, so 15.0 means 15x average volatility.
        for r in ep['returns']:
            if abs(r) > 15.0:
                if ep['start_vol'] < vol_threshold:
                    results['C4_Violations'] += 1

    # Print Verdicts
    print(f"C1 (Directional): Bear {results['C1_Bear']}/10, Bull {results['C1_Bull']}/10")
    print(f"C2 (Vol Clustering): {results['C2_Count']}/20 episodes passed")
    print(f"C3 (Bear Drawdowns): {results['C3_Bear_DD']}/10 episodes had >5% DD")
    print(f"C4 (Impossible Paths): {results['C4_Violations']} violations detected")

    print("\nFINAL VERDICT:")
    c1_pass = (results['C1_Bear'] >= 7 and results['C1_Bull'] >= 7)
    c2_pass = (results['C2_Count'] >= 11) # Majority
    c3_pass = (results['C3_Bear_DD'] >= 5)
    c4_pass = (results['C4_Violations'] == 0)

    if c1_pass and c2_pass and c3_pass and c4_pass:
        print(">>> ALL CRITERIA PASSED. Model is ready for Training Test.")
    else:
        if not c1_pass: print("- FAILED C1: Directional bias is weak. Deepen temporal encoder.")
        if not c2_pass: print("- FAILED C2: Volatility is too constant. Need more high-vol training data.")
        if not c3_pass: print("- FAILED C3: Bear episodes are too 'safe'. Check reward scaling/tail data.")
        if not c4_pass: print("- FAILED C4: Numerical instability detected in paths.")

    # --- CROSS-REGIME ACTION TEST ---
    print("\n" + "="*60)
    print("CROSS-REGIME ACTION TEST (Diagnostic)")
    print("="*60)
    
    print("Generating synthetic Bear + LONG episodes...")
    bear_long_episodes = run_rollouts(bear_file, "Bear", force_action=1)
    
    print("Generating synthetic Bull + SHORT episodes...")
    bull_short_episodes = run_rollouts(bull_file, "Bull", force_action=2)
    
    bear_long_mean = np.mean([np.mean(ep['returns']) for ep in bear_long_episodes])
    bull_short_mean = np.mean([np.mean(ep['returns']) for ep in bull_short_episodes])
    
    print(f"Bear + LONG  Mean Return: {bear_long_mean:+.6f}")
    print(f"Bull + SHORT Mean Return: {bull_short_mean:+.6f}")
    
    print("\nInterpretation:")
    if bear_long_mean < 0:
        print("  [OK] Bear regime persists despite LONG actions.")
    else:
        print("  [CAUTION] LONG actions are flipping the Bear regime to Bull.")
        
    if bull_short_mean > 0:
        print("  [OK] Bull regime persists despite SHORT actions.")
    else:
        print("  [CAUTION] SHORT actions are flipping the Bull regime to Bear.")

if __name__ == "__main__":
    inspect_synthetic_episodes()
