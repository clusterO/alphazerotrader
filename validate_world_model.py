import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
import os
from world_model import TransitionModel
from transition_memory import TransitionMemory

# Try to import advanced stats libs
try:
    from statsmodels.tsa.stattools import acf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

def validate_world_model():
    # 1. Load Config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    SYM = cfg['trading']['symbol'].replace("/", "_")
    TF = cfg['trading']['timeframe']
    run_folder = "./run/"
    trans_memory_path = f"{run_folder}memory/trans_memory_{SYM}_{TF}.p"
    model_path = f"{run_folder}models/world_model_{SYM}_{TF}.keras"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # 2. Load Data (Held-out set)
    memory = TransitionMemory()
    memory.load(trans_memory_path)
    
    total = len(memory)
    if total == 0:
        print("Error: Transition memory is empty.")
        return

    # We take the last 10% as the held-out set
    split = int(0.9 * total)
    held_out_indices = list(range(split, total))
    
    if len(held_out_indices) < 200:
        print(f"Warning: Held-out set is small ({len(held_out_indices)} samples). Results may be noisy.")

    # 3. Load Model
    window_size = cfg['trading']['window_size']
    n_features = 19 
    
    wm = TransitionModel((window_size, n_features), 3) 
    wm.model = tf.keras.models.load_model(model_path)
    
    # 4. Generate Residuals for Log Return (Feature Index 0)
    residuals = []
    
    print(f"Generating residuals on {len(held_out_indices)} held-out transitions...")
    
    # Pre-batching for speed if held-out set is large
    batch_size = 256
    for i in range(0, len(held_out_indices), batch_size):
        idx_batch = held_out_indices[i:i+batch_size]
        states = np.array([memory.buffer[idx]['state'] for idx in idx_batch])
        actions = [memory.buffer[idx]['action'] for idx in idx_batch]
        actions_onehot = np.eye(3)[actions]
        next_states_actual = np.array([memory.buffer[idx]['next_state'] for idx in idx_batch])
        
        preds = wm.model.predict([states, actions_onehot], verbose=0)
        
        # log_return is at index 0
        # We check the very last timestep of the prediction
        actual_returns = next_states_actual[:, -1, 0]
        pred_returns = preds[:, -1, 0]
        
        batch_residuals = actual_returns - pred_returns
        residuals.extend(batch_residuals)
    
    residuals = np.array(residuals)
    
    # --- CHECK 1: Residual Autocorrelation ---
    print("\n" + "="*60)
    print("GATE 1: CHECK 1 - RESIDUAL AUTOCORRELATION")
    print("="*60)
    
    def manual_acf(x, lag):
        if lag == 0: return 1.0
        x_mean = np.mean(x)
        num = np.sum((x[:-lag] - x_mean) * (x[lag:] - x_mean))
        den = np.sum((x - x_mean)**2) + 1e-9
        return num / den

    use_statsmodels = HAS_STATSMODELS
    if use_statsmodels:
        lags = acf(residuals, nlags=3)
        lag1, lag2, lag3 = lags[1], lags[2], lags[3]
    else:
        lag1 = manual_acf(residuals, 1)
        lag2 = manual_acf(residuals, 2)
        lag3 = manual_acf(residuals, 3)

    print(f"Lag-1 Autocorrelation: {lag1:+.4f}")
    print(f"Lag-2 Autocorrelation: {lag2:+.4f}")
    print(f"Lag-3 Autocorrelation: {lag3:+.4f}")
    
    if abs(lag1) > 0.2:
        print("VERDICT: [FAIL]")
        if lag1 > 0.2:
            print("  >> Model UNDERESTIMATES momentum. MCTS will find fake persistent trends.")
        else:
            print("  >> Model OVERESTIMATES mean-reversion. MCTS will time fake reversals.")
    else:
        print("VERDICT: [PASS] Errors look like white noise.")

    # --- CHECK 2: Volatility Clustering (GARCH Persistence) ---
    print("\n" + "="*60)
    print("GATE 1: CHECK 2 - VOLATILITY CLUSTERING")
    print("="*60)
    
    use_arch = HAS_ARCH
    if use_arch:
        # Rescale residuals to help GARCH convergence (scale doesn't affect alpha+beta)
        res_scaled = residuals * 100.0
        try:
            am = arch_model(res_scaled, vol='Garch', p=1, q=1, dist='Normal', rescale=False)
            res_garch = am.fit(disp='off')
            
            alpha = res_garch.params['alpha[1]']
            beta = res_garch.params['beta[1]']
            persistence = alpha + beta
            
            print(f"GARCH Alpha: {alpha:.4f}")
            print(f"GARCH Beta:  {beta:.4f}")
            print(f"Persistence (Alpha + Beta): {persistence:.4f}")
            
            if persistence > 0.7:
                print("VERDICT: [PASS] Model captures volatility clustering.")
            else:
                print("VERDICT: [FAIL] Persistence below 0.7. Model ignores regime structure.")
        except Exception as e:
            print(f"GARCH Fit Error: {e}. Falling back to squared residual AC.")
            use_arch = False

    if not use_arch:
        # Fallback: Check autocorrelation of squared residuals
        # Volatility clustering implies that squared residuals are correlated
        sq_res = residuals**2
        sq_lag1 = manual_acf(sq_res, 1)
        print(f"Squared Residuals Lag-1 AC: {sq_lag1:.4f}")
        if sq_lag1 > 0.15: 
            print("VERDICT: [PASS] (Fallback) Volatility clustering detected via squared AC.")
        else:
            print("VERDICT: [FAIL] No volatility clustering detected.")

    print("="*60 + "\n")

if __name__ == "__main__":
    validate_world_model()
