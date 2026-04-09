import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
import os
import matplotlib.pyplot as plt
from games.trading.game import TradingGame
from model import Residual_CNN, TransformerBlock
from loss import softmax_cross_entropy_with_logits
from agent import Agent
from settings import run_folder

def backtest(model_v=None, data_path='data/val.csv'):
    # 1. Load Config
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 2. Find Latest Model if not specified
    model_dir = run_folder + 'models/'
    if model_v is None:
        existing_models = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
        existing_models.sort()
        model_file = existing_models[-1]
    else:
        model_file = f"BTC_USDT_1h_v{str(model_v).zfill(4)}.keras"
    
    print(f"--- STARTING BACKTEST ---")
    print(f"Model: {model_file}")
    print(f"Data: {data_path}")
    
    # 3. Load Environment
    data = pd.read_csv(data_path)
    env = TradingGame(data, cfg)
    
    # 4. Load Model
    # We need to build the object first to get the correct shapes
    nn = Residual_CNN(cfg['rl']['learning_rate'], cfg['rl']['learning_rate'], env.input_shape, env.action_size)
    m_path = os.path.join(model_dir, model_file)
    m_tmp = tf.keras.models.load_model(m_path, custom_objects={
        'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits,
        'TransformerBlock': TransformerBlock
    })
    nn.model.set_weights(m_tmp.get_weights())
    
    agent = Agent('backtest_agent', env.state_size, env.action_size, cfg['rl']['mcts_sims'], cfg['rl']['cpuct'], nn)
    
    # 5. Run Walk-through
    state = env.reset()
    # Force start at the beginning of the window
    env.gameState.current_tick = env.window_size
    env.gameState.end_tick = len(data) - 1
    
    history = []
    done = False
    
    print("Running episode...", end="", flush=True)
    
    while not done:
        # Use tau=0 for deterministic best actions
        action, pi, mcts_v, nn_v = agent.act(env.gameState, 0)
        
        # Log before step
        price = data.iloc[env.gameState.current_tick]['close']
        pos = env.gameState.portfolio['position']
        bal = env.gameState.portfolio['balance']
        
        history.append({
            'tick': env.gameState.current_tick,
            'price': price,
            'position': pos,
            'balance': bal,
            'action': action,
            'mcts_value': mcts_v,
            'nn_value': nn_v
        })
        
        # Step
        _, reward, done, _ = env.step(action)
        if env.gameState.current_tick % 10 == 0: print(".", end="", flush=True)

    print(" Done.")
    
    df = pd.DataFrame(history)
    
    # 6. Metrics
    initial_bal = df['balance'].iloc[0]
    final_bal = df['balance'].iloc[-1]
    total_return = (final_bal / initial_bal) - 1
    
    # Daily returns for Sharpe (approx assuming 1h data)
    df['returns'] = df['balance'].pct_change().fillna(0)
    mu = df['returns'].mean()
    sigma = df['returns'].std()
    sharpe = (mu / (sigma + 1e-9)) * np.sqrt(252 * 24)
    
    # Drawdown
    df['cum_max'] = df['balance'].cummax()
    df['drawdown'] = (df['balance'] - df['cum_max']) / df['cum_max']
    max_dd = df['drawdown'].min()
    
    print("\n--- RESULTS ---")
    print(f"Initial Balance: {initial_bal:.2f}")
    print(f"Final Balance:   {final_bal:.2f}")
    print(f"Total Return:    {total_return*100:.2f}%")
    print(f"Max Drawdown:    {max_dd*100:.2f}%")
    print(f"Sharpe (ANN):    {sharpe:.4f}")
    
    # 7. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(df['tick'], df['price'], label='Price', color='black', alpha=0.3)
    # Highlight positions
    longs = df[df['position'] == 1]
    shorts = df[df['position'] == 2]
    ax1.scatter(longs['tick'], longs['price'], color='green', marker='^', label='Long', s=10)
    ax1.scatter(shorts['tick'], shorts['price'], color='red', marker='v', label='Short', s=10)
    ax1.set_ylabel('Price')
    ax1.legend()
    
    ax2.plot(df['tick'], df['balance'], label='Portfolio Value', color='blue')
    ax2.set_ylabel('Balance')
    ax2.set_xlabel('Tick')
    ax2.legend()
    
    plt.tight_layout()
    plot_path = f"backtest_v{model_file.split('_v')[-1].replace('.keras','')}.png"
    plt.savefig(plot_path)
    print(f"Saved plot to: {plot_path}")
    
    # Save CSV
    csv_path = f"backtest_trades_v{model_file.split('_v')[-1].replace('.keras','')}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved trades to: {csv_path}")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/test.csv'
    backtest(data_path=path)
