import os
import yaml
import time
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from dotenv import load_dotenv
from live_game import LiveTradingGame
from agent import Agent
from model import Residual_CNN, TransformerBlock, softmax_cross_entropy_with_logits
from natsort import natsorted

# Load environment variables
load_dotenv()

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def find_latest_model(symbol, timeframe):
    model_dir = 'run/models'
    if not os.path.exists(model_dir):
        return None
    sym = symbol.replace("/", "_")
    prefix = f"{sym}_{timeframe}_v"
    models = [f for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith('.keras')]
    if not models:
        return None
    return natsorted(models)[-1]

def journal_trade(tick_data):
    """
    Appends trade information to paper_trade_journal.csv
    """
    file_path = 'paper_trade_journal.csv'
    df = pd.DataFrame([tick_data])
    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)

def get_tf_seconds(tf_str):
    if tf_str.endswith('m'):
        return int(tf_str[:-1]) * 60
    elif tf_str.endswith('h'):
        return int(tf_str[:-1]) * 3600
    elif tf_str.endswith('d'):
        return int(tf_str[:-1]) * 86400
    return 60 # Default to 1m

def main():
    config = load_config()
    SYM = config['trading']['symbol']
    TF = config['trading']['timeframe']
    
    # 1. Initialize Live Environment
    exchange_config = {
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_API_SECRET'),
    }
    
    game = LiveTradingGame(config, exchange_config)
    
    # 2. Load Latest Agent
    latest_model_name = find_latest_model(SYM, TF)
    if not latest_model_name:
        print(f"❌ No model found for {SYM} {TF} in run/models. Waiting for model...")
        # We don't exit, we wait for a model to appear
        while not latest_model_name:
            time.sleep(10)
            latest_model_name = find_latest_model(SYM, TF)

    print(f"🤖 Loading Agent with model: {latest_model_name}")
    
    model_path = os.path.join('run/models', latest_model_name)
    # Using the same loading logic as main.py
    m_tmp = tf.keras.models.load_model(model_path, custom_objects={
        'TransformerBlock': TransformerBlock, 
        'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits
    })
    
    nn = Residual_CNN(config['rl']['learning_rate'], config['rl']['learning_rate'], game.input_shape, game.action_size)
    nn.model.set_weights(m_tmp.get_weights())
    
    agent = Agent('paper_trader', game.state_size, game.action_size, config['rl']['mcts_sims'], config['rl']['cpuct'], nn)
    
    print(f"🚀 Paper Trader Started. Symbol: {SYM} | Timeframe: {TF}")
    
    while True:
        try:
            # Check for config updates (UI might have changed symbol/TF)
            new_config = load_config()
            if new_config['trading']['symbol'] != SYM or new_config['trading']['timeframe'] != TF:
                print(f"🔄 Configuration changed to {new_config['trading']['symbol']} {new_config['trading']['timeframe']}. Restarting trader...")
                # Exit and let the C2 server restart us, or just re-initialize
                # For simplicity in this script, we re-initialize
                SYM = new_config['trading']['symbol']
                TF = new_config['trading']['timeframe']
                config = new_config
                game = LiveTradingGame(config, exchange_config)
                latest_model_name = None # Force reload

            # Check for newer model every tick
            current_model_name = find_latest_model(SYM, TF)
            if current_model_name and current_model_name != latest_model_name:
                print(f"🔄 Hot-swapping to new model: {current_model_name}")
                m_path = os.path.join('run/models', current_model_name)
                m_new = tf.keras.models.load_model(m_path, custom_objects={
                    'TransformerBlock': TransformerBlock, 
                    'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits
                })
                agent.model.model.set_weights(m_new.get_weights())
                latest_model_name = current_model_name

            # Fetch current state from exchange
            state = game.fetch_latest_state()
            
            # Agent decides action via MCTS (tau=0 for deterministic competitive play)
            action, pi, mcts_v, nn_v = agent.act(state, tau=0)
            
            # Execute on Binance Demo
            game.execute_action(action)
            
            # Journal results
            current_price = state.close_prices[state.current_tick]
            journal_trade({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'price': current_price,
                'position': state.portfolio['position'],
                'balance': state.portfolio['balance'],
                'action': action,
                'mcts_v': float(mcts_v),
                'nn_v': float(nn_v),
                'model': latest_model_name,
                'pi': str(pi.tolist())
            })
            
            # Sleep until the next candle
            tf_seconds = get_tf_seconds(TF)
            now = time.time()
            sleep_time = tf_seconds - (now % tf_seconds) + 5 # Add 5s buffer for candle closing
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {SYM} @ {current_price:.2f} | Pos: {state.portfolio['position']} | Act: {action} | Next in {sleep_time:.1f}s")
            time.sleep(sleep_time)

        except Exception as e:
            print(f"⚠️ Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60)

if __name__ == "__main__":
    main()
