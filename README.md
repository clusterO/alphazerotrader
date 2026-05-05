# AlphaZero Trader v3

An autonomous, risk-aware trading engine utilizing the AlphaZero reinforcement learning framework. The system treats financial markets as a single-player "market game" and optimizes for risk-adjusted returns (Sharpe Ratio) using a Transformer-based architecture.

## Current Project State: AlphaZero Trader v3
This version (v3) is a major overhaul that addresses the critical flaws of naive AlphaZero-to-Trading transitions:
- **No-Peek Inference**: MCTS is used as an "Oracle" during training but is disabled during evaluation/live trading to prevent lookahead bias.
- **Transformer Backbone**: Replaced the ResNet CNN with a **Transformer Encoder** to better capture long-range temporal dependencies in price action.
- **Risk-Adjusted Learning**: The system optimizes for **Sharpe Ratio** instead of raw profit, preferring stable growth over volatile gambles.
- **Realistic Friction**: Implements **Transaction Costs** (default 5 bps) and holding penalties to prevent overtrading.
- **Rich State Representation**: Input features include RSI, ATR (Volatility), Rolling Volatility, and Volume Delta, alongside a detailed Portfolio State (Position Age, PnL, Episode Progress).

---

## Data Management

The system uses a unified `data_manager.py` to handle all data requirements. 

### 1. Fetching Training Data
To prepare a new dataset for training (automatically creates 80/10/10 splits):
```bash
./venv/bin/python3 data_manager.py --symbol BTC/USDT --timeframe 5m --train --limit 15000
```

### 2. Fetching Backtest Data
To fetch data for a specific regime or test case:
```bash
./venv/bin/python3 data_manager.py --symbol BTC/USDT --timeframe 1h --limit 3000 --description "bear_market"
```

### 3. Switching Instruments/Timeframes
Everything is driven by `config.yaml`. To switch assets, update the `data_path`, `symbol`, and `timeframe` fields. The system automatically separates models and memories by these identifiers.

---

## Quick Start Guide

### 1. Installation
We recommend using a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install ccxt pandas pandas_ta PyYAML tensorflow keras matplotlib
```

### 2. Run Training
Start the autonomous Self-Play and Retraining loop:
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 main.py
```

---

## How it Works: The Training Cycle

1.  **Self-Play (Data Collection)**:
    - The agent plays through random slices of training data.
    - It uses **MCTS** to explore future price movements and "labels" the best actions.
    - These experiences are stored in a **Memory Buffer**.
2.  **Retraining (The Learning)**:
    - Once the buffer hits a threshold, the **Transformer** retrains its weights.
    - It learns to predict the optimal MCTS moves using only *past* data.
3.  **Tournament (The Evolution)**:
    - The new model plays against the current "Best" model on the **unseen validation data**.
    - If the new model achieves a significantly higher **Sharpe Ratio**, it is promoted to the new "Best Player."

---

## Configuration (`config.yaml`)
You can tune the engine without touching code:
- `trading`: Set symbols, timeframes, data paths, fees, and window size.
- `rl`: Control MCTS simulations, memory size, and learning rates.
- `model`: Adjust Transformer depth, number of heads, and dropout.
- `evaluation`: Set the tournament threshold and metrics.

---

## Training Configuration: 1h vs 5m Granularity

When switching between timeframes, specific parameters must be adjusted to account for the difference in price move volatility and frequency of signals.

### 1h Granularity (Default)
- **Transaction Fee**: 0.05% (5 bps). Hourly moves are large enough to absorb this cost.
- **Idling Penalty**: 0.0001 (1 bp per hour). Forces the agent to find trades within daily cycles.
- **Sharpe Annualization**: Traditionally uses $\sqrt{252 \times 24} \approx 77.7$.
- **Reward Scaling**: `tanh(sharpe / 2.0)` is effective as hourly Sharpe values are relatively stable.

### 5m Granularity
- **Transaction Fee**: 0.00015 (1.5 bps). 5m moves are much smaller; higher fees will cause the agent to learn that all trading is negative.
- **Idling Penalty**: 0.00001 (0.1 bp per 5m candle). High frequency "do nothing" penalties will drain the account faster than the agent can learn.
- **Sharpe Annualization**: For tournaments, **Raw Episode Sharpe** (no annualization factor) is preferred for a cleaner ranking signal. 
- **Reward Scaling**: `tanh(raw_sharpe * 3.0)`. Removing the large annualization factor from the reward loop requires increasing the multiplier to maintain gradient signal without saturating at -1.0 or 1.0.
- **Episode Length**: [150, 400] candles (approx 12-33 hours). Ensures the agent sees multiple micro-trends within a single episode.

---

## Project Structure
- **`data_manager.py`**: Centralized data fetcher and processor.
- **`game.py`**: The "Market Game" environment and logic.
- **`model.py`**: The Transformer Encoder architecture.
- **`MCTS.py`**: Monte Carlo Tree Search (The Oracle).
- **`agent.py`**: Bridge between the Transformer and MCTS.
- **`run/`**: Output directory for models (`.keras`), logs, and memory snapshots.
