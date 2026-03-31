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

## Quick Start Guide

### 1. Installation
We recommend using a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install ccxt pandas pandas_ta PyYAML tensorflow keras matplotlib
```

### 2. Data Preparation
Fetch the latest market data and prepare the Walk-Forward split:
```bash
python3 data_fetcher.py
```
This generates `data/train.csv` and `data/val.csv`. You can edit `data_fetcher.py` to change the symbol or timeframe.

### 3. Run Training
Start the autonomous Self-Play and Retraining loop:
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 main.py
```

---

## How it Works: The Training Cycle

1.  **Self-Play (Data Collection)**:
    - The agent plays through random slices of `train.csv`.
    - It uses **MCTS** to explore future price movements and "labels" the best actions.
    - These experiences are stored in a **Memory Buffer**.
2.  **Retraining (The Learning)**:
    - Once the buffer hits the `memory_size` (e.g., 30,000 moves), the **Transformer** retrains its weights.
    - It learns to predict the optimal MCTS moves using only *past* data.
3.  **Tournament (The Evolution)**:
    - The new model plays against the current "Best" model on the **unseen `val.csv` data**.
    - If the new model achieves a significantly higher **Sharpe Ratio**, it is promoted to the new "Best Player."

---

## Configuration (`config.yaml`)
You can tune the engine without touching code:
- `trading`: Set fees, window size, and initial balance.
- `rl`: Control MCTS simulations, memory size, and learning rates.
- `model`: Adjust Transformer depth, number of heads, and dropout.
- `evaluation`: Set the tournament threshold and metrics.

---

## Running on Google Colab
1. Clone the repository in a cell: `!git clone <your-repo-url>`
2. Install dependencies: `!pip install ccxt pandas pandas_ta PyYAML tensorflow keras matplotlib`
3. Run data fetcher: `!python3 data_fetcher.py`
4. Run training:
   ```python
   import os
   os.environ['PYTHONPATH'] += ":."
   !python3 main.py
   ```
   *Colab's T4 GPU will significantly accelerate the Transformer retraining phase.*

---

## Project Structure
- **`games/trading/game.py`**: The "Market Game" environment and logic.
- **`model.py`**: The Transformer Encoder architecture.
- **`MCTS.py`**: Monte Carlo Tree Search (The Oracle).
- **`agent.py`**: Bridge between the Transformer and MCTS.
- **`data_fetcher.py`**: Technical indicator calculation and walk-forward splitting.
- **`run/`**: Output directory for models (`.keras`), logs, and memory snapshots.
