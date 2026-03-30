# AlphaZero Trader

An implementation of the AlphaZero reinforcement learning algorithm adapted for financial trading. The system treats the market as a single-player game where the agent optimizes for terminal portfolio value.

## Setup

1. **Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install ccxt pandas tensorflow keras matplotlib
   ```

2. **Data Fetching**:
   Before running, fetch the historical data you want to train on:
   ```bash
   python3 data_fetcher.py
   ```
   *Edit `data_fetcher.py` to change the symbol, timeframe, or exchange.*

3. **Configuration**:
   All hyperparameters (window size, initial balance, MCTS simulations) are located in `config.py`.

## Running the Engine

To start the self-play and training loop:
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 main.py
```

## Running on Google Colab

Google Colab is ideal for utilizing free GPUs to speed up training.

1.  **Open Colab**: Go to [colab.research.google.com](https://colab.research.google.com).
2.  **Upload Files**: Upload the project folder to your Google Drive or clone the repository directly in a cell:
    ```python
    !git clone <your-repo-url>
    %cd AlphaZeroTrader
    ```
3.  **Install Dependencies**:
    ```python
    !pip install ccxt pandas tensorflow keras matplotlib
    ```
4.  **Fetch Data**:
    ```python
    !python3 data_fetcher.py
    ```
5.  **Run Training**:
    Since `main.py` runs in a loop, it is best to run it as a background process or directly in a cell:
    ```python
    import os
    os.environ['PYTHONPATH'] += ":."
    !python3 main.py
    ```
    *Note: If using a GPU, Colab will automatically detect it and TensorFlow will utilize it for the Residual CNN.*

## Architecture

- **`games/trading/game.py`**: The core trading environment.
- **`model.py`**: Residual CNN that predicts market direction (Policy) and expected return (Value).
- **`MCTS.py`**: Monte Carlo Tree Search adapted for single-player environmental exploration.
- **`data_fetcher.py`**: Utility to download public market data via CCXT.
- **`run/`**: Contains logs, saved models, and memory buffers.
