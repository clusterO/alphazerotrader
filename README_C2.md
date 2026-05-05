# AZT C2 Command Center & Paper Trader (Updated)

This system provides a modern web interface and a live paper trading engine for the AlphaZero Trading agent.

## Components

1.  **C2 Backend (FastAPI):** Manages processes, updates configuration, and streams logs.
2.  **C2 Frontend (Next.js):** A futuristic dashboard for command and control.
3.  **Paper Trader:** A live execution engine using CCXT and Binance Demo.
4.  **Refactored Trainer:** A controllable `train.py` script.

## Setup

### 1. Environment Variables
Create a `.env` file in the root directory (use `.env.example` as a template):
```bash
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
BINANCE_TESTNET=True
```

### 2. Backend Requirements
Install the new dependencies:
```bash
pip install fastapi uvicorn ccxt pandas-ta sqlmodel psutil websockets aiofiles
```

### 3. Run the Backend
```bash
python3 c2_server/main.py
```
The API will be available at `http://localhost:8000`.

### 4. Run the Frontend
Navigate to the `c2_ui` directory and run:
```bash
npm install
npm run dev
```
The dashboard will be available at `http://localhost:3000`.

## Features

- **Real-time Logs:** See training and trading logs as they happen via WebSockets.
- **Process Management:** Start and stop the Training Engine or Paper Trader with a single click.
- **Live Execution:** The Paper Trader fetches live candles (based on configuration), calculates features, and executes Market orders on Binance Spot (Demo).
- **Configuration Hub:** View and update `config.yaml` parameters directly from the UI.

## File Structure

- `train.py`: Standalone training loop.
- `paper_trader.py`: Live trading loop.
- `live_game.py`: Real-time trading environment.
- `c2_server/`: FastAPI backend source.
- `c2_ui/`: Next.js frontend source.
