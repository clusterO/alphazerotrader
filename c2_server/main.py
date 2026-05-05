from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import yaml
import os
import subprocess
import signal
import psutil
import asyncio
import sys
from datetime import datetime
from typing import Dict, Any, List

app = FastAPI(title="AZT C2 Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
# Tracking PIDs for managed processes
process_store: Dict[str, Any] = {
    "train": None,
    "paper": None
}

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

async def tail_logs(filename: str, proc_type: str):
    if not os.path.exists(filename):
        with open(filename, "w") as f: f.write(f"--- {proc_type} logs start ---\n")
    
    with open(filename, "r") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if line:
                await manager.broadcast(f"[{proc_type.upper()}] {line.strip()}")
            await asyncio.sleep(0.1)

@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def get_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def save_config(config: dict):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)

@app.get("/config")
def read_config():
    return get_config()

@app.patch("/config")
def update_config(config_update: dict):
    current = get_config()
    for key, value in config_update.items():
        if isinstance(value, dict) and key in current:
            current[key].update(value)
        else:
            current[key] = value
    save_config(current)
    return current

@app.get("/market")
def get_market_data():
    try:
        cfg = get_config()
        symbol = cfg['trading']['symbol']
        import ccxt
        exch = ccxt.binance()
        ticker = exch.fetch_ticker(symbol)
        return {
            "symbol": symbol,
            "price": ticker['last'],
            "change": ticker['percentage']
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/models")
def list_models():
    model_dir = os.path.join(PROJECT_ROOT, "run/models/")
    if not os.path.exists(model_dir):
        return []
    models = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
    models.sort(reverse=True)
    return models

@app.get("/memory")
def list_memory():
    mem_dir = os.path.join(PROJECT_ROOT, "run/memory/")
    if not os.path.exists(mem_dir):
        return []
    mems = [f for f in os.listdir(mem_dir) if f.endswith('.p')]
    mems.sort(reverse=True)
    return mems

@app.get("/data_files")
def list_data():
    data_dir = os.path.join(PROJECT_ROOT, "data/")
    if not os.path.exists(data_dir):
        return []
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    files.sort()
    return files

@app.get("/trades/{type}")
def get_trades(type: str):
    import pandas as pd
    if type == "paper":
        path = os.path.join(PROJECT_ROOT, "paper_trade_journal.csv")
    else:
        # Get the latest backtest file
        backtest_files = [f for f in os.listdir(PROJECT_ROOT) if f.startswith("backtest_trades_v") and f.endswith(".csv")]
        if not backtest_files:
            return []
        backtest_files.sort(reverse=True)
        path = os.path.join(PROJECT_ROOT, backtest_files[0])
    
    if not os.path.exists(path):
        return []
    
    try:
        # Load with strings for 'pi' if it exists to avoid parsing errors
        df = pd.read_csv(path, on_bad_lines='skip')
        # Ensure we don't return too much data
        df = df.tail(100)
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"Error reading trades: {e}")
        return []

@app.get("/logs/tail/{process}")
def tail_log_endpoint(process: str, lines: int = 100):
    log_file = os.path.join(PROJECT_ROOT, f"{process}.log")
    if not os.path.exists(log_file):
        return []
    try:
        with open(log_file, "r") as f:
            content = f.readlines()
            return content[-lines:]
    except:
        return []

@app.get("/status")
def get_status():
    status = {}
    for proc_name, pid in process_store.items():
        if pid and psutil.pid_exists(pid):
            p = psutil.Process(pid)
            status[proc_name] = "running" if p.status() != psutil.STATUS_ZOMBIE else "stopped"
        else:
            status[proc_name] = "stopped"
    return status

@app.get("/logs/{process}")
def get_recent_logs(process: str):
    log_file = os.path.join(PROJECT_ROOT, f"{process}.log")
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            return f.readlines()[-100:]
    return []

@app.delete("/logs/{process}")
def clear_logs(process: str):
    log_file = os.path.join(PROJECT_ROOT, f"{process}.log")
    if os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write(f"--- Logs cleared at {datetime.now()} ---\n")
    return {"message": "Logs cleared"}

@app.post("/control/{process}/{action}")
async def control_process(process: str, action: str, background_tasks: BackgroundTasks):
    if process not in ["train", "paper"]:
        raise HTTPException(status_code=400, detail="Invalid process name")
    
    log_file = f"{process}.log"
    flag_name = "stop_train.flag" if process == "train" else "stop_paper.flag"
    flag_path = os.path.join(PROJECT_ROOT, flag_name)

    if action == "start":
        if process_store[process] and psutil.pid_exists(process_store[process]):
            return {"message": f"{process} is already running"}
        
        if os.path.exists(flag_path):
            os.remove(flag_path)

        script_name = "train.py" if process == "train" else "paper_trader.py"
        script_path = os.path.join(PROJECT_ROOT, script_name)
        log_path = os.path.join(PROJECT_ROOT, log_file)
        
        with open(log_path, "a") as f:
            f.write(f"\n--- Process Started at {datetime.now()} ---\n")
            proc = subprocess.Popen([sys.executable, "-u", script_path], stdout=f, stderr=f, cwd=PROJECT_ROOT)
        
        process_store[process] = proc.pid
        background_tasks.add_task(tail_logs, log_path, process)
        return {"message": f"Started {process}", "pid": proc.pid}
    
    elif action == "stop":
        pid = process_store[process]
        if pid and psutil.pid_exists(pid):
            with open(flag_path, "w") as f: f.write("stop")
            process_store[process] = None
            return {"message": f"Stop signal sent to {process}"}
        return {"message": f"{process} is not running"}
    
    raise HTTPException(status_code=400, detail="Invalid action")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
