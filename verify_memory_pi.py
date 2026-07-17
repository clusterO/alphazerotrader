# -*- coding: utf-8 -*-
import pickle
import numpy as np
import os
import yaml

def verify_memory():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    SYM = cfg['trading']['symbol'].replace("/", "_")
    TF = cfg['trading']['timeframe']
    run_folder = "./run/"
    memory_path = f"{run_folder}memory/memory_{SYM}_{TF}.p"

    if not os.path.exists(memory_path):
        print(f"Memory file not found: {memory_path}")
        return

    print(f"Loading memory: {memory_path}")
    with open(memory_path, "rb") as f:
        memory = pickle.load(f)

    ltmemory = memory.ltmemory
    print(f"Total memories in buffer: {len(ltmemory)}")

    bear_samples = []
    bull_samples = []

    print("Analyzing regimes...")
    for mem in ltmemory:
        # Check if 'state' object exists and has 'binary'
        if 'state' in mem and hasattr(mem['state'], 'binary'):
            state_tensor = mem['state'].binary
        elif 'board' in mem:
            state_tensor = mem['board']
        else:
            continue
            
        # Regime detection: Mean log return of the window (first column)
        avg_ret = np.mean(state_tensor[:, 0])
        
        # AV is the MCTS visit distribution (pi target)
        pi_target = mem['AV']
        
        if avg_ret < -0.0005: # BEAR Threshold
            bear_samples.append(pi_target)
        elif avg_ret > 0.0005: # BULL Threshold
            bull_samples.append(pi_target)

    print("\n" + "="*40)
    print("   MEMORY PI TARGET ANALYSIS")
    print("="*40)
    
    if bear_samples:
        bear_mean = np.mean(bear_samples, axis=0)
        print(f"Bear Regime (N={len(bear_samples)}):")
        print(f"  Flat:  {bear_mean[0]:.2%}")
        print(f"  Long:  {bear_mean[1]:.2%}")
        print(f"  Short: {bear_mean[2]:.2%}")
    else:
        print("No bear samples found.")

    if bull_samples:
        bull_mean = np.mean(bull_samples, axis=0)
        print(f"\nBull Regime (N={len(bull_samples)}):")
        print(f"  Flat:  {bull_mean[0]:.2%}")
        print(f"  Long:  {bull_mean[1]:.2%}")
        print(f"  Short: {bull_mean[2]:.2%}")
    else:
        print("No bull samples found.")
    print("="*40)

    print("\nVERDICT:")
    if bear_samples and bear_mean[2] > 0.15: 
        print(">>> SUCCESS: Stored pi targets show MCTS discovery signal.")
        print(f"    Short signal in Bear: {bear_mean[2]:.2%}")
        print("    The pipeline is working. You can proceed with the Policy Head Reset.")
    else:
        print(">>> FAILURE: Stored pi targets are still Flat-dominant.")
        print("    MCTS discovery is not reaching the memory buffer.")

if __name__ == "__main__":
    verify_memory()
