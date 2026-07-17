import numpy as np
import yaml
import os
import tensorflow as tf
from world_model import TransitionModel
from transition_memory import TransitionMemory
import matplotlib.pyplot as plt

def train_world_model():
    # 1. Load Config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    SYM = cfg['trading']['symbol'].replace("/", "_")
    TF = cfg['trading']['timeframe']
    run_folder = "./run/"
    trans_memory_path = f"{run_folder}memory/trans_memory_{SYM}_{TF}.p"
    model_save_path = f"{run_folder}models/world_model_{SYM}_{TF}.keras"

    if not os.path.exists(trans_memory_path):
        print(f"Error: Transition memory not found at {trans_memory_path}. Run some self-play episodes first.")
        return

    # 2. Load Transition Memory
    print(f"Loading transition memory from {trans_memory_path}...")
    memory = TransitionMemory()
    memory.load(trans_memory_path)
    print(f"Memory size: {len(memory)}")

    if len(memory) < 1000:
        print("Warning: Memory size too small for training. Need at least 1000 transitions.")
        # We'll continue but results might be poor

    # 3. Prepare Data
    # Split into train and validation
    all_indices = list(range(len(memory)))
    np.random.shuffle(all_indices)
    
    split = int(0.9 * len(memory))
    train_indices = all_indices[:split]
    val_indices = all_indices[split:]
    
    def get_data(indices):
        batch = [memory.buffer[i] for i in indices]
        states = np.array([t['state'] for t in batch])
        next_states = np.array([t['next_state'] for t in batch])
        actions_idx = [t['action'] for t in batch]
        actions_onehot = np.eye(3)[actions_idx]
        return [states, actions_onehot], next_states

    train_X, train_Y = get_data(train_indices)
    val_X, val_Y = get_data(val_indices)

    # 4. Initialize Model
    input_shape = train_X[0].shape[1:] # (60, 19)
    wm = TransitionModel(input_shape, action_size=3, learning_rate=0.0001)
    
    print("Transition Model Architecture:")
    wm.model.summary()

    # 5. Training Loop
    epochs = 100
    batch_size = 64
    patience = 5
    best_val_loss = float('inf')
    wait = 0

    history = {'loss': [], 'val_loss': []}

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        # Shuffle train data
        perm = np.random.permutation(len(train_indices))
        train_X_shuf = [train_X[0][perm], train_X[1][perm]]
        train_Y_shuf = train_Y[perm]
        
        # Train on batches
        losses = []
        for i in range(0, len(train_indices), batch_size):
            X_batch = [train_X_shuf[0][i:i+batch_size], train_X_shuf[1][i:i+batch_size]]
            Y_batch = train_Y_shuf[i:i+batch_size]
            
            loss = wm.model.train_on_batch(X_batch, Y_batch)
            losses.append(loss)
        
        avg_train_loss = np.mean(losses)
        val_loss = wm.model.evaluate(val_X, val_Y, verbose=0)
        
        history['loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.6f} - val_loss: {val_loss:.6f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wm.save(model_save_path)
            print(f"  [SAVED] New best validation loss: {best_val_loss:.6f}")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [STOP] Early stopping at epoch {epoch+1}")
                break

    # 6. Plot Results
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Transition Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.yscale('log')
    plot_path = f"{run_folder}world_model_training.png"
    plt.savefig(plot_path)
    print(f"Saved training plot to {plot_path}")

if __name__ == "__main__":
    train_world_model()
