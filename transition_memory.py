import numpy as np
from collections import deque
import pickle
import os
import random

class TransitionMemory:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def append(self, state, action, next_state):
        """
        Add a (S, A, S') transition to the buffer.
        state, next_state: (window_size, n_features)
        action: int
        """
        self.buffer.append({
            'state': state,
            'action': action,
            'next_state': next_state
        })

    def sample(self, batch_size):
        """
        Sample a random batch of transitions.
        Returns:
            states: (batch_size, window_size, n_features)
            actions: (batch_size, action_size) as one-hot
            next_states: (batch_size, window_size, n_features)
        """
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        
        states = np.array([t['state'] for t in batch])
        next_states = np.array([t['next_state'] for t in batch])
        
        # Convert actions to one-hot
        # Assuming action_size=3 as per project specs
        actions_idx = [t['action'] for t in batch]
        actions_onehot = np.eye(3)[actions_idx]
        
        return states, actions_onehot, next_states

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.buffer = pickle.load(f)
        return self

    def __len__(self):
        return len(self.buffer)
