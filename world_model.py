import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class TransitionModel:
    def __init__(self, input_shape, action_size, learning_rate=0.0001):
        """
        Component 1: Transition Model for World Model Phase 1.
        input_shape: (window_size, n_features) - e.g. (60, 19)
        action_size: 3 (one-hot)
        """
        self.input_shape = input_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        # 1. State Input
        state_input = layers.Input(shape=self.input_shape, name='state_input')

        # 2. Temporal Encoder (Causal Convolutions + Stacked LSTMs)
        # v3: Balanced Depth (2x128 LSTMs)
        x = layers.Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu')(state_input)
        x = layers.LayerNormalization()(x)
        x = layers.Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu')(x)
        x = layers.LayerNormalization()(x)

        # Stacked LSTMs for temporal context
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.LayerNormalization()(x)
        state_repr = layers.LSTM(128, return_sequences=False)(x)

        # 3. Action Input (One-hot vector)
        action_input = layers.Input(shape=(self.action_size,), name='action_input')

        # 4. Fusion
        combined = layers.Concatenate()([state_repr, action_input])

        # 5. Prediction Layers
        x = layers.Dense(1024, activation='relu')(combined)
        x = layers.Dense(1024, activation='relu')(x)

        total_elements = self.input_shape[0] * self.input_shape[1]
        output_flat = layers.Dense(total_elements, activation='linear')(x)
        output = layers.Reshape(self.input_shape)(output_flat)

        model = models.Model(inputs=[state_input, action_input], outputs=output, name='TransitionModel')

        # Standard MSE Loss (v3)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')

        return model

    def predict(self, state, action_idx):
        """
        Deterministic prediction of the next state.
        """
        state_batch = np.expand_dims(state, axis=0)
        action_onehot = np.zeros((1, self.action_size))
        action_onehot[0, action_idx] = 1.0
        
        return self.model.predict([state_batch, action_onehot], verbose=0)[0]

    def predict_batch(self, states, action_indices):
        """
        Batched prediction for efficiency (Long-term fix for MCTS bottleneck).
        states: Array or list of state tensors (N, window, features)
        action_indices: Array or list of action indices (N,)
        """
        states_batch = np.array(states, dtype=np.float32)
        actions_onehot = np.zeros((len(action_indices), self.action_size), dtype=np.float32)
        for i, idx in enumerate(action_indices):
            actions_onehot[i, idx] = 1.0
            
        # verbose=0 is important for speed
        return self.model.predict([states_batch, actions_onehot], verbose=0, batch_size=len(states))

    def save(self, path):
        self.model.save(path)

    def set_trainable(self, trainable):
        """
        Enables or disables training for the model.
        Used to unfreeze the model during the retraining cycle.
        """
        self.model.trainable = trainable
        for layer in self.model.layers:
            layer.trainable = trainable
            
        if trainable:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            self.model.compile(optimizer=optimizer, loss='mse')
        
        status = "UNFROZEN" if trainable else "FROZEN"
        print(f"World Model status set to: {status}")

    def fit(self, states, actions_onehot, next_states, epochs=1, batch_size=32):
        """
        Train the model on a batch of transitions.
        """
        return self.model.fit(
            [states, actions_onehot],
            next_states,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

    @staticmethod
    def load(path):
        # Load the Keras model
        m = tf.keras.models.load_model(path)
        
        # Get metadata from model shape
        input_shape = m.input_shape[0][1:] # [(None, 60, 19), (None, 3)] -> (60, 19)
        action_size = m.input_shape[1][1] # 3
        
        # Create instance
        instance = TransitionModel(input_shape, action_size)
        instance.model = m
        
        # EXPLICIT FREEZE (Phase 1 Gate Requirement)
        instance.model.trainable = False
        for layer in instance.model.layers:
            layer.trainable = False
            
        print(f"World Model Loaded and FROZEN. Trainable weights: {len(instance.model.trainable_weights)}")
        return instance
