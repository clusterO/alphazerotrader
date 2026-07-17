import logging
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models, regularizers, Model
from loss import softmax_cross_entropy_with_logits
import loggers as lg
import config
from settings import run_folder, run_archive_folder

def positional_encoding(length, depth):
    depth = depth // 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

@keras.saving.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate})
        return config

class Gen_Model():
    def __init__(self, reg_const, learning_rate, input_dim, output_dim):
        self.reg_const = reg_const
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim

    def predict(self, x):
        return self.model.predict(x, verbose=0)

    @tf.function(reduce_retracing=True)
    def predict_batch(self, state_batch):
        # Compiled graph path for MCTS evaluation
        return self.model(state_batch, training=False)

    def set_lr(self, lr):
        self.learning_rate = lr
        # Modern Keras 2/3 compatible way to set LR
        if hasattr(self.model.optimizer.learning_rate, 'assign'):
            self.model.optimizer.learning_rate.assign(lr)
        else:
            self.model.optimizer.learning_rate = lr

    def fit(self, states, targets, epochs, verbose, validation_split, batch_size):
        return self.model.fit(states, targets, epochs=epochs, verbose=verbose, validation_split=validation_split, batch_size=batch_size)

    def write(self, game, version):
        self.model.save(run_folder + 'models/version' + "{0:0>4}".format(version) + '.keras')

    def read(self, game, run_number, version):
        path = run_archive_folder + game + '/run' + str(run_number).zfill(4) + "/models/version" + "{0:0>4}".format(version) + '.keras'
        return models.load_model(path, custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits, 'TransformerBlock': TransformerBlock})

    def printWeightAverages(self):
        lg.logger_model.info('Model weight logging skipped.')

class Residual_CNN(Gen_Model):
    def __init__(self, reg_const, learning_rate, input_dim, output_dim, hidden_layers=None):
        super().__init__(reg_const, learning_rate, input_dim, output_dim)
        import yaml
        with open("config.yaml", 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.model = self._build_model()

    def _build_model(self):
        m_cfg = self.cfg['model']
        inputs = layers.Input(shape=self.input_dim)
        
        if m_cfg['type'] == 'transformer':
            x = layers.Dense(m_cfg['d_model'])(inputs)
            x = x + positional_encoding(self.input_dim[0], m_cfg['d_model'])
            for _ in range(m_cfg['num_layers']):
                x = TransformerBlock(m_cfg['d_model'], m_cfg['num_heads'], m_cfg['d_model']*4, m_cfg['dropout'])(x)
            x = layers.GlobalAveragePooling1D()(x)

        elif m_cfg['type'] == 'causal_cnn_attn':
            # Option B: Causal CNN + Attention
            x = layers.Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Conv1D(filters=64, kernel_size=5, padding='causal', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            # Lightweight Global Attention
            attn = layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
            x = layers.LayerNormalization()(x + attn)
            x = layers.GlobalAveragePooling1D()(x)

        else: # Default: original ResNet
            x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
            for _ in range(3):
                shortcut = x
                x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
                x = layers.Conv1D(filters=64, kernel_size=3, padding='same')(x)
                x = layers.Add()([x, shortcut])
                x = layers.Activation('relu')(x)
            x = layers.GlobalAveragePooling1D()(x)

        # Common Heads (restored to d_model // 2 for compatibility)
        head_size = m_cfg['d_model'] // 2
        vh = layers.Dense(head_size, activation='relu')(x)
        vh = layers.Dense(1, activation='tanh', name='value_head')(vh)
        ph = layers.Dense(head_size, activation='relu')(x)
        ph = layers.Dense(self.output_dim, activation='linear', name='policy_head')(ph)
        
        model = Model(inputs=inputs, outputs=[vh, ph])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model.compile(loss={'value_head': 'mse', 'policy_head': softmax_cross_entropy_with_logits},
                      optimizer=optimizer, loss_weights={'value_head': 0.5, 'policy_head': 0.5})
        return model

    def set_policy_head_trainable(self, trainable):
        """
        Freezes or unfreezes the policy head.
        When frozen, only the value head and backbone are trained.
        """
        for layer in self.model.layers:
            if 'policy_head' in layer.name:
                layer.trainable = trainable
        
        # Recompile to apply changes and adjust loss weights
        vw = 1.0 if not trainable else 0.5
        pw = 0.0 if not trainable else 0.5
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        self.model.compile(loss={'value_head': 'mse', 'policy_head': softmax_cross_entropy_with_logits},
                          optimizer=optimizer, loss_weights={'value_head': vw, 'policy_head': pw})
        
        status = "FROZEN" if not trainable else "UNFROZEN"
        print(f"Policy Head is now: {status} (Loss Weights: V={vw}, P={pw})")

    def reset_policy_head(self):
        """
        Surgically resets only the policy head weights to random initialization.
        Overcomes prior inertia while preserving value head calibration.
        """
        print("SURGICAL RESET: Re-initializing Policy Head weights...")
        initializer = tf.keras.initializers.GlorotUniform()
        for layer in self.model.layers:
            if 'policy_head' in layer.name:
                if hasattr(layer, 'kernel_initializer'):
                    weights = layer.get_weights()
                    # Reset kernel and bias if present
                    new_weights = [initializer(w.shape) for w in weights]
                    layer.set_weights(new_weights)
                elif isinstance(layer, layers.Dense):
                    # Manual reset for Dense layers if initializer attribute is not exposed
                    weights = layer.get_weights()
                    new_kernel = initializer(shape=weights[0].shape)
                    new_bias = np.zeros(weights[1].shape)
                    layer.set_weights([new_kernel, new_bias])
        print("Policy Head Reset Complete.")

    def convertToModelInput(self, state):
        return state.binary
