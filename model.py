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
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

class Gen_Model():
    def __init__(self, reg_const, learning_rate, input_dim, output_dim):
        self.reg_const = reg_const
        self.learning_rate = learning_rate
        self.input_dim = input_dim # (window_size, features)
        self.output_dim = output_dim

    def predict(self, x):
        return self.model.predict(x, verbose=0)

    def fit(self, states, targets, epochs, verbose, validation_split, batch_size):
        return self.model.fit(states, targets, epochs=epochs, verbose=verbose, validation_split=validation_split, batch_size=batch_size)

    def write(self, game, version):
        self.model.save(run_folder + 'models/version' + "{0:0>4}".format(version) + '.keras')

    def read(self, game, run_number, version):
        # Keras 3 prefers .keras extension
        path = run_archive_folder + game + '/run' + str(run_number).zfill(4) + "/models/version" + "{0:0>4}".format(version) + '.keras'
        return models.load_model(path, custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})

    def printWeightAverages(self):
        lg.logger_model.info('Transformer Model weight logging skipped for brevity.')

class Residual_CNN(Gen_Model): # Keeping name for compatibility with main.py
    def __init__(self, reg_const, learning_rate, input_dim, output_dim, hidden_layers=None):
        super().__init__(reg_const, learning_rate, input_dim, output_dim)
        # We use config.yaml values instead of hidden_layers list
        import yaml
        with open("config.yaml", 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        self.model = self._build_model()

    def _build_model(self):
        m_cfg = self.cfg['model']
        inputs = layers.Input(shape=self.input_dim)
        
        # 1. Projection to d_model
        x = layers.Dense(m_cfg['d_model'])(inputs)
        
        # 2. Add Positional Encoding
        pos_enc = positional_encoding(self.input_dim[0], m_cfg['d_model'])
        x = x + pos_enc
        
        # 3. Transformer Layers
        for _ in range(m_cfg['num_layers']):
            x = TransformerBlock(m_cfg['d_model'], m_cfg['num_heads'], m_cfg['d_model']*4, m_cfg['dropout'])(x)
        
        # 4. Extract representation (Global Average Pooling across time)
        x = layers.GlobalAveragePooling1D()(x)
        
        # 5. Value Head
        vh = layers.Dense(m_cfg['d_model'] // 2, activation='relu')(x)
        vh = layers.Dense(1, activation='tanh', name='value_head')(vh)
        
        # 6. Policy Head
        ph = layers.Dense(m_cfg['d_model'] // 2, activation='relu')(x)
        ph = layers.Dense(self.output_dim, activation='linear', name='policy_head')(ph)
        
        model = Model(inputs=inputs, outputs=[vh, ph])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            loss={'value_head': 'mse', 'policy_head': softmax_cross_entropy_with_logits},
            optimizer=optimizer,
            loss_weights={'value_head': 0.5, 'policy_head': 0.5}
        )
        return model

    def convertToModelInput(self, state):
        return state.binary
