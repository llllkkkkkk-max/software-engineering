from jinja2.filters import K
from tensorflow.keras.layers import Layer
import tensorflow as tf
import os
import numpy as np
import random



seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)



class PositionEmbedding(Layer):
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size
        self.mode = mode
        super(PositionEmbedding, self).__init__(**kwargs)

    def call(self, x):
        if self.size is None or self.mode == 'sum':
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000., 2 * K.arange(self.size // 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.sin(position_ij[:, :, 0::2])  # Sinusoidal encoding
        position_ij_1 = K.cos(position_ij[:, :, 1::2])  # Cosine encoding
        position_ij = K.concatenate([position_ij, position_ij_1], axis=-1)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], axis=-1)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)

