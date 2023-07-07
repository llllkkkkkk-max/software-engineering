import random
import numpy as np
import tensorflow as tf



seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)



class ConcatenateLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ConcatenateLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # Split inputs into a list along the second axis
        block_level_code_output = tf.split(inputs, inputs.shape[1], axis=1)
        # Concatenate the list of tensors along the third axis
        block_level_code_output = tf.concat(block_level_code_output, axis=2)
        # Squeeze the tensor to remove the second axis
        block_level_code_output = tf.squeeze(block_level_code_output, axis=1)
        return block_level_code_output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * input_shape[2])
