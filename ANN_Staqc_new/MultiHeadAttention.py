from jax.nn import initializers
from keras import constraints, regularizers, activations
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
import tensorflow as tf
import os
import numpy as np
import random



seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)



class ScaledDotProductAttention(Layer):
    def __init__(self, return_attention=False, history_only=False, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_attention = return_attention
        self.history_only = history_only
        self.intensity = None
        self.attention = None

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:-1] + (input_shape[-1][-1],)
        if self.return_attention:
            attention_shape = input_shape[0][:2] + (input_shape[1][1],)
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if self.return_attention:
            return [mask, None]
        return mask

    def call(self, inputs, mask=None, **kwargs):
        query, key, value = inputs
        feature_dim = K.shape(query)[-1]
        e = tf.matmul(query, key, transpose_b=True) / tf.sqrt(tf.cast(feature_dim, dtype=tf.float32))

        if self.history_only:
            query_len, key_len = K.shape(query)[1], K.shape(key)[1]
            indices = tf.expand_dims(tf.range(0, key_len), axis=0)
            upper = tf.expand_dims(tf.range(0, query_len), axis=-1)
            e -= 10000.0 * tf.expand_dims(tf.cast(indices > upper, dtype=tf.float32), axis=0)

        if mask is not None:
            e -= 10000.0 * (1.0 - tf.cast(tf.expand_dims(mask, axis=-2), dtype=tf.float32))

        self.intensity = e
        e = tf.exp(e - tf.reduce_max(e, axis=-1, keepdims=True))
        self.attention = e / tf.reduce_sum(e, axis=-1, keepdims=True)
        v = tf.matmul(self.attention, value)

        if self.return_attention:
            return [v, self.attention]
        return v



class MultiHeadAttention(Layer):
    """Multi-head attention layer.
    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 head_num,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 history_only=False,
                 **kwargs):
        """Initialize the layer.
        :param head_num: Number of heads.
        :param activation: Activations for linear mappings.
        :param use_bias: Whether to use bias term.
        :param kernel_initializer: Initializer for linear mappings.
        :param bias_initializer: Initializer for linear mappings.
        :param kernel_regularizer: Regularizer for linear mappings.
        :param bias_regularizer: Regularizer for linear mappings.
        :param kernel_constraint: Constraints for linear mappings.
        :param bias_constraint: Constraints for linear mappings.
        :param history_only: Whether to only use history in attention layer.
        """
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.head_num = head_num
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.history_only = history_only

        self.Wq = None
        self.bq = None
        self.Wk = None
        self.bk = None
        self.Wv = None
        self.bv = None
        self.Wo = None
        self.bo = None

    def build(self, input_shape):
        feature_dim = int(input_shape[0][-1])
        if feature_dim % self.head_num != 0:
            raise IndexError('Invalid head number %d with the given input dim %d' % (self.head_num, feature_dim))

        self.Wq = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='Wq',
        )
        if self.use_bias:
            self.bq = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bq',
            )
        self.Wk = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='Wk',
        )
        if self.use_bias:
            self.bk = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bk',
            )
        self.Wv = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='Wv',
        )
        if self.use_bias:
            self.bv = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bv',
            )
        self.Wo = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='Wo',
        )
        if self.use_bias:
            self.bo = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bo',
            )
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        q, k, v = inputs
        q = K.dot(q, self.Wq)
        k = K.dot(k, self.Wk)
        v = K.dot(v, self.Wv)
        if self.use_bias:
            q += self.bq
            k += self.bk
            v += self.bv
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        scaled_dot_product_attention = ScaledDotProductAttention(history_only=self.history_only)
        y = scaled_dot_product_attention(inputs=[q, k, v], mask=mask)
        y = K.dot(y, self.Wo)
        if self.use_bias:
            y += self.bo
        if self.activation is not None:
            y = self.activation(y)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = {
            'head_num': self.head_num,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'history_only': self.history_only,
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

