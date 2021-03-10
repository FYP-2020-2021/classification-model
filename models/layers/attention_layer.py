"""
Source: https://github.com/FlorisHoogenboom/keras-han-for-docla/blob/master/keras_han/layers.py
"""

import tensorflow.keras as keras
import tensorflow as tf


class AttentionLayer(keras.layers.Layer):
    def __init__(self, context_vector_length=100, **kwargs):
        """
        An implementation of a attention layer. This layer
        accepts a 3d Tensor (batch_size, time_steps, input_dim) and
        applies a single layer attention mechanism in the time
        direction (the second axis).
        :param context_vector_lenght: (int) The size of the hidden context vector.
            If set to 1 this layer reduces to a standard attention layer.
        :param kwargs: Any argument that the baseclass Layer accepts.
        """
        self.context_vector_length = context_vector_length
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[2]

        # Add a weights layer for the
        self.W = self.add_weight(
            name='W', shape=(dim, self.context_vector_length),
            initializer=keras.initializers.get('uniform'),
            trainable=True
        )

        self.u = self.add_weight(
            name='context_vector', shape=(self.context_vector_length, 1),
            initializer=keras.initializers.get('uniform'),
            trainable=True
        )

        super(AttentionLayer, self).build(input_shape)

    def _get_attention_weights(self, X):
        """
        Computes the attention weights for each timestep in X
        :param X: 3d-tensor (batch_size, time_steps, input_dim)
        :return: 2d-tensor (batch_size, time_steps) of attention weights
        """
        # Compute a time-wise stimulus, i.e. a stimulus for each
        # time step. For this first compute a hidden layer of
        # dimension self.context_vector_length and take the
        # similarity of this layer with self.u as the stimulus
        shape = X.shape
        X = tf.reshape(X, [-1, shape[-1]])
        h = tf.matmul(X, self.W)
        u_tw = keras.activations.tanh(tf.reshape(h, [-1, shape[1], self.context_vector_length]))
        
        shape = u_tw.shape
        u_tw = tf.reshape(u_tw, [-1, shape[-1]])
        h = tf.matmul(u_tw, self.u)
        tw_stimulus = tf.reshape(h, [-1, shape[1], 1])

        # Remove the last axis an apply softmax to the stimulus to
        # get a probability.
        tw_stimulus = tf.reshape(tw_stimulus, [-1, tw_stimulus.shape[1]])
        att_weights = keras.layers.Softmax()(tw_stimulus)

        return att_weights

    def call(self, X):
        att_weights = self._get_attention_weights(X)

        # Reshape the attention weights to match the dimensions of X
        att_weights = tf.reshape(att_weights, [-1, att_weights.shape[1], 1])

        # Multiply each input by its attention weights
        weighted_input = X * att_weights

        # Sum in the direction of the time-axis.
        return tf.math.reduce_sum(weighted_input, 1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def get_config(self):
        config = {
            'context_vector_length': self.context_vector_length
        }
        base_config = super(AttentionLayer, self).get_config()
        return {**base_config, **config}