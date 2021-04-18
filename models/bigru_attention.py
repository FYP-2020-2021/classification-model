import tensorflow as tf

from .layers.bi_rnn_layers import BiRNNLayers
from .model import MyModels
from tensorflow.keras import Model
from tensorflow.keras import layers


class BiGRUAttention(MyModels):
    def __init__(self, cvl=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_vector_length = cvl

    def build(self):
        x = layers.Input((self.data_manager.max_sentence_number, self.data_manager.maxlen))
        core = BiRNNLayers(self.data_manager, self.embed_size, self.regularizers, self.dropout, self.state_sizes, cell_type='gru', context_vector_length=self.context_vector_length)
        h = layers.TimeDistributed(core)(x)
        h = tf.math.reduce_mean(h, 1)
        h = layers.Dense(self.data_manager.num_classes, 'relu', 
                         kernel_regularizer=self.regularizers)(h)
        h = layers.Softmax()(h)
        self.model = Model(x, h)