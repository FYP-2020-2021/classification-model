import tensorflow as tf

from .layers.bi_rnn_layers import BiRNNLayers
from .model import MyModels
from tensorflow.keras import Model
from tensorflow.keras import layers


class BiLSTM(MyModels):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self):
        x = layers.Input((self.data_manager.max_sentence_number, self.data_manager.maxlen))
        core = BiRNNLayers(self.data_manager, self.embed_size, self.regularizers, self.dropout, self.state_sizes)
        h = layers.TimeDistributed(core)(x)
        h = tf.math.reduce_mean(h, 1)
        h = layers.Softmax()(h)
        self.model = Model(x, h)