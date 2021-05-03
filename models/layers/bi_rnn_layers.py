import tensorflow as tf

from tensorflow.keras import layers
from .attention_layer import AttentionLayer


class BiRNNLayers(layers.Layer):
    def __init__(self, dm, es, r, d, ss, cell_type='lstm', context_vector_length=100):
        super(BiRNNLayers, self).__init__()
        # self.data_manager = dm
        self.maxlen = dm.maxlen
        self.vocab_size = dm.vocab_size
        self.num_classes = dm.num_classes
        self.embed_size = es
        self.regularizers = r
        self.dropout = d
        self.state_sizes = ss
        self.cell_type = cell_type
        self.context_vector_length = context_vector_length

    def build_lstm(self):
        self.embedding = layers.Embedding(self.vocab_size + 1, self.embed_size, 
                             embeddings_regularizer=self.regularizers, 
                             input_length=self.maxlen, mask_zero=True)
        num_layers = len(self.state_sizes)
        self.biRNN = []
        for i in range(num_layers):
            self.biRNN.append(layers.Bidirectional(layers.LSTM(self.state_sizes[i], 
                                    kernel_regularizer=self.regularizers, 
                                    recurrent_regularizer=self.regularizers, 
                                    dropout=self.dropout, 
                                    return_sequences=True)))
        self.max_pool = layers.MaxPool1D(self.state_sizes[-1] * 2, padding='same', data_format='channels_first')
        self.avg_pool = layers.AvgPool1D(self.state_sizes[-1] * 2, padding='same', data_format='channels_first')
        self.fc_layer = layers.Dense(self.num_classes, 'relu', 
                         kernel_regularizer=self.regularizers)

    def build_gru(self):
        self.embedding = layers.Embedding(self.vocab_size + 1, self.embed_size, 
                             embeddings_regularizer=self.regularizers, 
                             input_length=self.maxlen, mask_zero=True)
        num_layers = len(self.state_sizes)
        self.biRNN = []
        for i in range(num_layers):
            self.biRNN.append(layers.Bidirectional(layers.GRU(self.state_sizes[i], 
                                    kernel_regularizer=self.regularizers, 
                                    recurrent_regularizer=self.regularizers, 
                                    dropout=self.dropout, 
                                    return_sequences=True)))
        self.fc_layer = layers.Dense(self.num_classes, 'relu', 
                                    kernel_regularizer=self.regularizers)
        self.attention = AttentionLayer(self.context_vector_length)
    
    def build(self, input_shape):
        if self.cell_type == 'lstm':
            self.build_lstm()
        else:
            self.build_gru()
    
    def call(self, x):
        h = self.embedding(x)
        num_layers = len(self.state_sizes)
        for i in range(num_layers):
            h = self.biRNN[i](h)
        if self.cell_type == 'lstm':
            max = self.max_pool(h)
            max = layers.Flatten()(max)
            avg = self.avg_pool(h)
            avg = layers.Flatten()(avg)
            h = tf.concat([max, avg], 1)
            h = tf.reshape(h, [-1, self.maxlen * 2])
            h = self.fc_layer(h)
        elif self.cell_type == 'gru':
            h = self.attention(h)
            h = self.fc_layer(h)
        return h

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_classes)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            # 'data_manager': self.data_manager,
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'num_classes': self.num_classes,
            'embed_size': self.embed_size,
            'regularizers': self.regularizers,
            'dropout': self.dropout,
            'state_sizes': self.state_sizes,
            'cell_type': self.cell_type,
            'context_vector_length': self.context_vector_length
        })
        return config
