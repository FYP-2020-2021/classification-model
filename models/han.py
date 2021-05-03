"""
Source: https://github.com/FlorisHoogenboom/keras-han-for-docla/blob/master/keras_han/model.py
"""

import tensorflow as tf

from .layers.attention_layer import AttentionLayer
from .model import MyModels
from tensorflow.keras import Model
from tensorflow.keras import layers

class HAN(MyModels):
    def __init__(self, cvl=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_vector_length = cvl

    def _build_word_encoder(self):
        x = layers.Input(self.data_manager.maxlen)
        h = layers.Embedding(self.data_manager.vocab_size + 1, self.embed_size, 
                             embeddings_regularizer=self.regularizers, 
                             input_length=self.data_manager.maxlen, mask_zero=True)(x)
        for i in range(len(self.state_sizes)):
            h = layers.Bidirectional(layers.LSTM(self.state_sizes[i], 
                                        kernel_regularizer=self.regularizers, 
                                        recurrent_regularizer=self.regularizers, 
                                        dropout=self.dropout, 
                                        return_sequences=True))(h)
        h = AttentionLayer(self.context_vector_length)(h)
        return Model(inputs=x, outputs=h, name='word_encoder')

    def _build_sentence_encoder(self):
        x = layers.Input((self.data_manager.max_sentence_number, self.state_sizes[-1] * 2))
        # h = layers.Embedding(self.data_manager.vocab_size + 1, self.embed_size, 
        #                      embeddings_regularizer=self.regularizers, 
        #                      input_length=self.data_manager.maxlen, mask_zero=True)(x)
        h = layers.Bidirectional(layers.LSTM(self.state_sizes[0], 
                                        kernel_regularizer=self.regularizers, 
                                        recurrent_regularizer=self.regularizers, 
                                        dropout=self.dropout, 
                                        return_sequences=True))(x)
        for i in range(1, len(self.state_sizes)):
            h = layers.Bidirectional(layers.LSTM(self.state_sizes[i], 
                                        kernel_regularizer=self.regularizers, 
                                        recurrent_regularizer=self.regularizers, 
                                        dropout=self.dropout, 
                                        return_sequences=True))(h)
        h = AttentionLayer(self.context_vector_length)(h)
        return Model(inputs=x, outputs=h, name='sentence_encoder')
    
    def build(self):
        x = layers.Input((self.data_manager.max_sentence_number, self.data_manager.maxlen))

        word_encoder = self._build_word_encoder()
        h = layers.TimeDistributed(word_encoder)(x)

        sentence_encoder = self._build_sentence_encoder()
        h = sentence_encoder(h)
        
        h = layers.Dense(self.data_manager.num_classes)(h)
        h = layers.Softmax()(h)
        self.model = Model(x, h)