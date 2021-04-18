from data_manager import DataManager

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint


class MyModels:
    def __init__(self, data_manager, regularizers=None, dropout=0.3, 
                 embed_size=200, state_sizes=[128, 256], epochs=20,
                 batch_size=8, optimizer=Adam(0.0001),
                 checkpoint_path='/content/drive/MyDrive/Colab Notebooks/FIT3161/Implementation/checkpoints/', 
                 loss='sparse_categorical_crossentropy', 
                 metrics=['sparse_categorical_accuracy']):
        self.data_manager = data_manager
        self.embed_size = embed_size
        self.state_sizes = state_sizes
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.regularizers = regularizers
        self.dropout = dropout
        self.callbacks = [ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_sparse_categorical_accuracy',
                                         mode='max')]
        # self.callbacks = None

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def fit(self):
        self.model.fit(self.data_manager.train_set.batch(self.batch_size), 
                    epochs=self.epochs, 
                    validation_data=self.data_manager.val_set.batch(
                        self.batch_size), callbacks=self.callbacks)
        
    def evaluate(self, *args, **kwargs):
        self.model.evaluate(*args,**kwargs)

    def load(self, path):
        self.model = load_model(path)