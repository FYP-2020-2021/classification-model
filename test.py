import tensorflow as tf
import pickle
import numpy as np

from data_manager import DataManager

model = tf.keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/FIT3161/Implementation/HAN1/epoch_03-val_acc_0.87")

dm = DataManager("/content/classification-model/ArbitrarySet", encoding="latin")
dm.labels = np.asarray(["Elementary School", "Middle School", "High School", "Undergraduate"])


file = open("/content/classification-model/Final Dataset/1 Elementary School/1.txt", encoding="latin")
file2 = open("/content/classification-model/Final Dataset/4 Undergraduate/Amsterdam-adv.txt", encoding="latin")
file3 = open("/content/classification-model/Final Dataset/3 High School/1006.txt", encoding="latin")
file4 = open("/content/classification-model/Final Dataset/4 Undergraduate/university_50.txt", encoding="latin")
text = [file.read(), file2.read(), file3.read(), file4.read()]
print("Number of input texts:", len(text))
text = dm.predict_preprocess(text, "/content/classification-model/tokenizer.pkl")
print("Shape of input tensor:", text.shape)
result = model.predict(text)
print("Confidence levels:")
print(result)
amax = np.argmax(result, 1)
print("Result:", dm.labels[amax])