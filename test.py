import tensorflow as tf
import pickle
import numpy as np

model = tf.keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/FIT3161/Implementation/HAN1/epoch_03-val_acc_0.87")

pickle_file = open("/content/drive/MyDrive/Colab Notebooks/FIT3161/Implementation/datamanager.pkl", 'rb')
dm = pickle.load(pickle_file)
pickle_file.close()

file = open("/content/classification-model/Final Dataset/1 Elementary School/1.txt")
file2 = open("/content/classification-model/Final Dataset/4 Undergraduate/Amsterdam-adv.txt")
file3 = open("/content/classification-model/Final Dataset/3 High School/1006.txt")
text = [file.read(), file2.read(), file3.read()]
print("Number of input texts:", len(text))
text = dm.predict_preprocess(text)
result = model.predict(text)
print("Confidence levels:")
print(result)
amax = np.argmax(result, 1)
print("Result:", dm.labels[amax])