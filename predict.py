import cv2
import numpy as np
import pickle
from tensorflow import keras
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Loading LabelBinarizer (lb) from the saved file
with open('C:\\Users\\hp\\Desktop\\Sign_Language_MNIST\\saved_objects\\label_binarizer.pkl', 'rb') as file:
    loaded_lb = pickle.load(file)
def get_label(y):
    return loaded_lb.inverse_transform(y)[0][0]

def predict_image(img):
    prediction=[0,0,0]
    model_path = 'C:\\Users\\hp\\Desktop\\Sign_Language_MNIST\\model.h5'
    loaded_model = keras.models.load_model(model_path)
    if img is not None:
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        img=img.reshape((1, 64, 64, 3))
        prediction = loaded_model.predict(img)
        index=np.argmax(prediction)
        y=np.zeros((1,24))
        y[0][index]=1
        lettre=get_label(y)
        result = {'prediction': lettre}
        return result
    else:
        return "img is None"
