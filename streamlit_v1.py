import pickle
import streamlit as st
import pandas as pd

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

# Imports & Setup:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import PIL
import PIL.Image
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import utils
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)




page = st.selectbox("Choose your page", ["Overview", "FAQ"])

### Excluding Imports ###
st.title("Bird Shazam")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    def load_image_new(path):
        images = Image.open(path)
        new_image=images.resize((128,128))
        color_image=new_image.convert("RGB")
        arrays1 = np.asarray(color_image)
        floaters= arrays1.astype('float32')
        floaters2=floaters/255.0
        floaters3=np.asarray(floaters2)
        floaters4 = floaters3.reshape(1, 128, 128, 3)
        return floaters4
    order_names=["ANSERIFORMES","CAPRIMULGIFORMES","CHARADRIIFORMES","CORACIIFORMES","CUCULIFORMES","GAVIIFORMES","PASSERIFORMES","PELECANIFORMES","PICIFORMES","PODICIPEDIFORMES","PROCELLARIIFORMES","SULIFORMES"]

    def predict_birds_order(file):
        best_model_1 = tf.keras.models.load_model('saved_models/order_es.h5')
        preds1=best_model_1.predict(file)
        preds2=np.argmax(preds1,axis=1)
        return order_names[int(preds2)-1]


    image_new=load_image_new(uploaded_file)

    st.image(image_new, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying birb...")
    label = predict_birds_order(image_new)


    st.write(f'This bird belongs to the {label} order!')
