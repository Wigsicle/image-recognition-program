# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:52:05 2021

@author: grave
"""


import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

input_height = 150
input_width = 150

def import_and_predict(image_data, model):
    
        size = (input_height,input_width)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction

model = tf.keras.models.load_model('<change directory>/models/cnn_model_3-conv-32-nodes-1-dense-0.2-dropout.hdf5') #loading a trained model

st.write("""
         # CNN ikura sushi doughnut unknown classification
         """
         )

st.write("This is a simple image classification web app to predict rock-paper-scissor hand sign")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("Donut")
    elif np.argmax(prediction) == 1:
        st.write("Ikura")
    else:
        st.write("Unknown")
    
    st.text("Probability (0: Dount, 1: Not, 2: Unknown)")
    st.write(prediction)
