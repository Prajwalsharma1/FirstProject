import sys
import os
import subprocess

import streamlit as st
import numpy as np
from PIL import Image

import tensorflow as tf
import requests

# import time 

import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    # time.sleep(20)
    # Use the raw URL of the model file
    model_url = "https://raw.githubusercontent.com/Prajwalsharma1/FirstProject/master/my_model.keras"
    
    # Download the model
    response = requests.get(model_url)
    
    # Check for successful download
    if response.status_code == 200:
        # Create a temporary file to save the model
        temp_file_path = 'temp_model.keras'
        
        # Write the content to the temporary file
        with open(temp_file_path, 'wb') as f:
            f.write(response.content)
        
        # Load the model from the temporary file
        model = tf.keras.models.load_model(temp_file_path)
        
        # Clean up the temporary file
        os.remove(temp_file_path)
        
        return model
    else:
        raise Exception(f"Failed to download model. Status code: {response.status_code}")
model = load_model()

# st.write("reached")

def get_bar_graph(pred):
    plt.style.use(plt.style.available[6])
    pred_ = [pred[0][0],1-pred[0][0]]
    fig = plt.figure(figsize = (3,2))
    plt.bar(["Dog" , "Cat"],height = pred_,color = ["red","gray"])
    fig.set_label("Probability of Each Class")
    return fig
    
def preprocess_image(image, size=(128, 128)):
    image = image.resize(size)
    image = np.expand_dims(np.array(image),axis=0)
    return image

def predict(image, model):
    pred = model.predict(image,verbose = 0)
    return ["Dog" ,"Cat"][int(pred[0] < 0.5)] , get_bar_graph(pred)

uploaded_file = st.file_uploader("Upload an image", 
                        type=["jpg", "png",'jpeg','tiff',"webp"])

image_area = st.empty()
ans_area = st.empty()
probability_bar = st.empty()
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_area.image(image, 
                     caption="Uploaded Image", width = 400)
    image = preprocess_image(image)
    label,graph = predict(image, model)
    ans_area.write(f"<h2 style = 'text-align:center;'> Prediction: {label} </h2>",unsafe_allow_html=True)   
    probability_bar.pyplot(graph,use_container_width = False)
