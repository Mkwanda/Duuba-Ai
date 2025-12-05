import streamlit as st
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from keras.models import load_model
from PIL import Image
import cv2 
import numpy as np
from matplotlib import pyplot as plt


loaded_model = tf.saved_model.load('my_ssd_mobnetpod\export\saved_model')


labels = {1: 'Healthy Pod', 2: 'Infected Pod'}

fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']


def detect_fn(image):
    predictions = loaded_model(image)

    return(predictions)

    

def run():

    st.title("Cocoa 🍍 Classification")
    #print(list(loaded_model.signatures.keys()))
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        #img_raw = Image.open(img_file).resize((250, 250))
        #st.image(img_raw, use_column_width=False)
        #save_image_path = './upload_images/' + img_file.name
        #with open(save_image_path, "wb") as f:
         #   f.write(img_file.getbuffer())

        infer = loaded_model.signatures["serving_default"]
        print(infer.structured_outputs)

        #img = cv2.imread(img_file)
        image_np = np.array(img_file)
        image_np = image_np.shape
        image_np1 = st.image((np.expand_dims(image_np, 0)), use_column_width=False)
        input_tensor = tf.convert_to_tensor(image_np1, dtype=tf.float32)

        detections = detect_fn(input_tensor)

        print(detections)


 
        

run()