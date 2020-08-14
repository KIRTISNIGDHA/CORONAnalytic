import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np


def COVID_Xray_classification(image):
        model = tf.keras.models.load_model("X-ray_model.h5")
        image = image.convert('RGB')
        size = (128,128)    
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image = np.asarray(image)
        image = image.astype(np.float32)
        image = image[np.newaxis,...]
        prediction = model.predict(image)
        prediction = round(prediction[0][0],0)      
        return prediction