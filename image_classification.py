

from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

def COVID_CTscan_classification(image):
        model = tf.keras.models.load_model("mobilenetv2_final.h5")
        image = image.convert('RGB')
        size = (160,160)    
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image = np.asarray(image)
        image = image.astype(np.float32) 
        image = image[np.newaxis,...]
    
    
        prediction = model.predict(image)
        prediction = tf.nn.sigmoid(prediction)
        prediction = tf.where(prediction < 0.5, 0, 1)
        
        return prediction