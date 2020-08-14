import streamlit as st
import numpy as np
from PIL import Image
# image classification model
from image_classification import COVID_CTscan_classification 
from X_ray_image_classification import COVID_Xray_classification


st.title("CORONAnalytics ")
st.markdown("Welcome to CORONAnalytics- A machine learning based COVID 19 detection tool from Radiology scans. Please input the following values to get a prediction: ")

st.set_option('deprecation.showfileUploaderEncoding', False)

# for background color
st.markdown(""" 
<style> 
body {
color:#000000;
 background-color: #FEF9E7;
 }
 </style>
 """, unsafe_allow_html= True)
 
if st.checkbox('Patient Information'):
    
    Age = st.number_input('Age at diagnosis (in years)',min_value=0,max_value=120,value =0,step=1)
    Sex = st.multiselect(
      'Gender of the patient', 
       ('Male', 'Female'))
if st.checkbox('Radiology images'):
    if st.checkbox('Lung CT scan images'):
          Input_image= st.file_uploader("Upload the lung CT scan image as png file", type="png")
    
          if Input_image is not None:
               image = Image.open(Input_image)
               st.image(image, caption='Uploaded Image.', use_column_width=True)
               st.write("")
            #st.write(" Detecting...")
               label = COVID_CTscan_classification(image)
     
               if label == 0:
                   st.write("The lung CT scan shows  patient has been affected by COVID-19")
               else:
                   st.write("The lung CT scan shows patient is healthy") 
        
    if st.checkbox('Chest X-ray images'):  
        Input_image= st.file_uploader("Upload the chest X-ray scan image", type=["png", "jpg", "jpeg"])
    
        if Input_image is not None:
            image = Image.open(Input_image)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            #st.write(" Detecting...")
            label = COVID_Xray_classification(image)
     
            if label == 0:
                st.write("The chest X-ray scan shows  patient has been affected by COVID-19")
            else:
                st.write("The chest X-ray scan shows patient is healthy")
   


st.info('This tool is for the doctors to help the patient make an informed decision regarding treatment strategy and is a purely informational message. The predictions are made using machine learning algorithm.')