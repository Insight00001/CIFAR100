import streamlit as st 
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array,load_img
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
from PIL import Image
import io
st.set_page_config(
    page_title='CIFAR100',
    layout='wide',
    page_icon='bar_chart'
)
st.header('CIFAR100 with RESNET50')


# Load CIFAR-100 dataset

file = 'meta'
model = load_model("cifar100_model.h5")
with open(file,'rb') as f:
    data  = pk.load(f,encoding='latin1')
    class_names = data['fine_label_names']
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
@st.cache_resource
def get_true_class_label(image_index, dataset='test'):
    if dataset == 'test':
        true_class_index = y_test[image_index][0]  # Get the class index for the image
    elif dataset == 'train':
        true_class_index = y_train[image_index][0]  # Get the class index for the image
    true_class_name = class_names[true_class_index]  # Map index to class name
    return true_class_name


@st.cache_resource
def load_and_preprocess_img(img_path,target_size=(32,32)):
    # load the image
    image= Image.open(img_path)
    # resize the image
    image = image.resize(target_size)

    # convert the image to numpy
    img_array = np.array(image)
    # add batch normalization
    img_array = np.expand_dims(img_array,axis=0)
    # scale the image to [0,1]
    img_array = img_array/255.0
    
    return img_array
@st.cache_resource
def predict_image_class(image_path):
    preprocess_image = load_and_preprocess_img(image_path)
    predictions = model.predict(preprocess_image)
    predicted_class_index = np.argmax(predictions)
    class_name = class_names[predicted_class_index]
    confidence_score=np.max(predictions)

  
    return class_name,confidence_score


col1,col2,col3= st.columns(3)
with col1:
    st.write('Resnet50 Evaluation ')
    st.image('resnet_50_regularization.png',caption='Resnet50 Performance curve')
    
with col2:
    st.write('Resnet50 Classification and Misclassifiication')
    st.image('cfm_resnet.png', caption='Resnet50 classification and misclassification')
with col3:
    upload_file=st.file_uploader('choose an image file',type=['jpeg','jpg','png'])
    st.divider()
    if upload_file is not None:
        image = Image.open(upload_file)
        
        with col3:
            st.image(image,caption='Uploaded Image')
            if st.button('Classify Image'):
                class_name,confidence_score=predict_image_class(upload_file)
                
                st.write('Classifying.........')
                st.write(f'Predicted Class {class_name}, and the confidenc score is: {confidence_score*100:.2f}%')
              


