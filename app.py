import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

# Load the fixed model
model = load_model(r"H:\EdgeMatrix\TeachableMachineprojects\img classification\fixed_keras_model.h5", compile=False)

# Load the labels
class_names = open(r"H:\EdgeMatrix\TeachableMachineprojects\img classification\labels.txt", "r").readlines()

# Function to preprocess the image
def preprocess_image(image):
    size = (224, 224)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized_image_array, axis=0)
    return data

# Streamlit Web App UI
st.title("Image Classification with Your Model")

# File uploader widget
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert to numpy array and preprocess the image
    image_array = np.array(image)
    data = preprocess_image(image_array)
    
    # Predict the class
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    
    # Display the prediction result
    st.write(f"Prediction: **{class_name}**")
    st.write(f"Confidence Score: **{confidence_score:.2f}**")
