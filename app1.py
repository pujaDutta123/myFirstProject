import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import streamlit as st
from PIL import Image
import json
from streamlit_lottie import st_lottie
import tempfile

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
lottie_coding = load_lottiefile("funny_brain_animation.json")

# Set the title in the main content area
st.sidebar.title("WELCOME TO MY WEB APP")

st.title("Brain Tumor Detection")

# Display the Lottie animation in the sidebar

with st.sidebar:
    st_lottie(
        lottie_coding,
        speed=1,
        reverse=False,
        loop=True,
        quality="low",  # medium or high
        height=None,
        width=None,
        key=None,
    )

# Define a function for loading and preprocessing images
def load_and_preprocess_images():
    path = os.listdir('../IIT_intern2/brain_tumor/Training/')
    classes = {'no_tumor': 0, 'pituitary_tumor': 1}

    X = []
    Y = []
    for cls in classes:
        pth = '../IIT_intern2/brain_tumor/Training/' + cls
        for j in os.listdir(pth):
            img = cv2.imread(pth + '/' + j, 0)
            img = cv2.resize(img, (200, 200))
            X.append(img)
            Y.append(classes[cls])

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

# Define a function for training the model
def train_model(X_train, Y_train):
    xtrain = X_train.reshape(len(X_train), -1)
    xtrain = xtrain / 255

    pca = PCA(0.98)
    pca_train = pca.fit_transform(xtrain)

    lg = LogisticRegression(C=0.1)
    lg.fit(pca_train, Y_train)

    sv = SVC()
    sv.fit(pca_train, Y_train)

    return pca, lg, sv

# Define a function for classifying images
def classify_images(pca, model, uploaded_image):
    # Create a temporary file to save the uploaded image
    temp_image = tempfile.NamedTemporaryFile(delete=False)
    temp_image.write(uploaded_image.read())
    temp_image_path = temp_image.name

    # Read the image and perform classification
    img = cv2.imread(temp_image_path, 0)
    img = cv2.resize(img, (200, 200))
    img1 = img.reshape(1, -1) / 255
    img1 = pca.transform(img1)

    prediction = model.predict(img1)

    # Close and delete the temporary file
    temp_image.close()
    os.unlink(temp_image_path)

    return prediction

# Streamlit UI
# Add a background image using custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("file:///D:/IIT_intern2/images/bg_image.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.sidebar.header("Upload an Image for Classification")

# Add a description markdown section
st.markdown("This web app is designed for the detection and classification of brain tumors in medical images. You can upload an MRI image, and the app will predict whether it contains a brain tumor or not.")

# Load and preprocess images
X, Y = load_and_preprocess_images()

# Train the model
pca, lg, sv = train_model(X, Y)

# Sidebar options
st.sidebar.title("Options")

# File uploader widget for image upload
classify_image = st.sidebar.file_uploader("Upload an image for classification", type=["jpg", "png"])

# ...

# Main content area
if classify_image:
    st.image(classify_image, caption="Uploaded Image", use_column_width=True)

    classify_button = st.button("Classify Image")  # Add a button to trigger classification

    if classify_button:
        prediction = classify_images(pca, sv, classify_image)
        st.header("Prediction Result:")
        
        if prediction[0] == 0:
            st.write("Predicted : No Tumor")
        elif prediction[0] == 1:
            st.write("Predicted : Positive Tumor")
        else:
            st.write("Prediction failed.")