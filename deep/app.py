# app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# Load the trained model (use your own path or train one first)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_recognition_model.h5")

model = load_model()

# App title
st.title("🧠 Digit Recognizer (MNIST) with Streamlit")

# Upload image from user
uploaded_file = st.file_uploader("Upload an image of a digit (28x28 or larger)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image and convert to grayscale
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = np.array(image)
    image = cv2.resize(image, (28, 28))         # Resize to MNIST format
    image = 255 - image                         # Invert colors (white on black)
    image = image / 255.0                       # Normalize
    image = image.reshape(1, 28, 28, 1)         # Reshape for model

    # Predict
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)

    st.success(f"✅ Predicted Digit: **{predicted_digit}**")

