# Import required libraries
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
MODEL_PATH = './models/model.h5'

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Function to resize and normalize image
def preprocess_image(image):
    image = image.convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis = 0)  # Add batch dimension
    return image_array

# Function to predict the class of the uploaded image
def predict(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# Streamlit UI
def main():
    st.title("Handwritten Digit Classification")
    st.markdown("This app classifies handwritten digit images into digits from zero to nine.")
    
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image file", type = ["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption = "Uploaded Image", use_container_width = True)

        if st.button("Classify"):
            with st.spinner("Classifying..."):
                predicted_class, confidence = predict(image)

            st.success(f"Prediction: **{predicted_class}**")
            st.info(f"Confidence: **{confidence:.2f}%**")

if __name__ == "__main__":
    main()