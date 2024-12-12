# Import required libraries
import streamlit as st
import numpy as np
import tensorflow as tf 
from PIL import Image

# Load the trained model
MODEL_PATH = './models/cnn_model.h5'

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# CIFAR-10 class labels
CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Function to resize and normalize image
def preprocess_image(image):
    image = image.resize((32, 32))  # CIFAR-10 images are 32 x 32
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis = 0)  # Add batch dimension
    return image_array

# Function to predict the class of the uploaded image
def predict(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# Streamlit UI
def main():
    st.title("Image Classification using CNN")
    st.markdown("""
            This app classifies images into one of the following categories: 
        **Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck**
""")
    
    st.sidebar.title("Options")
    app_mode = st.sidebar.selectbox("Choose the mode", ["Home", "About"])

    if app_mode == "Home":
        st.header("Upload an Image")
        uploaded_file = st.file_uploader("Choose an image file", type = ["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width = True)

            # Predict button
            if st.button("Classify"):
                with st.spinner("Classifying..."):
                    predicted_class, confidence = predict(image)
                st.success(f"Prediction: **{predicted_class}**")
                st.info(f"Confidence: **{confidence:.2f}%**")

    elif app_mode == "About":
        st.header("About")
        st.markdown("""
            **Project:** Image Classification using CNN  
            **Techniques:** CNN, Tensorflow, Keras  
            **Dataset:** CIFAR-10  
            **Description:** This model classifies images into 10-predefined categories using a CNN trained on CIFAR-10 dataset.
""")
        
if __name__ == "__main__":
    main()
