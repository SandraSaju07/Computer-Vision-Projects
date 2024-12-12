# Import required libraries
import tensorflow as tf
from tensorflow.keras import datasets # type: ignore
import numpy as np

# Function to load and preprocess data
def load_preprocess_data():
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
    # Normalize pixel values to the range [0, 1]
    X_train, X_test = X_train / 255.0, X_test / 255.0
    # Convert from 2D to 1D
    y_train, y_test = y_train.reshape(-1,), y_test.reshape(-1,)
    return X_train, y_train, X_test, y_test

# Function to get class labels/names
def get_class_names():
    return ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]