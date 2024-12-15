# Import required libraries
import tensorflow as tf
from tensorflow.keras import datasets # type: ignore
import numpy as np

# Load preprocessed data
def load_preprocessed_data():
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    # Normalize pixel values to the range [0, 1]
    X_train, X_test = X_train / 255.0, X_test / 255.0 
    return X_train, y_train, X_test, y_test 