# Import required libraries
from data.data_loader import load_preprocessed_data
import tensorflow as tf
import numpy as np
from utils.helper_functions import plot_heatmap

# Function to predict model
def predict(model_path = './models/model.h5'):
    # Load preprocessed test data
    _, _, X_test, y_test = load_preprocessed_data()

    # Load trained model
    model = tf.keras.models.load_model(model_path)

    # Predict the model
    predictions = model.predict(X_test)

    # Get class labels
    predicted_classes = [np.argmax(ele) for ele in predictions]
    
    # Confusion Matrix
    matrix = tf.math.confusion_matrix(labels = y_test, predictions = predicted_classes)
    plot_heatmap(matrix)
    print(f"Confusion Matrix: {matrix}")

if __name__ == "__main__":
    predict()
    