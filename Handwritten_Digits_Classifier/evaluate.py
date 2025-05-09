# Import required libraries
import tensorflow as tf
from data.data_loader import load_preprocessed_data

# Function to evaluate model performance based on test data
def evaluate_model(model_path = './models/model.h5'):
    # Load test data
    _, _, X_test, y_test = load_preprocessed_data()

    # Load the saved model
    model = tf.keras.models.load_model(model_path)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose = 2)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    evaluate_model()
