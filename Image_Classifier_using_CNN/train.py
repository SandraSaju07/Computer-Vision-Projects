# Import required libraries
import tensorflow as tf
from data.data_loader import load_preprocess_data
from models.cnn_model import build_cnn_model
from utils.helper_functions import plot_training_data
from config import Config

def train_model():
    # Load the data
    X_train, y_train, X_test, y_test = load_preprocess_data()

    # Create the model
    model = build_cnn_model()

    # Train the model
    history = model.fit(X_train, y_train, 
                        epochs = Config.EPOCHS,
                        batch_size = Config.BATCH_SIZE,
                        validation_data = (X_test, y_test))

    # Plot training history
    plot_training_data(history)

    # Save the model
    model.save('./models/cnn_model.h5')
    print(f"Model saved to 'models/cnn_model.h5'")

if __name__ == '__main__':
    train_model()