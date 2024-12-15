# Import required libraries
import tensorflow as tf
from data.data_loader import load_preprocessed_data
from models.model import create_model
from utils.helper_functions import plot_training_history
from config import Config

# Function to train the model
def train():
    X_train, y_train, X_test, y_test, = load_preprocessed_data()

    # Create the model
    model = create_model()

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs = Config.EPOCHS,
        batch_size = Config.BATCH_SIZE,
        validation_data = (X_test, y_test)
    )

    # Plot training history
    plot_training_history(history)

    # Save the model
    model.save('./models/model.h5')
    print(f"Model saved to './models/model.h5'")

if __name__ =="__main__":
    train()