# Import required libraries
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore

# Define model
def create_model():
    model = models.Sequential([
        # Flatten and Fully Connected Layers
        layers.Flatten(input_shape = (28,28)),
        layers.Dense(100, activation = 'relu'),
        layers.Dense(10, activation = 'sigmoid')  # Output layer
    ])

    # Compile the model
    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    return model