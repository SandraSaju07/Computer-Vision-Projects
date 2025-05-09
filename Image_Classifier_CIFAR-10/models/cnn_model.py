# Import required libraries
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore

# Define CNN model
def build_cnn_model():
    model = models.Sequential([
        # Convolutional layer 1
        layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32, 32, 3)),
        layers.MaxPooling2D((2, 2)),

        # Convolutional layer 2
        layers.Conv2D(64, (3, 3), activation = 'relu'),
        layers.MaxPooling2D((2, 2)),

        # Convolutional layer 3
        layers.Conv2D(64, (3, 3), activation = 'relu'),


        # Flatten and Fully Connected layers
        layers.Flatten(),
        layers.Dense(64, activation = 'relu'),
        layers.Dense(10, activation = 'softmax')  # Output layer
    ])

    model.compile(optimizer = 'adam',
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy'])
    
    return model