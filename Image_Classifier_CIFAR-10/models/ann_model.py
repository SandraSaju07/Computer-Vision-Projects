# Import required libraries
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore

# Define ANN model
def build_ann_model():
    model = models.Sequential([
        # Flatten and Fully Connected layers
        layers.Flatten(input_shape = (32, 32, 3)),
        layers.Dense(3000, activation = 'relu'),
        layers.Dense(1000, activation = 'relu'),
        layers.Dense(10, activation = 'sigmoid')
    ])

    model.compile(optimizer = 'SGD',
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy'])

    return model

