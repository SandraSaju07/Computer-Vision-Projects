import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048


def build_feature_extractor():
    base_model = tf.keras.applications.InceptionV3(
        weights='imagenet', include_top=False, pooling='avg',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.inception_v3.preprocess_input(inputs)
    outputs = base_model(x)
    return models.Model(inputs, outputs, name='feature_extractor')


def build_sequence_model():
    inputs = layers.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask = layers.Input((MAX_SEQ_LENGTH,), dtype='bool')
    x = layers.GRU(64, return_sequences=True)(inputs, mask=mask)
    x = layers.BatchNormalization()(x)
    x = layers.GRU(32, return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.GRU(16)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    return models.Model([inputs, mask], output)