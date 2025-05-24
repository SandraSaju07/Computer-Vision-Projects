# Training Script
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from app.preprocessing import load_video
from app.model import build_feature_extractor, build_sequence_model
from sklearn.utils.class_weight import compute_class_weight

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
IMG_SIZE = 224

data_folder = "./data"
metadata_path = os.path.join(data_folder, "metadata.json")
df = pd.read_json(metadata_path).T

df = df[df['label'].isin(['REAL', 'FAKE'])]
df['label'] = df['label'].map({'FAKE': 1, 'REAL': 0})

train_df, val_df = train_test_split(df, test_size=0.2,
                                    stratify=df['label'])


def extract_features(df):
    feature_extractor = build_feature_extractor()
    features, masks, labels = [], [], []

    for name, label in df.iterrows():
        video_path = os.path.join(data_folder, name)
        frames = load_video(video_path)
        frame_mask = np.zeros((MAX_SEQ_LENGTH,), dtype=bool)
        frame_feature = np.zeros((MAX_SEQ_LENGTH, NUM_FEATURES),
                                 dtype='float32')
        length = min(MAX_SEQ_LENGTH, len(frames))

        for j in range(length):
            frame_feature[j] = feature_extractor.predict(
                frames[None, j]
            )
        frame_mask[:length] = 1

        features.append(frame_feature)
        masks.append(frame_mask)
        labels.append(label['label'])

    return np.array(features), np.array(masks), np.array(labels)


train_feat, train_mask, train_labels = extract_features(train_df)
val_feat, val_mask, val_labels = extract_features(val_df)

print(f"Frame features in the train set: {train_feat[0].shape}")
print(f"Frame masks in the train set: {train_feat[1].shape}")

print(f"Frame features in the validation set: {val_feat[0].shape}")
print(f"Frame masks in the validation set: {val_mask[1].shape}")

model = build_sequence_model()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "./model/deepfake_model_checkpoint.weights.h5",
    save_best_only=True,
    save_weights_only=True
)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = dict(enumerate(class_weights))

history = model.fit(
    [train_feat, train_mask], train_labels,
    validation_data=([val_feat, val_mask], val_labels),
    epochs=10,
    batch_size=16,
    callbacks=[checkpoint_cb],
    class_weight=class_weights
)

model.save("./model/deepfake_model.h5")
