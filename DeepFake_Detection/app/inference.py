import numpy as np
from app.preprocessing import load_video
from app.model import build_feature_extractor

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

feature_extractor = build_feature_extractor()


def prepare_single_video(path):
    frames = load_video(path)
    frames = frames[None, ...]
    frame_mask = np.zeros((1, MAX_SEQ_LENGTH), dtype=bool)
    frame_features = np.zeros((1, MAX_SEQ_LENGTH, NUM_FEATURES),
                              dtype='float32')
    for i, batch in enumerate(frames):
        length = min(MAX_SEQ_LENGTH, batch.shape[0])
        for j in range(length):
            frame_features[i, j] = feature_extractor.predict(
                batch[None, j])
        frame_mask[i, :length] = 1
    return frame_features, frame_mask