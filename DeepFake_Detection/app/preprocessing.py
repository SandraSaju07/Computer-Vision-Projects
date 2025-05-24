import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 224


def crop_center_square(frame):
    y, x = frame.shape[:2]
    min_dim = min(y, x)
    start_x = (x - min_dim) // 2
    start_y = (y - min_dim) // 2
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def augment_frame(frame):
    frame = tf.image.random_flip_left_right(frame)
    frame = tf.image.random_brightness(frame, 0.2)
    frame = tf.image.random_contrast(frame, 0.8, 1.2)
    return frame


def load_video(path, max_frames=20, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or len(frames) == max_frames:
            break
        frame = crop_center_square(frame)
        frame = cv2.resize(frame, resize)
        frame = frame[:, :, ::-1]  # BGR to RGB
        frame = augment_frame(frame)
        frames.append(frame)
    cap.release()
    return np.array(frames)
