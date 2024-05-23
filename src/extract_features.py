import json
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from transformers import BertTokenizer, TFBertForQuestionAnswering

# Load video frames using name mapping
def load_video_frames(video_id, max_frames=120):
    video_name = name_mapping.get('vid' + str(video_id), None)
    if not video_name:
        raise ValueError(f"No video found for video_id: {video_id}")

    video_full_path = video_path + video_name + '.avi'
    cap = cv2.VideoCapture(video_full_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_full_path}")

    frames = []
    try:
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))  # Resize to model's input size
            frame = frame.astype('float32') / 255.0  # Normalization
            frames.append(frame)
    finally:
        cap.release()
    return np.array(frames)

# TensorFlow Hub model
model_url = 'https://tfhub.dev/deepmind/i3d-kinetics-400/1'
video_model = hub.KerasLayer(model_url)

def extract_features(frames):
    # Ensure frames are in the right shape: (batch_size, num_frames, height, width, channels)
    frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    features = video_model(frames)  # Extract features
    return features
