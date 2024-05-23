import json
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from transformers import BertTokenizer, TFBertForQuestionAnswering


# Load BERT for QA
qa_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
qa_model = TFBertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Precompute and cache video features
video_features_cache = {}

for video_id in clarify_train_df['video_id'].unique():
    print(video_id)
    frames = load_video_frames(video_id)
    video_features = extract_features(frames)

# Prepare data for training
X_train = []
y_train = []

for _, row in clarify_train_df.iterrows():
    video_id = row['video_id']
    question = row['question']
    answer = row['answer']
    clarifications = row['clarifications']

    # Retrieve precomputed video features
    video_features = video_features_cache[video_id]
    video_features = tf.reshape(video_features, [1, -1])  # Reshape to (1, num_features)

    input_ids = []
    attention_masks = []

    encoded_dict = qa_tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length=128,
        truncation=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='tf',
    )

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

    for clarification in clarifications:
        clarifying_question = clarification['clarifying_question']
        clarifying_answer = clarification['clarifying_answer']
        clarifying_text = clarifying_question + " " + clarifying_answer

        encoded_dict = qa_tokenizer.encode_plus(
            clarifying_text,
            add_special_tokens=True,
            max_length=128,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='tf',
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = tf.concat(input_ids, axis=0)
    attention_masks = tf.concat(attention_masks, axis=0)

    text_features = qa_model([input_ids, attention_masks])[0]
    text_features = tf.reduce_mean(text_features, axis=0)
    text_features = tf.expand_dims(text_features, axis=0)

    # Ensure video and text features have compatible dimensions
    print("Video features shape:", video_features.shape)  # Expected shape: (1, num_features)
    print("Text features shape:", text_features.shape)    # Expected shape: (1, 128)

    combined_features = tf.concat([video_features, text_features], axis=1)

    X_train.append(combined_features.numpy())
    y_train.append(answer)

X_train = np.array(X_train)
y_train = np.array(y_train)
