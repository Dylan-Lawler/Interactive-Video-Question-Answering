import json
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from google.colab import drive
import cv2
from transformers import BertTokenizer, TFBertForQuestionAnswering

drive.mount('/content/drive')

data_path = '/content/drive/MyDrive/MSVD-QA/'
video_path = data_path + 'video/'
train_qa_path = data_path + 'train_qa.json'
clarify_train_qa_path = data_path + 'clarify_train_qa.json'
val_qa_path = data_path + 'val_qa.json'

# Function to load JSON data
def load_json_data(path):
    with open(path, 'r') as file:
        return json.load(file)

# Loading QA data
train_qa = load_json_data(train_qa_path)
clarify_train_qa = load_json_data(clarify_train_qa_path)
val_qa = load_json_data(val_qa_path)

# Convert QA data to DataFrame
train_df = pd.DataFrame(train_qa)
clarify_train_df = pd.DataFrame(clarify_train_qa)
val_df = pd.DataFrame(val_qa)

name_mapping_path = data_path + 'youtube_mapping.txt'

def load_name_mapping(path):
    mapping = {}
    with open(path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                video_name, video_id = parts
                mapping[video_id] = video_name
    return mapping

name_mapping = load_name_mapping(name_mapping_path)
