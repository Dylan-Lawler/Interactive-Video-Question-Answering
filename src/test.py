import json
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from transformers import BertTokenizer, TFBertForQuestionAnswering

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the test data
test_qa_path = '/content/drive/MyDrive/MSVD-QA/test_qa.json'

def load_json_data(path):
    with open(path, 'r') as file:
        return json.load(file)

# Load the test QA data
test_qa = load_json_data(test_qa_path)
test_df = pd.DataFrame(test_qa)

# Load the BERT model and tokenizer
qa_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
qa_model = TFBertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Define a function to calculate confidence score
def get_confidence_score(logits):
    probs = tf.nn.softmax(logits, axis=-1)
    confidence = tf.reduce_max(probs, axis=-1)
    return confidence.numpy()

# Define a function to generate clarifying question
def generate_clarifying_question(question, context):
    input_text = f"Q: {question} Context: {context} What is unclear?"
    encoded_dict = qa_tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=128,
        truncation=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='tf',
    )
    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    
    outputs = qa_model(input_ids, attention_mask=attention_mask)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    
    start_idx = tf.argmax(start_scores, axis=1).numpy()[0]
    end_idx = tf.argmax(end_scores, axis=1).numpy()[0]
    clarifying_question = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(input_ids[0][start_idx:end_idx+1]))
    return clarifying_question

# Define a function to test the model
def test_ivqa_model(test_df, video_features_cache, model, threshold=0.8):
    y_true = []
    y_pred = []
    confidence_scores = []
    
    for index, row in test_df.iterrows():
        video_id = row['video_id']
        question = row['question']
        true_answer = row['answer']
        
        # Retrieve precomputed video features
        video_features = video_features_cache[video_id]
        video_features = tf.reshape(video_features, [1, -1])  # Reshape to (1, num_features)
        
        # Encode the question
        encoded_dict = qa_tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=128,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='tf',
        )
        
        input_ids = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']
        
        text_features = qa_model([input_ids, attention_mask])[0]
        text_features = tf.reduce_mean(text_features, axis=0)
        text_features = tf.expand_dims(text_features, axis=0)
        
        combined_features = tf.concat([video_features, text_features], axis=1)
        
        # Get model prediction and confidence score
        logits = model(combined_features)
        confidence_score = get_confidence_score(logits)
        
        if confidence_score < threshold:
            # Generate clarifying question
            clarifying_question = generate_clarifying_question(question, "")
            print(f"Clarifying question: {clarifying_question}")
            
            # Get clarifying answer from the user
            clarifying_answer = input("Please provide the clarifying answer: ")
            
            # Encode the clarifying question and answer
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
            
            input_ids = tf.concat([input_ids, encoded_dict['input_ids']], axis=0)
            attention_mask = tf.concat([attention_mask, encoded_dict['attention_mask']], axis=0)
            
            text_features = qa_model([input_ids, attention_mask])[0]
            text_features = tf.reduce_mean(text_features, axis=0)
            text_features = tf.expand_dims(text_features, axis=0)
            
            combined_features = tf.concat([video_features, text_features], axis=1)
        
        # Get final model prediction
        logits = model(combined_features)
        predicted_answer = tf.argmax(logits, axis=-1).numpy()[0]
        
        y_true.append(true_answer)
        y_pred.append(predicted_answer)
        confidence_scores.append(confidence_score)
    
    return y_true, y_pred, confidence_scores

# Assuming `video_features_cache` and `model` are preloaded and defined
y_true, y_pred, confidence_scores = test_ivqa_model(test_df, video_features_cache, model)

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
