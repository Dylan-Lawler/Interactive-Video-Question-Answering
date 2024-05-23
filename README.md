

# Interactive Video Question Answering (IVQA)

Expands on MSVD-QA by adding clarifying questions and answers to the dataset to prompt the model to ask clarifying questions if its confidence score isn't high enough. Trained the Base BERT Model for text and I3D for Video.

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Usage](#usage)
4. [Testing](#testing)
5. [Switching Between IVQA and Baseline Models](#switching-between-ivqa-and-baseline-models)

## Introduction

Visual Question Answering (VQA) systems aim to answer questions about visual content, but their performance often suffers from ambiguous or insufficient context in the initial queries. This project proposes an enhanced VQA framework that incorporates clarifying questions into the training process to improve the model’s ability to seek additional context when its confidence is low. By extending the MSVD-QA dataset to include clarifying questions and answers, our approach allows the model to engage in a dialogue-like interaction to refine its understanding before providing a final response.

## Setup

### Clone the Repository
```sh
git clone https://github.com/yourusername/Interactive-VideoQA.git
cd Interactive-Video-Question-Answering
```

### Download the Dataset
Download the dataset and place it in the `data/` folder. You can follow the instructions provided in the dataset source or use a custom script to download it. Please refer to https://github.com/xudejing/video-question-answering (the MSVD-QA GitHub) for the folder of the videos. 

## Usage

### Data Preparation
The Pretrained model can be found in the IVQA.pynb script prepared to be trained on the clarifying training set.

1. **Load Data**: This script loads the dataset and prepares it for training and testing.
   ```sh
   python src/load_data.py
   ```

2. **Extract Features**: This script extracts video features using the I3D model and text features using BERT.
   ```sh
   python src/extract_features.py
   ```

3. **Prepare Data for Training**: This script combines the video and text features and prepares them for model training.
   ```sh
   python src/prep_data.py
   ```

### Model Training
1. **Train the Model**: This script trains the model using the prepared data.
   ```sh
   python src/train.py
   ```

### Model Testing
1. **Test the Model**: This script tests the model and evaluates its performance.
   ```sh
   python src/test.py
   ```

## Testing

To change the test being run, modify the `test.py` script. You can change the confidence threshold for asking clarifying questions, or adjust other parameters. Here's an example section of the `test.py` script you might want to modify:

```python
# Define a function to test the model
def test_ivqa_model(test_df, video_features_cache, model, threshold=0.8):
    # your testing code here
```

You can change the `threshold` parameter to adjust the confidence level required for the model to ask clarifying questions and add a loop to force number of questions being ask. For the time test I used Google CoLab's Runtime. 

## Switching Between IVQA and Baseline Models

To switch between using the Interactive Visual Question Answering (IVQA) model and the baseline VQA model, modify the training dataset in the `train.py` script. Change the dataset from `clarify_train_qa` to `train_qa` as follows:

```python
# Use this for the IVQA model
train_qa_path = '/path/to/clarify_train_qa.json'

# Use this for the baseline model
# train_qa_path = '/path/to/train_qa.json'
```

Comment or uncomment the appropriate lines to switch between the IVQA and baseline models.

