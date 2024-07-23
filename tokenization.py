import os
import tensorflow as tf
import random
from tensorflow import keras
from keras._tf_keras.keras.layers import *
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertModel, DistilBertConfig
from keras import *


def batch_encode(tokenizer, texts, BATCH_SIZE, MAX_LENGTH):
    """
        Encodes the training data:
            --> Generates the padded input as well as the mask as required by the Keras model
            --> Returns these for the model to use as input data
    """
    input_ids = []
    attention_mask = [] # Attention mask has 1 or 0 and so is multiplied so that the input padding is left as zeros so not accounted for
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer.batch_encode_plus(batch, max_length=MAX_LENGTH, padding='longest', truncation=True, return_attention_mask=True, return_token_type_ids=False)
        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])

    max_len = max(len(sublist) for sublist in input_ids)
    min_len = min(len(sublist) for sublist in input_ids)
    padded_input_ids = keras.preprocessing.sequence.pad_sequences(input_ids, padding='post')
    padded_attention_mask = keras.preprocessing.sequence.pad_sequences(attention_mask, padding='post')

    return tf.convert_to_tensor(padded_input_ids), tf.convert_to_tensor(padded_attention_mask)

def tokenize_data(x_train, x_valid, x_test,BATCH_SIZE,MAX_LENGTH):
    """
        Simply generates the input data for the model to use
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    X_train_ids, X_train_attention = batch_encode(tokenizer, x_train, BATCH_SIZE, MAX_LENGTH)
    X_valid_ids, X_valid_attention = batch_encode(tokenizer, x_valid, BATCH_SIZE, MAX_LENGTH)
    X_test_ids, X_test_attention = batch_encode(tokenizer, x_test, BATCH_SIZE, MAX_LENGTH)

    return (X_train_ids, X_train_attention), (X_valid_ids, X_valid_attention), (X_test_ids, X_test_attention)