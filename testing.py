import tensorflow as tf
from transformers import DistilBertTokenizerFast
from tokenization import batch_encode
from model import pad_sequences,create_model
import os,json
import numpy as np
from tensorflow import keras



def load_modelo(model_path):
    """
        Preloading the model:
            --> requires the hyperparameters to be redefined
    """
    DISTILBERT_DROPOUT = 0.2
    DISTILBERT_ATT_DROPOUT = 0.2
    LAYER_DROPOUT = 0.2
    MAX_LENGTH = 500
    RANDOM_STATE = 42
    LEARNING_RATE = 0.00001
    EPOCHS = 1
    BATCH_SIZE=1
    new_model = create_model(DISTILBERT_DROPOUT,DISTILBERT_ATT_DROPOUT,LAYER_DROPOUT,MAX_LENGTH,LEARNING_RATE,RANDOM_STATE,BATCH_SIZE,EPOCHS)
    new_model.load_weights(model_path)
    return new_model


def prepare_text(new_texts):
    """
        Prepares the text in a format for the Keras model to understand:
            
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    BATCH_SIZE=1
    MAX_LENGTH=500
    
    # Tokenize the new texts
    inputs_ids, inputs_attention = batch_encode(tokenizer, new_texts,BATCH_SIZE, MAX_LENGTH)
    inputs_ids_padded = np.array(keras.preprocessing.sequence.pad_sequences(inputs_ids, maxlen=MAX_LENGTH, padding='post'), dtype=np.int32)
    inputs_attention_padded = np.array(keras.preprocessing.sequence.pad_sequences(inputs_attention, maxlen=MAX_LENGTH, padding='post'), dtype=np.int32)
    return inputs_ids_padded,inputs_attention_padded

# Make predictions
def predict(data_to_compare):
    """
        Uses model to infer on the data input as parameter:
            --> requires the model path
            --> the threshold values bellow can be tuned to print or not positives, unsure or negatives. What the model takes as a real positive will be based on the htmlExtract main function

    """
    model_path = 'saved_model/unfrozenModel_weights4_batched.h5'
    new_model = load_modelo(model_path)

    inputs_ids_padded,inputs_attention_padded = prepare_text(data_to_compare)
    
    predictions = new_model.predict({'input_ids': inputs_ids_padded, 'attention_mask': inputs_attention_padded})
    
    for count,p in enumerate(predictions,start=1):
        if p < 0.5:
            pass
            #print(f"Item {count}: NO with:{p}")
        elif p > 0.999:
            print(f"Item {count}: YES with:{p} item: {data_to_compare[count-1]}")
        else:
            print(f"Item {count}: Unsure with:{p} item: {data_to_compare[count-1]}m")
    return predictions

def main1(text_lines):
    prediction_vals = predict(text_lines) 
    return prediction_vals

