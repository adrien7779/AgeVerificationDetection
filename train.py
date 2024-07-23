import os
import tensorflow as tf
from tensorflow import keras
from data_preparation import get_data
from tokenization import tokenize_data
from model import create_model, pad_sequences
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def train(DISTILBERT_DROPOUT,DISTILBERT_ATT_DROPOUT,LAYER_DROPOUT,MAX_LENGTH,LEARNING_RATE,RANDOM_STATE,BATCH_SIZE,EPOCHS,modelo=None):
    """
        Passes the tokenized data for training:
            --> if model doesn't yest exist, it creates one
            --> The batches per epoch is the main parameter to look out for here
    """
    
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = get_data()
    (X_train_ids, X_train_attention), (X_valid_ids, X_valid_attention), (X_test_ids, X_test_attention) = tokenize_data(x_train, x_valid, x_test, BATCH_SIZE,MAX_LENGTH)
    (X_train_ids_padded, X_train_attention_padded), (X_valid_ids_padded, X_valid_attention_padded), (X_test_ids_padded, X_test_attention_padded) = pad_sequences(X_train_ids, X_train_attention, X_valid_ids, X_valid_attention, X_test_ids, X_test_attention)

    y_train = np.array(y_train, dtype=np.float32)
    y_valid = np.array(y_valid, dtype=np.float32)

    train_dataset = tf.data.Dataset.from_tensor_slices(({"input_ids": X_train_ids_padded, "attention_mask": X_train_attention_padded}, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE).repeat()

    valid_dataset = tf.data.Dataset.from_tensor_slices(({"input_ids": X_valid_ids_padded, "attention_mask": X_valid_attention_padded}, y_valid))
    valid_dataset = valid_dataset.batch(len(X_valid_ids_padded)) 
    
    if modelo is None:
        modelo = create_model(DISTILBERT_DROPOUT,DISTILBERT_ATT_DROPOUT,LAYER_DROPOUT,MAX_LENGTH,LEARNING_RATE,RANDOM_STATE,BATCH_SIZE,EPOCHS)
    
    batch_per_epoch = 200 # This is to ensure training data repetition accross batches 
    history = modelo.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=batch_per_epoch, validation_data=valid_dataset, verbose=1)  # steps_per_epoch=len(X_train_ids_padded) // BATCH_SIZE
    
    # Retreiving the final training values from the history object returned by Keras:
    final_training_accuracy = history.history['accuracy'][-1]
    final_validation_accuracy = history.history['val_accuracy'][-1]
    
    print(f"Model Trained With Hyperparameters: epochs={EPOCHS},batches={BATCH_SIZE},lr={LEARNING_RATE}")
    return final_training_accuracy, final_validation_accuracy, modelo