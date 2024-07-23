import train
import os
import csv
import warnings
import tensorflow as tf
from transformers import DistilBertTokenizerFast
from tokenization import batch_encode
from model import create_model,pad_sequences
import numpy as np
from tensorflow import keras
warnings.filterwarnings("ignore")

############################ IGNORE #################################
# Hyper parameters
DISTILBERT_DROPOUT = 0.2
DISTILBERT_ATT_DROPOUT = 0.2
LAYER_DROPOUT = 0.2
MAX_LENGTH = 500
LEARNING_RATE = [0.1]
RANDOM_STATE = 42
BATCH_SIZE = [1] # Can create a list to run a grid-search 
EPOCHS = [1]

filename = "grid_search_bce_5.csv"
header = ["learning rate","epochs","batch_size","training accuracy","validation accuracy"]
######################################################################

if __name__ == "__main__":
    """
        Runs the code to train / retrain a model:
        --> Will need to specify the path as well as the hyperparameters bellow

    """
    with open(filename, 'a', newline="") as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(header)
        
        reTrain = False # Change to True if want to retrain an existin model, instead of overwriting the weights
        model_path = 'saved_model/unfrozenModel_weights4_batched.h5' # Select model file name
        
        
        if reTrain == True:
            print(f"Loading model from {model_path}")
            LEARNING_RATE = 0.000001
            EPOCHS = 25
            BATCH_SIZE=1
            new_model = create_model(DISTILBERT_DROPOUT,DISTILBERT_ATT_DROPOUT,LAYER_DROPOUT,MAX_LENGTH,LEARNING_RATE,RANDOM_STATE,BATCH_SIZE,EPOCHS)
            new_model.load_weights(model_path)
            accuracy,val_accuracy,modelo = train.train(DISTILBERT_DROPOUT,DISTILBERT_ATT_DROPOUT,LAYER_DROPOUT,MAX_LENGTH,LEARNING_RATE,RANDOM_STATE,BATCH_SIZE,EPOCHS,modelo=new_model)
        else:
            print("Training a new Model")
            new_model = None
            BATCH_SIZE = 12
            EPOCHS = 10
            LEARNING_RATE = 1e-5
            accuracy,val_accuracy,modelo = train.train(DISTILBERT_DROPOUT,DISTILBERT_ATT_DROPOUT,LAYER_DROPOUT,MAX_LENGTH,LEARNING_RATE,RANDOM_STATE,BATCH_SIZE,EPOCHS,modelo=new_model)
           
        data = [str(LEARNING_RATE),str(EPOCHS),str(BATCH_SIZE),str(accuracy),str(val_accuracy)]
        csvwriter.writerow(data)

        model_save_path = 'saved_model/unfrozenModel_weights4_batched.h5'  
        modelo.save_weights(model_save_path)

        new_model

    file.close()
    print(f"FINSIHED WITH MODIFICATIONS, INCLUDING SAVING MODEL")

