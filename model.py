import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertConfig
from tensorflow import keras
import numpy as np




def create_model(DISTILBERT_DROPOUT,DISTILBERT_ATT_DROPOUT,LAYER_DROPOUT,MAX_LENGTH,LEARNING_RATE,RANDOM_STATE,BATCH_SIZE,EPOCHS):
    """
        Function to create a model:
            --> Starts by freezing the network layers except for the last two, allowing fine tuning of the model
            --> Then the input layers are added for the model to be fed the data
            --> The classification token is extracted from the base model outputs, this is the token used for classification given it contains gneeral understanding of the phrase as a whole
            --> Then a dropout layer is added to avoid overfitting
            --> Finall the classificatio layer is added

            --> Function creates a Keras model with the binary cross-entropy function and returns it
    """
    # Config and loading DistilBERT + freezing
    
    config = DistilBertConfig(dropout=DISTILBERT_DROPOUT, attention_dropout=DISTILBERT_ATT_DROPOUT, output_hidden_states=True)
    distilBERT = TFDistilBertModel.from_pretrained('distilbert-base-uncased', config=config)

    for layer in distilBERT.layers:
        layer.trainable = False
    for layer in distilBERT.layers[-2:]:  # Unfreeze the last 2 layers
        layer.trainable = True

    # Input layers, required for DistilBERT to work correctly
    input_ids = keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="input_ids")
    attention_mask = keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="attention_mask")

    sequence_output = distilBERT([input_ids, attention_mask])[0]
    cls_token = sequence_output[:, 0, :] # Extracting this token as used for classification tasks

    # Adding custom layers -> Dropout to avoid overfitting and dense for the final prediction
    cls_token = keras.layers.Dropout(LAYER_DROPOUT)(cls_token)
    out = keras.layers.Dense(1, activation='sigmoid')(cls_token)

    # Creating the model
    modelo = keras.Model(inputs=[input_ids, attention_mask], outputs=out)
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy', # could change to focal loss function / weighted binary cross-entropy to deal with over/underpredictions
                  metrics=['accuracy'])
    return modelo

def pad_sequences(X_train_ids, X_train_attention, X_valid_ids, X_valid_attention, X_test_ids, X_test_attention):
    """
        Function to pad the data to feed the model, ensuring expecterd size by model
    """
    MAX_LENGTH = 500 # Max allowed by DistilBERT
    X_train_ids_padded = np.array(keras.preprocessing.sequence.pad_sequences(X_train_ids, maxlen=MAX_LENGTH, padding='post'), dtype=np.int32)
    X_train_attention_padded = np.array(keras.preprocessing.sequence.pad_sequences(X_train_attention, maxlen=MAX_LENGTH, padding='post'), dtype=np.int32)
    X_valid_ids_padded = np.array(keras.preprocessing.sequence.pad_sequences(X_valid_ids, maxlen=MAX_LENGTH, padding='post'), dtype=np.int32)
    X_valid_attention_padded = np.array(keras.preprocessing.sequence.pad_sequences(X_valid_attention, maxlen=MAX_LENGTH, padding='post'), dtype=np.int32)
    X_test_ids_padded = np.array(keras.preprocessing.sequence.pad_sequences(X_test_ids, maxlen=MAX_LENGTH, padding='post'), dtype=np.int32)
    X_test_attention_padded = np.array(keras.preprocessing.sequence.pad_sequences(X_test_attention, maxlen=MAX_LENGTH, padding='post'), dtype=np.int32)
    return (X_train_ids_padded, X_train_attention_padded), (X_valid_ids_padded, X_valid_attention_padded), (X_test_ids_padded, X_test_attention_padded)

# --> Can use bellow to replace the binary-crossentropy, make sure to tune weights by testing <-- 
def weighted_binary_crossentropy(y_true,y_pred, weight_false_negative=1.0,weight_false_positive=1.0):
    epsilon = tf.keras.baclend.epsilon()
    y_pred = tf.clip_by_value(y_pred,epsilon,1-epsilon)

    #calculate
    loss = -(weight_false_negative*y_true*tf.math.log(y_pred) + weight_false_positive*(1-y_true)*tf.math.log(1-y_pred))
    return tf.reduce_mean(loss)

