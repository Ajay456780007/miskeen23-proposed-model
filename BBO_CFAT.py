from collections import Counter

import numpy as np
import pandas as pd
import keras
from keras.src.layers import Conv1D, BatchNormalization, LayerNormalization, Dense, Dropout, MultiHeadAttention, Add, \
    Flatten
from keras import Sequential, Model, Input
import tensorflow as tf
from keras.src.optimizers import Adam

from Sub_Functions.Evaluate import main_est_parameters_mul
from Sub_Functions.Load_data import Load_data2, balance2, train_test_split2


def BBO_CFAT_1(x_train,x_test,y_train,y_test,epochs):
    unique=Counter(y_train)
    y_train = keras.utils.to_categorical(y_train)
    x_train = np.expand_dims(x_train, axis=-1)  # Shape becomes (samples, 76, 1)
    x_test = np.expand_dims(x_test, axis=-1)
    def ffn(x, d_ff=2048, dropout=0.1):
        d_model = x.shape[-1]
        residual = x
        x = LayerNormalization()(x)
        x = Dense(d_ff, activation="gelu")(x)
        x = Dropout(dropout)(x)
        x = Dense(d_model)(x)
        x = Dropout(dropout)(x)
        return x + residual

    inputs = Input(shape=x_train.shape[1:])
    x = Conv1D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(filters=32, kernel_size=1, activation="gelu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=32, kernel_size=1, activation="gelu", padding="same")(x)
    x = Conv1D(filters=16, kernel_size=2, strides=2, activation="relu", padding="same")(x)

    #LPU Block
    x = Conv1D(filters=16, kernel_size=2, activation="relu", padding="same")(x)
    x = LayerNormalization()(x)

    #Multihead Attention
    attention_output = MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
    x = Add()([x, attention_output])
    x = LayerNormalization()(x)

    #Feed Forward Network
    x=ffn(x,d_ff=64)

    # final output
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(len(unique), activation="softmax")(x)

    model=Model(inputs=inputs,outputs=outputs)

    model.compile(optimizer=Adam(),loss="categorical_crossentropy",metrics=["accuracy"])
    model.fit(x_train,y_train,batch_size=10,epochs=epochs,validation_split=0.2)

    preds=model.predict(x_test)
    pred=np.argmax(preds,axis=1)
    y_true=y_test

    metrics = main_est_parameters_mul(y_true, pred)

    return metrics

feat,labels=Load_data2("UNSW-NB15")
balanced_feat,balanced_label=balance2("UNSW-NB15",feat,labels)
x_train,x_test,y_train,y_test=train_test_split2(balanced_feat,balanced_label,percent=80)
metrics=BBO_CFAT_1(x_train,x_test,y_train,y_test,epochs=50)
print(metrics)