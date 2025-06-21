from collections import Counter

import keras
import numpy as np
from keras import Sequential, Input
from keras.src.layers import Dense, Dropout, LSTM, Bidirectional, Reshape, MaxPooling2D, Conv2D, Conv1D, MaxPooling1D
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical
import keras
from Sub_Functions.Evaluate import main_est_parameters
from Sub_Functions.Load_data import Load_data2, balance2, train_test_split2


def CNN_LSTM_1(x_train,x_test,y_train,y_test,epoch):

    model = Sequential()
    unique = Counter(y_train)
    y_train = keras.utils.to_categorical(y_train)
    x_train = np.expand_dims(x_train, axis=-1)  # Shape becomes (samples, 76, 1)
    x_test = np.expand_dims(x_test, axis=-1)

    model.add(Conv1D(32, 3, padding="same", activation="relu", input_shape=x_train.shape[1:]))
    model.add(MaxPooling1D(2, strides=2))

    model.add(Conv1D(32, 3, padding="same", activation="relu"))
    model.add(MaxPooling1D(2, strides=2))

    model.add(Conv1D(64, 3, padding="same", activation="relu"))
    model.add(MaxPooling1D(2, strides=2))

    model.add(Conv1D(64, 3, padding="same", activation="relu"))
    model.add(MaxPooling1D(2, strides=2))

    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(128))

    model.add(Dropout(0.2))
    model.add(Dense(len(unique), activation="softmax"))


    model.compile(optimizer=Adam(),loss="categorical_crossentropy",metrics=["accuracy"])

    model.fit(x_train,y_train,epochs=epoch,batch_size=10,validation_split=0.2)

    y_pred=model.predict(x_test)

    y_true=y_test
    y_pred_classes = np.argmax(y_pred, axis=1)

    metrics=main_est_parameters(y_true,y_pred_classes)

    return metrics

feat,labels=Load_data2("UNSW-NB15")
balanced_feat,balanced_label=balance2("UNSW-NB15",feat,labels)
x_train,x_test,y_train,y_test=train_test_split2(balanced_feat,balanced_label,percent=80)
metrics=CNN_LSTM_1(x_train,x_test,y_train,y_test,epoch=50)
print(metrics)