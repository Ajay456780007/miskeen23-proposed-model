import numpy as np
from keras.src.layers import BatchNormalization, Dropout, Conv1D, MaxPooling1D
from keras.src.metrics.accuracy_metrics import accuracy
from keras.src.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.models import load_model
from collections import Counter
from Sub_Functions.Evaluate import main_est_parameters
from keras.utils import to_categorical
from Sub_Functions.Load_data import Load_data2, train_test_split2
from Sub_Functions.Load_data import balance2


def LW_CNN(x_train,x_test,y_train,y_test,epoch):
    x_train = np.expand_dims(x_train, axis=-1)  # Shape becomes (samples, 76, 1)
    x_test = np.expand_dims(x_test, axis=-1)

    model = Sequential([
    Conv1D(32, 2, activation='relu', input_shape=(x_train.shape[1:])),
    MaxPooling1D(2, 2),
    BatchNormalization(),
    Dropout(0.3),

    Conv1D(64, 2, activation='relu'),
    MaxPooling1D(2, 2),
    BatchNormalization(),
    Dropout(0.3),

    Conv1D(128, 2, activation='relu'),
    MaxPooling1D(2, 2),
    BatchNormalization(),
    Dropout(0.5),

    Flatten(),
    Dense(256, activation='relu'),
    Dense(64,activation="relu"),
    Dense(10, activation='softmax')
    ])

    model.compile(optimizer=Adam(),loss="categorical_crossentropy",metrics=["accuracy"])


    # print("THe unique values count in Y_train:",Counter(y_train))
    #
    # print("THe unique values count in Y_test",Counter(y_test))


    y_train_cat = to_categorical(y_train)

    y_test_cat=to_categorical(y_test)

    model.fit(x_train,y_train_cat,epochs=epoch,batch_size=10,validation_split=0.2)


    y_pred=model.predict(x_test)

    y_true=np.argmax(y_test_cat,axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    metrics=main_est_parameters(y_true,y_pred_classes)

    return metrics


