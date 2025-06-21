from collections import Counter
import keras.utils
import numpy as np
import pandas as pd
from keras import Sequential, Model, Input
from keras.src.layers import Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dense, BatchNormalization, Dropout
import os
from keras.src.optimizers import Adam
from Sub_Functions.Evaluate import main_est_parameters, main_est_parameters_mul
from Sub_Functions.Load_data import Load_data2, balance2, train_test_split2

def build_model(x_train,num_classes):
    # Encoder
    input_layer = Input(shape=x_train.shape[1:])
    x = Conv1D(32, 3, activation='relu', padding='same')(input_layer)
    x = MaxPooling1D(2, padding='same')(x)  # Max Pooling
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv1D(64, 3, activation='relu', padding='same')(x)  # Bottleneck layer
    # decoder
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    dense_layer = Dense(16, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(dense_layer)
    # Build autoencoder model
    AE = Model(inputs=input_layer, outputs=output_layer)
    AE.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return AE

def proposed_model_main(x_train,x_test,y_train,y_test,train_percent,DB):
    unique = Counter(y_train)
    output = len(unique)

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    base_epochs=[100,200,300,400,500]
    total_epochs=base_epochs[-1]

    Checkpoint_dir=f"Checkpoints/{DB}/TP_{train_percent}"
    os.makedirs(Checkpoint_dir,exist_ok=True)
    metric_path=f"Analysis/Performance_Analysis/{DB}/"
    os.makedirs(metric_path,exist_ok=True)
    prev_epoch=0
    model=build_model(x_train,num_classes=output)

    for ep in reversed(base_epochs):
        ckt_path=os.path.join(Checkpoint_dir,f"model_epoch_{ep}.weights.h5")
        metrics_path=os.path.join(metric_path,f"metrics_{train_percent}percent_epoch{ep}.npy")
        if os.path.exists(ckt_path) and os.path.exists(metrics_path):
            print(f"Found existing full checkpoint and metrics for epoch {ep}, loading and resuming...")
            model.load_weights(ckt_path)
            prev_epoch = ep
            break

    metrics_all={}
    for end_epochs in base_epochs:
        if end_epochs<=prev_epoch:
            continue

        print(f" Training from epoch {prev_epoch + 1} to {end_epochs} for TP={train_percent}%...")

        ckt_path = os.path.join(Checkpoint_dir, f"model_epoch_{end_epochs}.weights.h5")
        metrics_path = os.path.join(metric_path, f"metrics_{train_percent}percent_epoch{end_epochs}.npy")

        try:

            model.fit(x_train,y_train, epochs=end_epochs - prev_epoch, batch_size=8,validation_split=0.2)

            model.save_weights(ckt_path)
            print(f"Checkpoint saved at: {ckt_path}")

            if y_test.ndim > 1:
                y_true = np.argmax(y_test, axis=1)
            else:
                y_true = y_test

            preds = model.predict(x_test)
            pred=np.argmax(preds,axis=1)
            metrics = main_est_parameters_mul(y_true, pred)
            drift=False
            if DB=="CICIDS2015":
                met=np.load("Threshold/CICIDS2015/metrics_stored.npy")
                if metrics[0]< met[0]-0.1 and metrics[1]<met[1]-0.1:
                    drift=True
                if drift:
                    print("Drift is detected , Retraining model is new data!..........")
                    feat1, labels1 = Load_data2("UNSW-NB15")
                    balanced_feat1, balanced_label1 = balance2("UNSW-NB15", feat1, labels1)
                    x_train1, x_test1, y_train1, y_test1 = train_test_split2(balanced_feat1, balanced_label1, percent=80)
                    metrics1 = proposed_model_main(x_train1, x_test1, y_train1, y_test1,  DB="UNSW-NB15",train_percent=train_percent)
                else:
                    metrics_all[f"epoch_{end_epochs}"] = metrics
                    print(f"NO Drift detected saving metrics in {metrics_path}")
                    np.save(metrics_path, metrics)
                    print(f"Metrics saved at: {metrics_path}")
                    prev_epoch = end_epochs
            if DB=="N-BaIoT":
                met = np.load("Threshold/CICIDS2015/metrics_stored.npy")
                if metrics[0] < met[0] - 0.1 and metrics[1] < met[1] - 0.1:
                    drift = True
                if drift:
                    print("Drift is detected , Retraining model is new data!..........")
                    feat1, labels1 = Load_data2("UNSW-NB15")
                    balanced_feat1, balanced_label1 = balance2("UNSW-NB15", feat1, labels1)
                    x_train1, x_test1, y_train1, y_test1 = train_test_split2(balanced_feat1, balanced_label1, percent=80)
                    metrics2 = proposed_model_main(x_train1, x_test1, y_train1, y_test1, DB="UNSW-NB15",train_percent=train_percent)
                else:
                    metrics_all[f"epoch_{end_epochs}"] = metrics
                    print(f"NO Drift detected saving metrics in {metrics_path}")
                    np.save(metrics_path, metrics)
                    print(f"Metrics saved at: {metrics_path}")
                    prev_epoch = end_epochs
            if DB=="UNSW-NB15":
                met = np.load("Threshold/UNSW-NB15/metrics_stored.npy")
                if metrics[0] < met[0] - 0.1 and metrics[1] < met[1] - 0.1:
                    drift = True
                if drift:
                    print("Drift is detected , Retraining model is new data!..........")
                    feat1, labels1 = Load_data2("UNSW-NB15")
                    balanced_feat1, balanced_label1 = balance2("UNSW-NB15", feat1, labels1)
                    x_train1, x_test1, y_train1, y_test1 = train_test_split2(balanced_feat1, balanced_label1, percent=80)
                    metrics2 = proposed_model_main(x_train1, x_test1, y_train1, y_test1,  DB="UNSW-NB15",train_percent=train_percent)
                else:
                    metrics_all[f"epoch_{end_epochs}"] = metrics
                    print(f"NO Drift detected saving metrics in {metrics_path}")
                    np.save(metrics_path, metrics)
                    print(f"Metrics saved at: {metrics_path}")
                    prev_epoch = end_epochs
        except KeyboardInterrupt:
            print(
                f"Training interrupted during epoch chunk {prev_epoch + 1}-{end_epochs}. Not saving checkpoint or metrics.")
            raise
    print(f"\nCompleted training for {train_percent}% up to {prev_epoch} epochs.")
    return metrics_all


