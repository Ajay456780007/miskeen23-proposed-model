import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from Sub_Functions.Plot import ALL_GRAPH_PLOT
from Sub_Functions.Load_data import balance
from sklearn.decomposition import PCA
from Sub_Functions.ED_Analysis import apply_smote

def Preprocessing(DB):
    if DB=="UNSW-NB15":
        # loading features
        features = pd.read_csv(f"Dataset/{DB}/NUSW-NB15_features.csv", encoding="ISO-8859-1")
        column_names = features["Name"].tolist()

        if 'Label' not in column_names:
            column_names.append('Label')
        if 'attack_cat' not in column_names:
            column_names.append('attack_cat')

        duplicates = [item for item, count in Counter(column_names).items() if count > 1]
        if duplicates:
            raise ValueError("Found Duplicate column names, Check the features.csv file")

        fresh_data = [
            f"Dataset/{DB}/{DB}_1.csv",
            f"Dataset/{DB}/{DB}_2.csv",
            f"Dataset/{DB}/{DB}_3.csv",
            f"Dataset/{DB}/{DB}_4.csv"
        ]

        fresh_df = [pd.read_csv(file, encoding="ISO-8859-1", header=None, names=column_names) for file in fresh_data]
        full_data = pd.concat(fresh_df, ignore_index=True)
        print(f"Loaded Full dataset for {DB} dataset")

        full_data.drop(columns=["ct_flw_http_mthd", "is_ftp_login", "srcip"], inplace=True)

        attack_cat_le = LabelEncoder()
        full_data['attack_cat'] = attack_cat_le.fit_transform(full_data['attack_cat'].astype(str))

        os.makedirs(f"Dataset/Encoders/", exist_ok=True)
        with open(f"Dataset/Encoders/{DB}_attack_cat_encoder.pkl", "wb") as f:
            pickle.dump(attack_cat_le, f)


        object_columns = full_data.select_dtypes(include=["object"]).columns.tolist()
        if 'attack_cat' in object_columns:
            object_columns.remove('attack_cat')

        label_encoders = {}
        for col in object_columns:
            le = LabelEncoder()
            full_data[col] = le.fit_transform(full_data[col].astype(str))
            label_encoders[col] = le

        with open(f"Dataset/Encoders/{DB}_label_encoders.pkl", "wb") as f:
            pickle.dump(label_encoders, f)
        print(f"Label Encoding completed for {DB} dataset")

        label_columns = ['Label', 'attack_cat']
        numeric_columns = [col for col in full_data.select_dtypes(include=["int32","int64", "float64"]).columns if
                           col not in label_columns]


        scaler = StandardScaler()
        full_data[numeric_columns] = scaler.fit_transform(full_data[numeric_columns])


        with open(f"Dataset/Encoders/{DB}_standard_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print("Standardization completed")

        nap_data = balance(DB,full_data, label_col='attack_cat')

        temp_features=nap_data.drop(columns=["Label","attack_cat"])

        temp_label=nap_data["attack_cat"]
        features,label=apply_smote(temp_features,temp_label)

        np.save(f"data_loader/{DB}_features.npy",features)
        np.save(f"data_loader/{DB}_labels.npy",label)

        print(f"The Data saved Successfully for {DB} dataset")

    if DB=="N-BaIoT":

        data1 = pd.read_csv(f"Dataset/{DB}/UNSW_2018_IoT_Botnet_Full5pc_1.csv", encoding="ISO-8859-1")
        data2 = pd.read_csv(f"Dataset/{DB}/UNSW_2018_IoT_Botnet_Full5pc_2.csv", encoding="ISO-8859-1")
        data3 = pd.read_csv(f"Dataset/{DB}/UNSW_2018_IoT_Botnet_Full5pc_3.csv", encoding="ISO-8859-1")
        data4 = pd.read_csv(f"Dataset/{DB}/UNSW_2018_IoT_Botnet_Full5pc_4.csv", encoding="ISO-8859-1")


        data = pd.concat([data1, data2, data3, data4], ignore_index=True)

        label_column = "category"
        if label_column not in data.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset.")

        print(f"\n[INFO] Value counts in '{label_column}':\n", data[label_column].value_counts())

        object_columns = data.select_dtypes(include=["object"]).columns.tolist()

        label_encoders = {}
        for col in object_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le

        os.makedirs(f"Dataset/Encoders/", exist_ok=True)
        with open(f"Dataset/Encoders/{DB}_label_encoders.pkl", "wb") as f:
            pickle.dump(label_encoders, f)
        print(f" Label encoding completed for {DB} dataset")

        # Gathering the numeric columns so that we can apply the standard Scalar to normalize the column values , so that maximum values only dont get priority
        numeric_columns = [col for col in data.select_dtypes(include=["int32", "int64", "float64"]).columns
                           if col not in [label_column, "attack"]]

        # creating instance for standard scalar
        scaler = StandardScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns]) # normalizing the numeric columns

        # Saving the standard scalar instance in pickle format
        with open(f"Dataset/Encoders/{DB}_standard_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print(f" Standardization completed for {DB} dataset")

        nap_data = balance(DB, data, label_col=label_column)

        # storing the columns containing the label related contents
        drop_cols = [c for c in ["attack", "subcategory", label_column] if c in nap_data.columns]
        # dropping the labels columns from the features
        temp_features = nap_data.drop(columns=drop_cols)
        # label_column="category" which we are saving in labels
        temp_label = nap_data[label_column]

        # applying smote algorithm
        features, label = apply_smote(temp_features, temp_label)

        # creating directory to store the data
        os.makedirs("data_loader", exist_ok=True)
        np.save(f"data_loader/{DB}_features.npy", features) # saving features
        np.save(f"data_loader/{DB}_labels.npy", label) # saving labels

        print(f"Data saved successfully for {DB} Dataset")

    if DB=="CICIDS2015":

        # Loading dataset
        data1 = pd.read_csv(f"Dataset/{DB}/Data.csv")
        labels = pd.read_csv(f"Dataset/{DB}/Label.csv")
        labels=labels["Label"]

        print(f"THe data loaded successfully for {DB} dataset")

        # since we have no object columns in the existing dataset we are skipping the label Encoder
        object_columns = data1.select_dtypes(include=["object"]).columns.tolist()

        # Gathering the numeric columns so that we can apply the standard Scalar to normalize the column values , so that maximum values only dont get priority
        numeric_columns = [col for col in data1.select_dtypes(include=["int32", "int64", "float64"]).columns]

        # creating instance for standard scalar
        scaler = StandardScaler()
        data1[numeric_columns] = scaler.fit_transform(data1[numeric_columns]) # normalizing the numeric columns

        # Saving the standard scalar instance in pickle format
        with open(f"Dataset/Encoders/{DB}_standard_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print(f" Standardization completed for {DB} dataset")

        # for applying smote algorithm we take the minimum samples from all classes
        nap_data,nap_lab =balance(DB,data1,label_col="Label",labels=labels)

        # stores the label and feature in a temporary variables
        temp_features= nap_data
        temp_label=nap_lab

        # applying smote algorithm to maximize the count of less available classes samples
        features,label=apply_smote(temp_features,temp_label)

        # creating directory to store the data
        os.makedirs("data_loader",exist_ok=True)
        np.save(f"data_loader/{DB}_features.npy",features) # saving features
        np.save(f"data_loader/{DB}_labels.npy",label) # saving labels

        print(f"Data saved successfully for {DB} Dataset")  # success message after saving the data











