import pickle

import numpy as np


def Load_data2(data):

    feat=np.load(f"data_loader/{data}_features.npy")
    labels=np.load(f"data_loader/{data}_labels.npy")
    return feat, labels


def train_test_split2(balanced_feat,balanced_label,percent):

    data_size = balanced_feat.shape[0]  # Checks the shape of balanced_feat to convert the training_percentage to integer
    actual_percentage = int((data_size / 100) * percent)  # Converted the float training percentage to integer
    training_sequence = balanced_feat[:actual_percentage]  # splitting the training data
    training_labels = balanced_label[:actual_percentage]  # splitting the training label
    testing_sequence = balanced_feat[actual_percentage:]   # splitting the testing sequence
    testing_labels = balanced_label[actual_percentage:]     # splitting the Testing labels

    return training_sequence,testing_sequence,training_labels,testing_labels   #The function  train_test_split1 return the training and testing data


def models_return_metrics(data,ok=True,percent=None):

    CNN_LSTM1 = []  # empty list to store KNN metrics
    LW_CNN1 = []  # empty list to store the CNN metrics
    Resnet_50_WA1 = []  # empty list to store  the CNN_Resnet metrics
    PM_WA1 = []  # empty list to store  the DiT metrics
    MobilenetV2_1 = []  # empty list to store  the HGNN metrics
    LSTM_BiLSTM1 = []  # empty list to store  the SVM metrics
    Seq_WA1 = []  # empty list to store  the WA metrics
    proposed_model1 = []  # empty list to store  the proposed_model metrics

    training_percentage = [40, 50, 60, 70, 80, 90]



    if ok:
        for i in training_percentage:
            print(f"THe training for comparative modl for {i}% training is Starting")
            x_train,x_test,y_train,y_test=train_test_split2(data,i)

            CNN_LSTM_metrics1= CNN_LSTM_1(x_train,x_test,y_train,y_test)
            CNN_LSTM1.append(CNN_LSTM_metrics1)
            print("Completed 1")

            LW_CNN_metrics1=LW_CNN(x_train,x_test,y_train,y_test)
            LW_CNN1.append(LW_CNN_metrics1)
            print("Completed 2")

            LSTM_BiLSTM_metrics1=LSTM_BiLSTM(x_train,x_test,y_train,y_test)
            LSTM_BiLSTM1.append(LSTM_BiLSTM_metrics1)
            print("Completed 3")

            MobilenetV2_metrics1=MobileNet_V2(x_train,x_test,y_train,y_test)
            MobilenetV2_1.append(MobilenetV2_metrics1)
            print("Completed 4")

            Resnet_50_WA_metrics1=Resnet_50_WA(x_train,x_test,y_train,y_test)
            Resnet_50_WA1.append(Resnet_50_WA_metrics1)
            print("Completed 5")

            Squeeze_net_WA_metrics1=Squeeze_net_WA(x_train,x_test,y_train,y_test)
            Seq_WA1.append(Squeeze_net_WA_metrics1)
            print("Completed 6")

            PM_WA_metrics1=PM_WA(x_train,x_test,y_train,y_test)
            PM_WA1.append(PM_WA_metrics1)
            print("Completed 7")

            PM_metrics1=Proposed_model(x_train,x_test,y_train,y_test)
            proposed_model1.append(PM_metrics1)
            print("Completed 8")

    else:

        x_train, x_test, y_train, y_test = train_test_split2(data, i)

        CNN_LSTM_metrics1 = CNN_LSTM_1(x_train, x_test, y_train, y_test)
        CNN_LSTM1.append(CNN_LSTM_metrics1)

        LW_CNN_metrics1 = LW_CNN(x_train, x_test, y_train, y_test)
        LW_CNN1.append(LW_CNN_metrics1)

        LSTM_BiLSTM_metrics1 = LSTM_BiLSTM(x_train, x_test, y_train, y_test)
        LSTM_BiLSTM1.append(LSTM_BiLSTM_metrics1)

        MobilenetV2_metrics1 = MobileNet_V2(x_train, x_test, y_train, y_test)
        MobilenetV2_1.append(MobilenetV2_metrics1)

        Resnet_50_WA_metrics1 = Resnet_50_WA(x_train, x_test, y_train, y_test)
        Resnet_50_WA1.append(Resnet_50_WA_metrics1)

        Squeeze_net_WA_metrics1 = Squeeze_net_WA(x_train, x_test, y_train, y_test)
        Seq_WA1.append(Squeeze_net_WA_metrics1)

        PM_WA_metrics1 = PM_WA(x_train, x_test, y_train, y_test)
        PM_WA1.append(PM_WA_metrics1)

        PM_metrics1 = Proposed_model(x_train, x_test, y_train, y_test)
        proposed_model1.append(PM_metrics1)

    return CNN_LSTM1, LSTM_BiLSTM1, LW_CNN1, MobilenetV2_1, PM_WA1, Resnet_50_WA1, Seq_WA1, proposed_model1


def balance(DB,full_dataset, label_col=None,labels=None):
    balanced_feat=None
    balanced_lab=None
    if DB=="UNSW-NB15":
        label = full_dataset[label_col]
        # We are doing the below steps to take equal amount of data from all the classes
        class_0_indices = np.where(label == 0)[0]  # this line chooses all the 0-label index in the label data
        class_1_indices = np.where(label == 1)[0]  # this line chooses all the 1-label index in the label data
        class_2_indices = np.where(label == 2)[0]  # this line chooses all the 0-label index in the label data
        class_3_indices = np.where(label == 3)[0]  # this line chooses all the 1-label index in the label data
        class_4_indices = np.where(label == 4)[0]  # this line chooses all the 0-label index in the label data
        class_5_indices = np.where(label == 5)[0]  # this line chooses all the 1-label index in the label data
        class_6_indices = np.where(label == 6)[0]  # this line chooses all the 0-label index in the label data
        class_7_indices = np.where(label == 7)[0]  # this line chooses all the 1-label index in the label data
        class_8_indices = np.where(label == 8)[0]  # this line chooses all the 0-label index in the label data
        class_9_indices = np.where(label == 9)[0]  # this line chooses all the 1-label index in the label data
        class_10_indices = np.where(label == 10)[0]  # this line chooses all the 0-label index in the label data
        class_11_indices = np.where(label == 11)[0]  # this line chooses all the 1-label index in the label data
        class_12_indices = np.where(label == 12)[0]  # this line chooses all the 0-label index in the label data
        class_13_indices = np.where(label == 13)[0]  # this line chooses all the 0-label index in the label data

        print("0:", len(class_0_indices))
        print("1:", len(class_1_indices))
        print("2:", len(class_2_indices))
        print("3:", len(class_3_indices))
        print("4:", len(class_4_indices))
        print("5:", len(class_5_indices))
        print("6:", len(class_6_indices))
        print("7:", len(class_7_indices))
        print("8:", len(class_8_indices))
        print("9:", len(class_9_indices))
        print("10:", len(class_10_indices))
        print("11:", len(class_11_indices))
        print("12:", len(class_12_indices))
        print("13:", len(class_13_indices))

        selected_class_0 = np.random.choice(class_0_indices, 5050,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_1 = np.random.choice(class_1_indices, 10000,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices

        selected_class_2 = np.random.choice(class_2_indices, 10000,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_3 = np.random.choice(class_3_indices, 1200,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_4 = np.random.choice(class_4_indices, 2600,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_5 = np.random.choice(class_5_indices, 1500,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_6 = np.random.choice(class_6_indices, 500,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_7 = np.random.choice(class_7_indices, 10000,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_8 = np.random.choice(class_8_indices, 10000,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_9 = np.random.choice(class_9_indices, 20000,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_10 = np.random.choice(class_10_indices, 1500,
                                             replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_11 = np.random.choice(class_11_indices, 223,
                                             replace=False)  # randomly chooses 1000 indices from the class_1_indices

        selected_class_12 = np.random.choice(class_12_indices, 174,
                                             replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_13 = np.random.choice(class_13_indices, 10000,
                                             replace=False)  # randomly chooses 1000 indices from the class_0_indices

        selected_indices = np.concatenate(
            [selected_class_0, selected_class_1, selected_class_2, selected_class_3, selected_class_4, selected_class_5,
             selected_class_6, selected_class_7, selected_class_8, selected_class_9, selected_class_10,
             selected_class_11,selected_class_12, selected_class_13])  # joining all the classes together
        np.random.shuffle(selected_indices)  # randomly shuffles the  selected indices

        balanced_feat = full_dataset.iloc[selected_indices]
    if DB=="N-BaIoT":
        label = full_dataset[label_col]
        # We are doing the below steps to take equal amount of data from all the classes
        class_0_indices = np.where(label == 0)[0]  # this line chooses all the 0-label index in the label data
        class_1_indices = np.where(label == 1)[0]  # this line chooses all the 1-label index in the label data
        class_2_indices = np.where(label == 2)[0]  # this line chooses all the 0-label index in the label data
        class_3_indices = np.where(label == 3)[0]  # this line chooses all the 1-label index in the label data
        class_4_indices = np.where(label == 4)[0]  # this line chooses all the 0-label index in the label data

        print("0:", len(class_0_indices))
        print("1:", len(class_1_indices))
        print("2:", len(class_2_indices))
        print("3:", len(class_3_indices))
        print("4:", len(class_4_indices))

        selected_class_0 = np.random.choice(class_0_indices, 30000,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_1 = np.random.choice(class_1_indices, 30000,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices

        selected_class_2 = np.random.choice(class_2_indices, 477,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_3 = np.random.choice(class_3_indices, 30000,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_4 = np.random.choice(class_4_indices, 79,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices

        selected_indices = np.concatenate(
            [selected_class_0, selected_class_1, selected_class_2, selected_class_3,
             selected_class_4])  # joining all the classes together
        np.random.shuffle(selected_indices)  # randomly shuffles the  selected indices

        balanced_feat = full_dataset.iloc[selected_indices]

    if DB=="CICIDS2015":
        label = labels
        # We are doing the below steps to take equal amount of data from all the classes
        class_0_indices = np.where(label == 0)[0]  # this line chooses all the 0-label index in the label data
        class_1_indices = np.where(label == 1)[0]  # this line chooses all the 1-label index in the label data
        class_2_indices = np.where(label == 2)[0]  # this line chooses all the 0-label index in the label data
        class_3_indices = np.where(label == 3)[0]  # this line chooses all the 1-label index in the label data
        class_4_indices = np.where(label == 4)[0]  # this line chooses all the 0-label index in the label data
        class_5_indices = np.where(label == 5)[0]  # this line chooses all the 1-label index in the label data
        class_6_indices = np.where(label == 6)[0]  # this line chooses all the 0-label index in the label data
        class_7_indices = np.where(label == 7)[0]  # this line chooses all the 1-label index in the label data
        class_8_indices = np.where(label == 8)[0]  # this line chooses all the 0-label index in the label data
        class_9_indices = np.where(label == 9)[0]  # this line chooses all the 1-label index in the label data

        print("0:", len(class_0_indices))
        print("1:", len(class_1_indices))
        print("2:", len(class_2_indices))
        print("3:", len(class_3_indices))
        print("4:", len(class_4_indices))
        print("5:", len(class_5_indices))
        print("6:", len(class_6_indices))
        print("7:", len(class_7_indices))
        print("8:", len(class_8_indices))
        print("9:", len(class_9_indices))

        selected_class_0 = np.random.choice(class_0_indices, 20000,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_1 = np.random.choice(class_1_indices, 385,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices

        selected_class_2 = np.random.choice(class_2_indices, 452,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_3 = np.random.choice(class_3_indices, 4467,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_4 = np.random.choice(class_4_indices, 20000,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_5 = np.random.choice(class_5_indices, 20000,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_6 = np.random.choice(class_6_indices, 4467,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_7 = np.random.choice(class_7_indices, 16735,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_8 = np.random.choice(class_8_indices, 2102,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_9 = np.random.choice(class_9_indices, 246,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices

        selected_indices = np.concatenate(
            [selected_class_0, selected_class_1, selected_class_2, selected_class_3, selected_class_4, selected_class_5,
             selected_class_6, selected_class_7, selected_class_8,
             selected_class_9])  # joining all the classes together
        np.random.shuffle(selected_indices)  # randomly shuffles the  selected indices

        balanced_feat = full_dataset.iloc[selected_indices]
        balanced_lab = labels.loc[selected_indices]
    if DB=="CICIDS2015":
        return balanced_feat,balanced_lab
    else:
        return balanced_feat



def balance2(DB,feat,labels):
    balanced_feat=None
    balanced_lab=None
    if DB=="UNSW-NB15":
        label = labels
        # We are doing the below steps to take equal amount of data from all the classes
        class_0_indices = np.where(label == 0)[0]  # this line chooses all the 0-label index in the label data
        class_1_indices = np.where(label == 1)[0]  # this line chooses all the 1-label index in the label data
        class_2_indices = np.where(label == 2)[0]  # this line chooses all the 0-label index in the label data
        class_3_indices = np.where(label == 3)[0]  # this line chooses all the 1-label index in the label data
        class_4_indices = np.where(label == 4)[0]  # this line chooses all the 0-label index in the label data
        class_5_indices = np.where(label == 5)[0]  # this line chooses all the 1-label index in the label data
        class_6_indices = np.where(label == 6)[0]  # this line chooses all the 0-label index in the label data
        class_7_indices = np.where(label == 7)[0]  # this line chooses all the 1-label index in the label data
        class_8_indices = np.where(label == 8)[0]  # this line chooses all the 0-label index in the label data
        class_9_indices = np.where(label == 9)[0]  # this line chooses all the 1-label index in the label data
        class_10_indices = np.where(label == 10)[0]  # this line chooses all the 0-label index in the label data
        class_11_indices = np.where(label == 11)[0]  # this line chooses all the 1-label index in the label data
        class_12_indices = np.where(label == 12)[0]  # this line chooses all the 0-label index in the label data
        class_13_indices = np.where(label == 13)[0]  # this line chooses all the 0-label index in the label data

        print("0:", len(class_0_indices))
        print("1:", len(class_1_indices))
        print("2:", len(class_2_indices))
        print("3:", len(class_3_indices))
        print("4:", len(class_4_indices))
        print("5:", len(class_5_indices))
        print("6:", len(class_6_indices))
        print("7:", len(class_7_indices))
        print("8:", len(class_8_indices))
        print("9:", len(class_9_indices))
        print("10:", len(class_10_indices))
        print("11:", len(class_11_indices))
        print("12:", len(class_12_indices))
        print("13:", len(class_13_indices))

        selected_class_0 = np.random.choice(class_0_indices, 3000,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_1 = np.random.choice(class_1_indices, 3000,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices

        selected_class_2 = np.random.choice(class_2_indices, 3000,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_3 = np.random.choice(class_3_indices, 3000,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_4 = np.random.choice(class_4_indices, 3000,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_5 = np.random.choice(class_5_indices, 3000,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_6 = np.random.choice(class_6_indices, 3000,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_7 = np.random.choice(class_7_indices, 3000,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_8 = np.random.choice(class_8_indices, 3000,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_9 = np.random.choice(class_9_indices, 3000,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_10 = np.random.choice(class_10_indices, 3000,
                                             replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_11 = np.random.choice(class_11_indices, 3000,
                                             replace=False)  # randomly chooses 1000 indices from the class_1_indices

        selected_class_12 = np.random.choice(class_12_indices, 3000,
                                             replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_13 = np.random.choice(class_13_indices, 3000,
                                             replace=False)  # randomly chooses 1000 indices from the class_0_indices

        selected_indices = np.concatenate(
            [selected_class_0, selected_class_1, selected_class_2, selected_class_3, selected_class_4, selected_class_5,
             selected_class_6, selected_class_7, selected_class_8, selected_class_9, selected_class_10,
             selected_class_11,selected_class_12, selected_class_13])  # joining all the classes together
        np.random.shuffle(selected_indices)  # randomly shuffles the  selected indices

        balanced_feat = feat[selected_indices]
        balanced_lab=labels[selected_indices]
    if DB=="N-BaIoT":
        label = labels
        # We are doing the below steps to take equal amount of data from all the classes
        class_0_indices = np.where(label == 0)[0]  # this line chooses all the 0-label index in the label data
        class_1_indices = np.where(label == 1)[0]  # this line chooses all the 1-label index in the label data
        class_2_indices = np.where(label == 2)[0]  # this line chooses all the 0-label index in the label data
        class_3_indices = np.where(label == 3)[0]  # this line chooses all the 1-label index in the label data
        class_4_indices = np.where(label == 4)[0]  # this line chooses all the 0-label index in the label data

        print("0:", len(class_0_indices))
        print("1:", len(class_1_indices))
        print("2:", len(class_2_indices))
        print("3:", len(class_3_indices))
        print("4:", len(class_4_indices))

        selected_class_0 = np.random.choice(class_0_indices, 8000,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_1 = np.random.choice(class_1_indices, 8000,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices

        selected_class_2 = np.random.choice(class_2_indices, 8000,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices
        selected_class_3 = np.random.choice(class_3_indices, 8000,
                                            replace=False)  # randomly chooses 1000 indices from the class_1_indices
        selected_class_4 = np.random.choice(class_4_indices, 8000,
                                            replace=False)  # randomly chooses 1000 indices from the class_0_indices

        selected_indices = np.concatenate(
            [selected_class_0, selected_class_1, selected_class_2, selected_class_3,
             selected_class_4])  # joining all the classes together
        np.random.shuffle(selected_indices)  # randomly shuffles the  selected indices

        balanced_feat = feat[selected_indices]
        balanced_lab=labels[selected_indices]

    if DB=="CICIDS2015":
        label = labels
        # We are doing the below steps to take equal amount of data from all the classes
        class_0_indices = np.where(label == 0)[0]  # this line chooses all the 0-label index in the label data
        class_1_indices = np.where(label == 1)[0]  # this line chooses all the 1-label index in the label data
        class_2_indices = np.where(label == 2)[0]  # this line chooses all the 0-label index in the label data
        class_3_indices = np.where(label == 3)[0]  # this line chooses all the 1-label index in the label data
        class_4_indices = np.where(label == 4)[0]  # this line chooses all the 0-label index in the label data
        class_5_indices = np.where(label == 5)[0]  # this line chooses all the 1-label index in the label data
        class_6_indices = np.where(label == 6)[0]  # this line chooses all the 0-label index in the label data
        class_7_indices = np.where(label == 7)[0]  # this line chooses all the 1-label index in the label data
        class_8_indices = np.where(label == 8)[0]  # this line chooses all the 0-label index in the label data
        class_9_indices = np.where(label == 9)[0]  # this line chooses all the 1-label index in the label data

        print("0:", len(class_0_indices))
        print("1:", len(class_1_indices))
        print("2:", len(class_2_indices))
        print("3:", len(class_3_indices))
        print("4:", len(class_4_indices))
        print("5:", len(class_5_indices))
        print("6:", len(class_6_indices))
        print("7:", len(class_7_indices))
        print("8:", len(class_8_indices))
        print("9:", len(class_9_indices))

        selected_class_0 = np.random.choice(class_0_indices, 4500,
                                            replace=False)
        selected_class_1 = np.random.choice(class_1_indices, 4500,
                                            replace=False)

        selected_class_2 = np.random.choice(class_2_indices, 4500,
                                            replace=False)
        selected_class_3 = np.random.choice(class_3_indices, 4500,
                                            replace=False)
        selected_class_4 = np.random.choice(class_4_indices, 4500,
                                            replace=False)
        selected_class_5 = np.random.choice(class_5_indices, 4500,
                                            replace=False)
        selected_class_6 = np.random.choice(class_6_indices, 4500,
                                            replace=False)
        selected_class_7 = np.random.choice(class_7_indices, 4500,
                                            replace=False)
        selected_class_8 = np.random.choice(class_8_indices, 4500,
                                            replace=False)
        selected_class_9 = np.random.choice(class_9_indices, 4500,
                                            replace=False)

        selected_indices = np.concatenate(
            [selected_class_0, selected_class_1, selected_class_2, selected_class_3, selected_class_4, selected_class_5,
             selected_class_6, selected_class_7, selected_class_8,
             selected_class_9])  # joining all the classes together
        np.random.shuffle(selected_indices)  # randomly shuffles the  selected indices

        balanced_feat = feat[selected_indices]
        balanced_lab = labels[selected_indices]

    return balanced_feat,balanced_lab
