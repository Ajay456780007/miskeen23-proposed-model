import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from termcolor import cprint, colored

from Proposed_model.PM1 import proposed_model_main
from Sub_Functions.Load_data import train_test_split2, Load_data2, balance2


class Analysis:

    def __init__(self,Data):
        self.lab=None
        self.feat=None
        self.DB=Data
        self.E=[20,40,60,80,100]

    def Data_loading(self):
        feat=np.load("dat_loader/feature.npy")
        label=np.load("data_loader/labels.npy")

    def COMP_Analysis(self):
        self.Data_loading()
        tr = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Each model will return metrics over 6 thresholds × 8 metrics
        C1, C2, C3, C4, C5, C6, C7, C8 = [[] for _ in range(8)]

        (KNN_metrics, CNN_metrics, CNN_Resnet_metrics, SVM_metrics,
         DIT_metrics, HGNN_metrics, WA_metrics) = models_return_metrics(self.DB, ok=True)

        # Each is of shape (6, 8)
        C1 = KNN_metrics
        C2 = CNN_metrics
        C3 = CNN_Resnet_metrics
        C4 = SVM_metrics
        C5 = DIT_metrics
        C6 = HGNN_metrics
        C7 = WA_metrics


        # Now create a list of all model metrics
        all_models = [C1, C2, C3, C4, C5, C6, C7]

        perf_names = ["ACC", "SEN", "SPE", "F1score", "REC", "PRE", "TPR", "FPR"]
        files_name = [f"Analysis/Comparative_Analysis/{self.DB}/{name}_1.npy" for name in perf_names]

        # For each metric index j (0-7), collect it across all models and thresholds
        for j in range(len(perf_names)):
            new = []
            for model_metrics in all_models:  # each is shape (6, 8)
                # Extract j-th metric across all percentages (i.e., column j)
                x = [row[j] for row in model_metrics]  # shape (6,)
                new.append(x)  # new will be shape (8 models × 6 thresholds)
            np.save(files_name[j], np.array(new))  # shape (8, 6)

    def KF_Analysis(self):
        self.Data_loading()

        kr = [6, 7, 8, 9, 10]
        k1, k2, k3, k4, k5, k6, k7, k8 = [[] for _ in range(8)]
        comp = [k1, k2, k3, k4, k5, k6, k7, k8]

        self.feat = np.nan_to_num(self.feat)
        perf_names = ["ACC", "SEN", "SPE", "F1score", "REC", "PRE", "TPR", "FPR"]

        for w in range(len(kr)):
            print(colored(str(kr[w]) + "------Fold", color='magenta'))
            kr[w] = 2
            strtfdKFold = StratifiedKFold(n_splits=kr[w])
            kfold = strtfdKFold.split(self.feat, self.lab)

            C1, C2, C3, C4, C5, C6, C7, C8 = [[] for _ in range(8)]

            for k, (train, test) in enumerate(kfold):
                x_train, y_train, x_test, y_test = train_test_split2(self.DB, percent=60)
                (
                    KNN_metrics, CNN_metrics, CNN_Resnet_metrics,
                    SVM_metrics, DIT_metrics, HGNN_metrics,
                    WA_metrics, proposed_model_metrics
                ) = models_return_metrics(self.DB, percent=60, ok=False)

                C1.append(KNN_metrics)
                C2.append(CNN_metrics)
                C3.append(CNN_Resnet_metrics)
                C4.append(SVM_metrics)
                C5.append(DIT_metrics)
                C6.append(HGNN_metrics)
                C7.append(WA_metrics)
                C8.append(proposed_model_metrics)

            met_all = [C1, C2, C3, C4, C5, C6, C7, C8]
            for m in range(len(met_all)):
                new = []
                for n in range(len(perf_names)):
                    x = [fold[n] for fold in met_all[m]]
                    x = np.mean(x)
                    new.append(x)
                comp[m].append(new)

        files_name = [f'Analysis/KF_Analysis/{self.DB}/{name}_2.npy' for name in perf_names]
        for j in range(len(perf_names)):
            new = []
            for i in range(len(comp)):
                x = [fold[j] for fold in comp[i]]
                new.append(x)
            np.save(files_name[j], np.array(new))


    def PERF_Analysis(self):
        epoch=[0]
        Performance_Results=[]
        Training_Percentage=40
        epochs=[100,200,300,400,500]

        for i in range(6):
            cprint(f"[⚠️] Performance Analysis Count Is {i + 1} Out Of 6", 'cyan', on_color='on_grey')

            feat, labels = Load_data2("UNSW-NB15")
            balanced_feat, balanced_label = balance2("UNSW-NB15", feat, labels)
            x_train, x_test, y_train, y_test = train_test_split2(balanced_feat, balanced_label, percent=Training_Percentage)
            output=[]
            for ep in epoch:
                result = proposed_model_main(x_train, x_test, y_train, y_test,train_percent=int(Training_Percentage),DB=self.DB)
                output.append(result)

            Performance_Results.append(output)

            Training_Percentage+=10
        np.save("Analysis/Performance_Analysis/Complete_results.npy",Performance_Results)

        print("The results are saved successfully")
        cprint("[✅] Execution of Performance Analysis Completed", 'green', on_color='on_grey')











