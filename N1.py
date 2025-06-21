from plott import PlotAll
import numpy as np

aa= np.load('Analysis/KF_Analysis/Zea_mays/ACC_2.npy')


xt = ["6","7","8","9","10"]
p = PlotAll(save=True, legends=["KNN", "CNN", "CNN_Resnet", "SVM", "DIT", "HGNN", "WA", "PM"], show=True)
p.box_plot(aa.T, "K-Fold", "Accuracy(%)", path="Results/Zea_mays/KF_Analysis/Bar/", filename="BLEU", xticks=xt)
p.line_plot(aa.T, "K-Fold", "Accuracy(%)", path="Results/Zea_mays/KF_Analysis/Bar/", filename="BLEU", xticks=xt)

