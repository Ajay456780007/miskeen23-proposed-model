from Sub_Functions.Analysis import Analysis
from Sub_Functions.Plot import ALL_GRAPH_PLOT
from Sub_Functions.Read_data import Preprocessing

DB=["UNSW-NB15","N-BaIoT","CICIDS2015"]

for i in range(len(DB)):
    # Preprocessing(DB[i])

    TP=Analysis(DB[i])

    # TP.COMP_Analysis()

    TP.PERF_Analysis()

    # TP.KF_Analysis()

    # PL=ALL_GRAPH_PLOT()

    # PL.GRAPH_RESULT(DB[i])




