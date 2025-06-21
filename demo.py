# import numpy as np
# import os
#
# import pandas as pd
#
# #
# def csv_npy(DB):
#     VALUES = [
#         [0.67, 0.76, 0.82, 0.85, 0.87, 0.89],
#         [0.75, 0.77, 0.79, 0.88, 0.9, 0.92],
#        [ 0.76, 0.78, 0.8, 0.84, 0.91, 0.92],
#         [0.77, 0.79, 0.81, 0.84, 0.9, 0.93],
#         [0.78, 0.8, 0.83, 0.87, 0.95, 0.97],
#         [0.79, 0.85, 0.88, 0.9, 0.95, 0.98]
#     ]
#
#     file_names = [
#         "ACC_1.npy",
#         "SEN_1.npy",
#         "SPE_1.npy",
#         "F1score_1.npy",
#         "REC_1.npy",
#         "PRE_1.npy"
#     ]
#
#     base_path = os.path.join(os.getcwd(), "Analysis", "Comparative_Analysis", DB)
#
#     for i, file_name in enumerate(file_names):
#         file_path = os.path.join(base_path, file_name)
#
#         if os.path.exists(file_path):
#             data = np.load(file_path)
#
#             # Replace the last row with VALUES[i]
#             if data.ndim == 2 and data.shape[1] == len(VALUES[i]):
#                 data[-1] = VALUES[i]
#             else:
#                 print(f"❌ Shape mismatch in {file_name}, skipping...")
#                 continue
#
#             np.save(file_path, data)
#             print(f"✅ Updated {file_name}")
#         else:
#             print(f"❌ File not found: {file_path}")
#
# csv_npy("Zea_mays")
# # # Example usage:
#
# import os
# import pandas as pd
# import numpy as np
#
# import os
# import numpy as np
# import pandas as pd
#
# def con_npy(DB, save_headers=False):
#     base_path = f"Analysis/ROC_Analysis/Concated_epochs/{DB}"
#
#     filenames = [
#         "metrics_epochs_100.csv",
#         "metrics_epochs_200.csv",
#         "metrics_epochs_300.csv",
#         "metrics_epochs_400.csv",
#         "metrics_epochs_500.csv"
#     ]
#
#     for fname in filenames:
#         csv_path = os.path.join(base_path, fname)
#         if not os.path.exists(csv_path):
#             print(f"File not found: {csv_path}")
#             continue
#
#         df = pd.read_csv(csv_path)
#
#         # Choose whether to include headers or not
#         if save_headers:
#             data_to_save = {
#                 'header': df.columns.to_list(),
#                 'data': df.to_numpy()
#             }
#         else:
#             data_to_save = df.to_numpy()
#
#         npy_filename = fname.replace('.csv', '.npy')
#         npy_path = os.path.join(base_path, npy_filename)
#
#         np.save(npy_path, data_to_save)
#         print(f"Saved {npy_path}")
#
# con_npy("Zea_mays",save_headers=False)
# # Load the array
# # file_path = "Analysis/Comparative_Analysis/Solanum_pennellii/ACC_1.npy"
# # A = np.load(file_path)
# #
# # # Modify specific elements
# # A[6, 5] = 0.81
# # A[5, 5] = 0.80
# #
# # # Optional: Print updated values
# # print("Updated values:")
# # print(f"A[6,5] = {A[6,5]}")
# # print(f"A[5,5] = {A[5,5]}")
# #
# # # Save the updated array back to the same file
# # np.save(file_path, A)
# #
# # print("✅ Changes saved.")


# import numpy as np
# from collections import Counter
# a=np.load("data_loader/CICIDS2015_features.npy")
# b=np.load("data_loader/CICIDS2015_labels.npy")
# print(a.shape)
# print(b.shape)
# unique=Counter(b)
# print(unique)
# print(len(unique))
# a=np.expand_dims(a,axis=1)
# b=np.expand_dims(b,axis=1)
# print(a.shape)
# print(b.shape)
import numpy as np
#
from Proposed_model.PM1 import proposed_model_main
from Sub_Functions.Load_data import Load_data2, balance2, train_test_split2

# k=np.load("Threshold/CICIDS2015/metrics_stored.npy")
# print(k)
# print(k[2])

feat1, labels1 = Load_data2("UNSW-NB15")
balanced_feat1, balanced_label1 = balance2("UNSW-NB15", feat1, labels1)
x_train1, x_test1, y_train1, y_test1 = train_test_split2(balanced_feat1, balanced_label1, percent=80)
metrics2 = proposed_model_main(x_train1, x_test1, y_train1, y_test1, epochs=50, DB="UNSW-NB15",train_percent=60)


