import os
import pandas as pd
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt
import copy
import numpy as np

# Enable logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

paths = [
    "/home/lopezurl/Documents/nn_inversion/deployed/models/resnet_solo_grid_400_no/seeds",
    "/home/lopezurl/Documents/nn_inversion/deployed/models/resnet_solo_grid_400_no_preserved/seeds"
]

fig = plt.figure(figsize=(5,4), dpi=150)
axs = fig.subplots(2,2)

for j, (path, axs_) in enumerate(zip(paths, axs)):

    logs_train = []
    logs_validation = []

    for i, folder in enumerate(os.listdir(path)):

        if False:
            # Create an empty list to store the log data
            log_data = {
                "epoch_dense_1_nan_mae": [],
                "epoch_dense_1_nan_mse": [],
                "epoch_dense_3_nan_binary_crossentropy": [],
                "epoch_dense_3_nan_binary_accuracy": [],
                "epoch_loss": []
            }
        else:
            log_data = {
                "epoch_nan_mae": [],
                "epoch_nan_mse": [],
                "epoch_loss": []
            }

        log_data_train = copy.deepcopy(log_data)
        log_data_validation = copy.deepcopy(log_data)
        del log_data

        for name, log_data in zip(("train", "validation"), (log_data_train, log_data_validation)):

            log_folder = os.path.join(path, folder, "logs", name)
            log_file = os.path.join(log_folder, os.listdir(log_folder)[0])
        
            # Open the event file
            event_file_iterator = summary_iterator(log_file)

            # Loop over all the events in the event file
            for event in event_file_iterator:
                # Check if the event is a summary event
                if event.summary:
                    # Loop over all the values in the summary event
                    for value in event.summary.value:
                        t = tf.make_ndarray(value.tensor)
                        if value.tag in log_data.keys():
                            log_data[value.tag].append(t)

        # Convert the log data to a Pandas DataFrame
        log_df_train = pd.DataFrame(log_data_train)
        log_df_validation = pd.DataFrame(log_data_validation)

        logs_train.append(log_df_train)
        logs_validation.append(log_df_validation)

    if False:
        key = "epoch_dense_1_nan_mae"
    else:
        key = "epoch_nan_mae"

    for ax, logs_df, color in zip(axs_, (logs_train, logs_validation), ((60/255, 160/255, 45/255, 0.1), (244/255, 160/255, 22/255, 0.1))):

        color = np.array(color)

        if j == 1:
            color = 1 - np.array(color)

        color[-1] = .4

        percentage = 90

        train_sup = np.percentile([log_df[key] for log_df in logs_df], q=percentage, axis=0).astype(np.float16)
        train_low = np.percentile([log_df[key] for log_df in logs_df], q=100-percentage, axis=0).astype(np.float16)
        train_med = np.median([log_df[key] for log_df in logs_df], axis=0).astype(np.float16)

        ax.fill_between(np.arange(1,33), train_sup, train_low, color=np.array(color))
        color[-1] = 1
        ax.plot(np.arange(1,33), train_med, color=np.array(color) ** 2)

for axs_ in axs:
    axs_[0].set_ylim(10,30)
    axs_[1].set_ylim(10,30)

    axs_[0].set_ylabel("Training dataset MAE")
    axs_[1].set_ylabel("Validation dataset MAE")

    axs_[0].set_xticks([0, 8, 16, 24, 32])
    axs_[1].set_xticks([0, 8, 16, 24, 32])

    axs_[0].set_yticks([10, 15, 20, 25, 30])
    axs_[1].set_yticks([10, 15, 20, 25, 30])

for ax in axs.flatten():
    ax.set_xlim(1,len(log_df_validation[key]))
    ax.set_xlabel("Epoch")

fig.tight_layout()
fig.savefig("metrics_diff")
#plt.show()