import uuid
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))

from dataset import get_time_status
from model import DeepSurv
from train import *
from testing import *
from config import *


if __name__ == "__main__":
    df = pd.read_csv('{}_parsed.csv'.format(DATASET))
    time_name, status_name = get_time_status(DATASET)

    X = df.drop(columns=[status_name, time_name])
    y = df[[status_name, time_name]]

    input_dim = len(X.columns)

    # normalization for features
    X = (X - X.min()) / (X.max() - X.min())

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    t_train, e_train = y_train[time_name], y_train[status_name]
    t_val, e_val = y_val[time_name], y_val[status_name]

    X_train, t_train, e_train = \
        torch.tensor(X_train.to_numpy()).float(), torch.tensor(t_train.to_numpy()).float(), \
        torch.tensor(e_train.to_numpy()).float()
    X_val, t_val, e_val = \
        torch.tensor(X_val.to_numpy()).float(), torch.tensor(t_val.to_numpy()).float(), \
        torch.tensor(e_val.to_numpy()).float()

    model = DeepSurv(inputdim=input_dim, layers=[25, 25])

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # optimizer = get_optimizer(model, lr)

    train_losses, val_losses, ci_curve = [], [], []

    for epoch in range(EPOCHS):

        # train_step_start = time.time()
        train_loss = train_step(model, X_train, t_train, e_train, optimizer, BATCH_SIZE, seed=epoch)
        # print(f'Duration of train-step: {time.time() - train_step_start}')
        # test_step_start = time.time()
        val_loss, ci = test_step(model, X_val, t_val, e_val)
        # print(f'Duration of test-step: {time.time() - test_step_start}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        ci_curve.append(ci)

        if epoch % 10 == 0:
            print("epoch:\t", epoch + 1,
                  "\t train loss:\t", "%.6f" % round(train_loss, 6),
                  "\t valid loss:\t", "%.6f" % round(val_loss, 6),
                  "\t concordance index:\t", "%.4f" % round(ci, 4))

    fig, axs = plt.subplots(3, figsize=[15, 20])
    axs[0].plot(train_losses), axs[0].set_title("Average Train Loss")
    axs[1].plot(val_losses), axs[1].set_title("Average Validation Loss")
    axs[2].plot(ci_curve), axs[2].set_title("Concordance Index")
    fig.savefig("models/dsurv/img/dsurv_valid_{}_lr={}_bs={}_time={}.png".format(DATASET, LR, BATCH_SIZE, uuid.uuid1()))
    plt.show()
