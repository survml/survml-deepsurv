import uuid

import torch
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))

from dataset import get_time_status
from model import DeepSurv, partial_ll_loss
# from test import *
from config import *


def train_step(model, x, t, e, optimizer, bs=256, seed=100):
    """

    :param model: initialized model to train
    :param x: tensor of features
    :param t: tensor of time
    :param e: tensor of event
    :param optimizer: optimizer used to train
    :param bs: batch size for training
    :param seed: seed for initialization
    :return:
    """
    x, t, e = shuffle(x, t, e, random_state=seed)

    n = x.shape[0]

    batches = (n // bs) + 1

    epoch_loss = 0

    for i in range(batches):
        xb = x[i * bs:(i + 1) * bs]
        tb = t[i * bs:(i + 1) * bs]
        eb = e[i * bs:(i + 1) * bs]

        # Training Step
        torch.enable_grad()
        optimizer.zero_grad()
        loss = partial_ll_loss(model(xb), tb, eb)
        loss.backward()
        optimizer.step()

        epoch_loss += float(loss)

    return epoch_loss / n


if __name__ == "__main__":
    df = pd.read_csv('{}_parsed.csv'.format(DATASET))
    time_name, status_name = get_time_status(DATASET)

    X = df.drop(columns=[status_name, time_name])
    y = df[[status_name, time_name]]

    input_dim = len(X.columns)

    # normalization for features
    X = (X - X.min()) / (X.max() - X.min())

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    t_train, e_train = y_train[time_name], y_train[status_name]
    # t_val, e_val = y_val[time_name], y_val[status_name]

    X_train, t_train, e_train = \
        torch.tensor(X_train.to_numpy()).float(), torch.tensor(t_train.to_numpy()).float(), \
        torch.tensor(e_train.to_numpy()).float()
    # X_val, t_val, e_val = \
    #     torch.tensor(X_val.to_numpy()).float(), torch.tensor(t_val.to_numpy()).float(), \
    #     torch.tensor(e_val.to_numpy()).float()

    model = DeepSurv(inputdim=input_dim, layers=[25, 25])

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # optimizer = get_optimizer(model, lr)

    train_losses, val_losses, ci_curve = [], [], []

    for epoch in range(1600):

        # train_step_start = time.time()
        train_loss = train_step(model, X_train, t_train, e_train, optimizer, BATCH_SIZE, seed=epoch)
        # print(f'Duration of train-step: {time.time() - train_step_start}')
        # test_step_start = time.time()
        # val_loss, ci = test_step(model, X_val, t_val, e_val)
        # print(f'Duration of test-step: {time.time() - test_step_start}')

        train_losses.append(train_loss)
        # val_losses.append(val_loss)
        # ci_curve.append(ci)

        if epoch % 10 == 0:
            print("epoch:\t", epoch + 1,
                  "\t train loss:\t", "%.6f" % round(train_loss, 6))

    plt.plot(train_losses)
    plt.title("Average Train Loss")
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    plt.savefig("models/dsurv/img/dsurv_train_{}_lr={}_bs={}_time={}.png".format(DATASET, LR, BATCH_SIZE, uuid.uuid1()))
    plt.show()

    torch.save(model.state_dict(),
               'models/dsurv/saved_models/dsurv_{}_lr={}_bs={}_time={}.pt'
               .format(DATASET, LR, BATCH_SIZE, uuid.uuid1()))
