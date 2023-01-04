import random
import uuid

import numpy as np
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
from testing import test_step
from config import DATASET, LR, BATCH_SIZE, TRAIN_EPOCHS


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


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # # When running on the CuDNN backend, two further options must be set
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print("Random seed set as {}".format(seed))


if __name__ == "__main__":
    set_seed()

    df = pd.read_csv('data/{}_parsed.csv'.format(DATASET))
    time_name, status_name = get_time_status(DATASET)

    X = df.drop(columns=[status_name, time_name])
    y = df[[status_name, time_name]]

    input_dim = len(X.columns)

    # normalization for features
    X = (X - X.min()) / (X.max() - X.min())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    t_train, e_train = y_train[time_name], y_train[status_name]
    t_test, e_test = y_test[time_name], y_test[status_name]

    X_train, t_train, e_train = \
        torch.tensor(X_train.to_numpy()).float(), torch.tensor(t_train.to_numpy()).float(), \
        torch.tensor(e_train.to_numpy()).float()

    X_test, t_test, e_test = \
        torch.tensor(X_test.to_numpy()).float(), torch.tensor(t_test.to_numpy()).float(), \
        torch.tensor(e_test.to_numpy()).float()

    model = DeepSurv(inputdim=input_dim, layers=[25, 25])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses, ci_curve = [], [], []

    for epoch in range(TRAIN_EPOCHS):
        train_loss = train_step(model, X_train, t_train, e_train, optimizer, BATCH_SIZE, seed=epoch)
        train_losses.append(train_loss)

        if epoch % 10 == 0:
            print("epoch:\t", epoch,
                  "\t train loss:\t", "%.4f" % round(train_loss, 4))

    plt.plot(train_losses)
    plt.title("Average Train Loss")
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    plt.savefig("img/train_{}_lr={}_bs={}_epoch={}_time={}.png"
                .format(DATASET, LR, BATCH_SIZE, TRAIN_EPOCHS, uuid.uuid1()))
    # plt.show()

    # save the trained model
    model_name = 'DeepSurv_{}_lr={}_bs={}_time={}.pt'.format(DATASET, LR, BATCH_SIZE, uuid.uuid1())
    torch.save(model.state_dict(), 'saved_models/{}'.format(model_name))

    # test the saved model tight away
    test_model = DeepSurv(inputdim=input_dim, layers=[25, 25])
    test_model.load_state_dict(
        torch.load('saved_models/{}'.format(model_name)))
    test_model.eval()

    _, ci = test_step(test_model, X_test, t_test, e_test)
    print("Concordance Index:", round(ci, 4))
