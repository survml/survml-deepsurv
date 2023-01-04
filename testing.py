import pandas as pd
import torch
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))

from model import partial_ll_loss
from dataset import get_time_status
from model import DeepSurv
from config import *


def test_step(model, x, t, e):
    with torch.no_grad():
        loss = float(partial_ll_loss(model(x), t, e))
    valid_ci = concordance_index(t, pd.DataFrame(-model(x).detach().numpy()), e)

    return loss / x.shape[0], valid_ci


if __name__ == "__main__":
    df = pd.read_csv('data/survival-data-parsed/{}_parsed.csv'.format(DATASET))
    time_name, status_name = get_time_status(DATASET)

    X = df.drop(columns=[status_name, time_name])
    y = df[[status_name, time_name]]

    input_dim = len(X.columns)

    # normalization for features
    X = (X - X.min()) / (X.max() - X.min())

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    t_test, e_test = y_test[time_name], y_test[status_name]

    X_test, t_test, e_test = \
        torch.tensor(X_test.to_numpy()).float(), torch.tensor(t_test.to_numpy()).float(), \
        torch.tensor(e_test.to_numpy()).float()

    model = DeepSurv(inputdim=input_dim, layers=[25, 25])
    model.load_state_dict(
        torch.load('models/dsurv/saved_models/'
                   'dsurv_metabric_lr=1e-05_bs=32_time=74619dfe-4de7-11ed-b9fe-acde48001122.pt'))
    model.eval()

    _, ci = test_step(model, X_test, t_test, e_test)
    print("Concordance Index:", ci)
