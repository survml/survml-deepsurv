import uuid

from matplotlib import pyplot as plt

from models.dsurv._temp.dcph_torch import DeepCoxPHTorch
# from sksurv.linear_model.coxph import BreslowEstimator
# from models.dsurv.dcph_utilities import *
import torch
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append(os.path.abspath(os.getcwd()))

from models.utils.dataset import get_time_status

SHAPE_SCALE = 1
SCALE_SCALE = 100
LR = 0.001
BATCH_SIZE = 32
EPOCHS = 30
DATASET = 'metabric'


def get_optimizer(model, lr):
    """

    :param model:
    :param lr:
    :return:
    """
    if model.optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif model.optimizer == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif model.optimizer == 'RMSProp':
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise NotImplementedError('Optimizer ' + model.optimizer + ' is not implemented')


def partial_ll_loss(lrisks, tb, eb, eps=1e-3):
    tb = tb + eps * np.random.random(len(tb))
    sindex = np.argsort(-tb)

    tb = tb[sindex]
    eb = eb[sindex]

    lrisks = lrisks[sindex]
    lrisksdenom = torch.logcumsumexp(lrisks, dim=0)

    plls = lrisks - lrisksdenom
    pll = plls[eb == 1]

    pll = torch.sum(pll)

    return -pll


def _reshape_tensor_with_nans(data):
    """Helper function to unroll padded RNN inputs."""
    data = data.reshape(-1)
    return data[~torch.isnan(data)]


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
        loss = partial_ll_loss(model(xb), _reshape_tensor_with_nans(tb), _reshape_tensor_with_nans(eb))
        loss.backward()
        optimizer.step()

        epoch_loss += float(loss)

    return epoch_loss / n


def test_step(model, x, t, e):
    with torch.no_grad():
        loss = float(partial_ll_loss(model(x), t, e))

    return loss / x.shape[0]


random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

df = pd.read_csv('data/survival-data-parsed/{}_parsed.csv'.format(DATASET))
time_name, status_name = get_time_status(DATASET)

X = df.drop(columns=[status_name, time_name])
y = df[[status_name, time_name]]

# normalization for features
X = (X - X.min()) / (X.max() - X.min())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

t_train = y_train[time_name]
e_train = y_train[status_name]
t_val = y_val[time_name]
e_val = y_val[status_name]

X_train = X_train.to_numpy()
# y_train = y_train.to_numpy()
t_train = t_train.to_numpy()
e_train = e_train.to_numpy()
X_val = X_val.to_numpy()
# y_val = y_val.to_numpy()
t_val = t_val.to_numpy()
e_val = e_val.to_numpy()

input_dim, output_dim = len(X.columns), 1

# model = DeepWeiSurv(input_dim, output_dim)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])

# if val_data is None:
#     val_data = train_data

# xt, tt, et = train_data
xt, tt, et = X_train, t_train, e_train
# xv, tv, ev = val_data
xv, tv, ev = X_val, t_val, e_val

xt, tt, et = torch.tensor(xt).float(), torch.tensor(tt).float(), torch.tensor(et).float()
xv, tv, ev = torch.tensor(xv).float(), torch.tensor(tv).float(), torch.tensor(ev).float()

# tt_ = _reshape_tensor_with_nans(tt)
# et_ = _reshape_tensor_with_nans(et)
# tv_ = _reshape_tensor_with_nans(tv)
# ev_ = _reshape_tensor_with_nans(ev)

# 'L2_reg': 10.0,
# 'batch_norm': True,
# 'dropout': 0.4,
# 'hidden_layers_sizes': [25, 25],
# 'learning_rate': 1e-05,
# 'lr_decay': 0.001,
# 'momentum': 0.9,
# 'n_in': train_data['x'].shape[1],
# 'standardize': True

lr = 1e-05
epochs = 2000
bs = 32

model = DeepCoxPHTorch(inputdim=input_dim, layers=[25, 25])

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = get_optimizer(model, lr)

# valc = np.inf
# patience_ = 0
# breslow_spline = None

train_losses, val_losses = [], []

for epoch in range(epochs):

    # train_step_start = time.time()
    train_loss = train_step(model, xt, tt, et, optimizer, bs, seed=epoch)
    # print(f'Duration of train-step: {time.time() - train_step_start}')
    # test_step_start = time.time()
    val_loss = test_step(model, xv, tv, ev)
    # print(f'Duration of test-step: {time.time() - test_step_start}')

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print("epoch:\t", epoch + 1,
          "\t train loss:\t", "%.6f" % round(train_loss, 6),
          "\t valid loss:\t", "%.6f" % round(val_loss, 6))

    # if epoch % 1 == 0:
    #     if debug: print(patience_, epoch, valcn)

    # if valcn > valc:
    #     patience_ += 1
    # else:
    #     patience_ = 0
    # if patience_ == patience:
    #     breslow_spline = fit_breslow(model, xt, tt_, et_)

    # if return_losses:
    #     return (model, breslow_spline), losses
    # else:
    #     return (model, breslow_spline)

    # valc = val_loss

# breslow_spline = fit_breslow(model, xt, tt_, et_)

# if return_losses:
#     return (model, breslow_spline), losses
# else:
#     return model, breslow_spline

fig, axs = plt.subplots(2, figsize=[15, 20])
axs[0].plot(train_losses), axs[0].set_title("Average Train Loss")
axs[1].plot(val_losses), axs[1].set_title("Average Validation Loss")
fig.savefig("tests/img/dsurv_old/dsurv_{}_lr{}_{}.png".format(DATASET, LR, uuid.uuid1()))
plt.show()
