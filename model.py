import torch
import torch.nn as nn
import numpy as np


class DeepSurv(nn.Module):
    """
    Model architecture of DeepSurv
    """
    def _init_coxph_layers(self, lastdim):
        self.expert = nn.Linear(lastdim, 1, bias=False)

    def __init__(self, inputdim, layers=None, optimizer='Adam'):

        super(DeepSurv, self).__init__()

        self.optimizer = optimizer

        if layers is None:
            layers = []
        self.layers = layers

        if len(layers) == 0:
            lastdim = inputdim
        else:
            lastdim = layers[-1]

        self._init_coxph_layers(lastdim)

        # constructed neural networks with specified layers
        self.embedding = create_representation(inputdim, layers, 'ReLU6')

    def forward(self, x):
        return self.expert(self.embedding(x))


def create_representation(inputdim, layers, activation):
    r"""Helper function to generate the representation function for DSM.

    Deep Survival Machines learns a representation (\ Phi(X) \) for the input
    data. This representation is parameterized using a Non Linear Multilayer
    Perceptron (`torch.nn.Module`). This is a helper function designed to
    instantiate the representation for Deep Survival Machines.

    .. warning::
    Not designed to be used directly.

    Parameters
    ----------
    inputdim: int
      Dimensionality of the input features.
    layers: list
      A list consisting of the number of neurons in each hidden layer.
    activation: str
      Choice of activation function: One of 'ReLU6', 'ReLU' or 'SeLU'.

    Returns
    ----------
    an MLP with torch.nn.Module with the specfied structure.

    """

    if activation == 'ReLU6':
        act = nn.ReLU6()
    elif activation == 'ReLU':
        act = nn.ReLU()
    elif activation == 'SeLU':
        act = nn.SELU()

    modules = []
    prevdim = inputdim

    for hidden in layers:
        modules.append(nn.Linear(prevdim, hidden, bias=False))
        modules.append(act)
        prevdim = hidden

    return nn.Sequential(*modules)


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
