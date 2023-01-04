import torch
import numpy as np

SHAPE_SCALE = 1
SCALE_SCALE = 100
# LR = 0.001
# BATCH_SIZE = 32
# EPOCHS = 30
DATASET = 'support'
LR = 1e-05
EPOCHS = 2000
BATCH_SIZE = 32

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
