import torch
from torch import tensor

# image, age, sex, study

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

B_TRUE = tensor([
    [0,0,0,0],
    [1,0,0,0],
    [1,0,0,0],
    [1,0,0,0],
]).to(DEVICE)
