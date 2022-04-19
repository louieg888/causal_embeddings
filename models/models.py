import functools
from collections import OrderedDict

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import torch

from scipy.linalg import block_diag
from scipy.special import expit as sigmoid
import torch.nn as nn
from torchsummary import summary

from constants import DEVICE
from loaders.features import CausalEmbeddingsDataset



class DAG_Layer(nn.Module):
    def __init__(self, schema):
        super().__init__()
        self.schema = schema
        self.mask = self._mask()
        self.w_est_len = self._get_w_est_len()
        self.d = self.mask.shape[0]
        self.w_est = torch.nn.parameter.Parameter(data=torch.normal(0, 0.1, size=(self.w_est_len,)), requires_grad=True)

    def _mask(self):
        """
        Mask the block diagonal for an expanded W matrix with zeros, removes the possibility for self loops
        """
        block_matrices = [np.ones((dim, dim)) for dim in list(self.schema.values())]
        f_mask = block_diag(*block_matrices)
        return torch.tensor(1 - f_mask).to(DEVICE)

    def _get_w_est_len(self):
        """
        Number of non-zero variable entries in W after masking the block diagonal
        """
        return int(torch.sum(self.mask.to(DEVICE)).item())

#     @functools.lru_cache(maxsize=100, typed=False)
    def reconstruct_W(self, w):
        """
        Expand w to block matrix W
        """
        W = torch.zeros(self.d, self.d, dtype=torch.float64)
        nonzero_locations = torch.nonzero(self.mask)
        for ind, tup in enumerate([tuple(val) for val in nonzero_locations]):
            W[tup] = w[ind]
        return W.to(DEVICE)

    def forward(self, X):
        # X –> w_est, W_true
        X = X.to(DEVICE)
        W = self.reconstruct_W(self.w_est)
        return torch.matmul(X, W)


if __name__ == "__main__":
    pass
#     dataset = CausalEmbeddingsDataset()
#     conv_ae = ConvolutionalAE(dataset.schema)
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

#     for images, obs_dict, _ in data_loader:
#         res = conv_ae(obs_dict, images)


