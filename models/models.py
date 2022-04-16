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

from loaders.features import CausalEmbeddingsDataset


class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True),
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128), nn.ReLU(True), nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

class DAG_Layer(nn.Module):
    def __init__(self, schema, max_iter=100, h_tol=1e-8, rho_max=1e16, w_threshold=0.3):
        super().__init__()
        self.schema = schema
        self.mask = self._mask()
        self.w_est_len = self._get_w_est_len()
        self.d = self.mask.shape[0]
        self.w_est = torch.normal(0, 0.1, size=(self.w_est_len,))
        #self.w_est = torch.zeros(self.w_est_len)
        self.w_est.requires_grad = True

    def _mask(self):
        """
        Mask the block diagonal for an expanded W matrix with zeros, removes the possibility for self loops
        """
        block_matrices = [np.ones((dim, dim)) for dim in list(self.schema.values())]
        f_mask = block_diag(*block_matrices)
        return torch.tensor(1 - f_mask)

    def _get_w_est_len(self):
        """
        Number of non-zero variable entries in W after masking the block diagonal
        """
        return int(torch.sum(self.mask).item())

    @functools.lru_cache(maxsize=100, typed=False)
    def reconstruct_W(self, w):
        """
        Expand w to block matrix W
        """
        W = torch.zeros(self.d, self.d, dtype=torch.float64)
        nonzero_locations = torch.nonzero(self.mask)
        for ind, tup in enumerate([tuple(val) for val in nonzero_locations]):
            W[tup] = w[ind]
        return W

    def forward(self, X):
        # X –> w_est, W_true
        W = self.reconstruct_W(self.w_est)
        return torch.matmul(X, W)


# enc = Encoder(4)
# dec = Decoder(4)
# # summary(enc, input_size=(1, 28, 28))
# # enc(torch.ones(2,1,28,28))

"""
data path: 
    encoded_images = self.enc(ims)
    X = torch.cat([encoded_images, tabular_data])
    X_hat = self.DAG_Layer(X)
    recon_images = self.dec(encoded_images

    return X_hat, recon_images
"""


class ConvolutionalAE(nn.Module):
    def __init__(self, schema, embedding_dim=4):
        super().__init__()
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(embedding_dim)
        self.dag_layer = DAG_Layer(schema)
        self.embedding_dim = embedding_dim

    @staticmethod
    def infer_data_schema(obs_dict):
        # keys in obs_dict are the node_names
        # non batch dimension of the values are the node_dims
        # return an ordereddict node_names, node_values
        return OrderedDict(
            (node_name, obs_value.shape[1]) for node_name, obs_value in obs_dict.items()
        )

    # for an example batch with 3 samples
    # OrderedDict
    # obs_dict = OrderedDict("thickness":torch.zeros((3,4)), "intensity":torch.ones((3,1))*2)

    def forward(self, obs_dict, images):
        im_embs = self.encoder(images)
        pred_ims = self.decoder(im_embs)

        X = im_embs
        for node_name, node_obs in obs_dict.items():
            X = torch.cat([X, torch.Tensor(node_obs)], axis=1)

        X_hat = self.dag_layer(X)

        return X, X_hat, pred_ims


if __name__ == "__main__":
    dataset = CausalEmbeddingsDataset()
    conv_ae = ConvolutionalAE(dataset.schema)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    for images, obs_dict, _ in data_loader:
        res = conv_ae(obs_dict, images)


