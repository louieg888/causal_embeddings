import torch

from collections import OrderedDict
from torch import nn

from constants import DEVICE
from models.autoencoder import AutoEncoder
from models.models import DAG_Layer


class CausalAutoEncoder(nn.Module):
    def __init__(self, schema, embedding_dim=8):
        super().__init__()
        self.autoencoder = AutoEncoder(
            channels=(32, 64, 128),
            embedding_dimension=embedding_dim,
            in_channels=1,
            out_channels=1,
            spatial_dims=2,
            strides=(2, 2, 2),
        ).to(DEVICE)
        self.dag_layer = DAG_Layer(schema).to(DEVICE)
        self.embedding_dim = embedding_dim
        self.schema = schema

    # for an example batch with 3 samples
    # OrderedDict
    # obs_dict = OrderedDict("thickness":torch.zeros((3,4)), "intensity":torch.ones((3,1))*2)

    def forward(self, obs_dict, images):
        recon_images, im_embs = self.autoencoder(images)

        X = im_embs
        for node_name, node_obs in obs_dict.items():
            X = torch.cat([X, node_obs], axis=1)

        X = X.to(DEVICE)
        X_hat = self.dag_layer(X)

        return X, X_hat, recon_images
