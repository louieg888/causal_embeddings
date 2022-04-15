import matplotlib.pyplot as plt
import pickle
import torch


from loaders.features import CausalEmbeddingsDataset
from models.autoencoder import AutoEncoder

BEST_MODEL_PATH = '../logs/flow_best.pth.tar'

class TestBasicAutoencoderUsage:
    def test_view_image_reconstruction(self):
        dataset = CausalEmbeddingsDataset()
        batch_size = 16
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # with open('../logs/preprocessing.pkl', 'rb') as handle:
        #     min_max_dict = pickle.load(handle)

        model = AutoEncoder(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 128),
            strides=(2, 2, 2),
        )

        model.load_state_dict(torch.load(BEST_MODEL_PATH)['state_dict'])

        for images, obs_dict, ids in data_loader:
            recon_images, embeddings = model(images)

            for i in range(len(ids)):
                id = ids[i].item()
                recon_image = recon_images[i]
                image = images[i]
                plt.imshow(image.detach().numpy().reshape((176,216)))
                plt.imshow(recon_image.detach().numpy().reshape((176,216)))




