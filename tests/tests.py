from collections import OrderedDict

import matplotlib.pyplot as plt
import pickle

import numpy as np
import torch


from loaders.features import CausalEmbeddingsDataset, IMAGE_DIMENSIONS
from models.autoencoder import AutoEncoder
from models.causal_autoencoder import CausalAutoEncoder

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
                plt.imshow(image.detach().numpy().reshape(IMAGE_DIMENSIONS))
                plt.imshow(recon_image.detach().numpy().reshape(IMAGE_DIMENSIONS))


class TestDataLoaderReproducibility:
    def test_train_dataloader_reproducibility(self):
        verification_iterations = 10
        image_lst, obs_dict_lst, ids_lst = [], [], []
        first_pass = True
        random_seed = np.random.randint(1, 10001)

        for _ in range(verification_iterations + 1):
            batch_size = 16

            dataset = CausalEmbeddingsDataset(random_seed=random_seed)
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, test_size], generator=torch.Generator().manual_seed(random_seed)
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                generator=torch.Generator().manual_seed(random_seed)
            )

            for idx, sample in enumerate(train_loader):
                images, obs_dict, ids = sample

                if first_pass:
                    image_lst.append(images)
                    obs_dict_lst.append(obs_dict)
                    ids_lst.append(ids)
                else:
                    prev_images = image_lst[idx]
                    prev_obs_dict = obs_dict_lst[idx]
                    prev_ids = ids_lst[idx]

                    assert(torch.equal(prev_images, images))
                    assert(torch.equal(prev_ids, ids))

                    for ((curr_key, curr_value), (prev_key, prev_value)) in zip(obs_dict.items(), prev_obs_dict.items()):
                        assert(torch.equal(prev_value, curr_value))
                        assert(prev_key == curr_key)

            first_pass = False


    def test_test_dataloader_reproducibility(self):
        verification_iterations = 10
        image_lst, obs_dict_lst, ids_lst = [], [], []
        first_pass = True
        random_seed = np.random.randint(1, 10001)

        for _ in range(verification_iterations + 1):
            batch_size = 16

            dataset = CausalEmbeddingsDataset(random_seed=random_seed)
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, test_size], generator=torch.Generator().manual_seed(random_seed)
            )

            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                generator=torch.Generator().manual_seed(random_seed)
            )

            for idx, sample in enumerate(test_loader):
                images, obs_dict, ids = sample

                if first_pass:
                    image_lst.append(images)
                    obs_dict_lst.append(obs_dict)
                    ids_lst.append(ids)
                else:
                    prev_images = image_lst[idx]
                    prev_obs_dict = obs_dict_lst[idx]
                    prev_ids = ids_lst[idx]

                    assert(torch.equal(prev_images, images))
                    assert(torch.equal(prev_ids, ids))

                    for ((curr_key, curr_value), (prev_key, prev_value)) in zip(obs_dict.items(), prev_obs_dict.items()):
                        assert(torch.equal(prev_value, curr_value))
                        assert(prev_key == curr_key)

            first_pass = False

    def test_causal_autoencoder_instantiation_reproducibility(self):
        verification_iterations = 3
        embedding_dimension = 8
        params = None
        first_pass = True
        random_seed = np.random.randint(1, 10001)

        for _ in range(verification_iterations + 1):
            torch.manual_seed(random_seed)
            dataset = CausalEmbeddingsDataset(embedding_dimension=embedding_dimension)
            causal_autoencoder = CausalAutoEncoder(dataset.schema)

            if first_pass:
                params = OrderedDict(list(causal_autoencoder.named_parameters()))
                first_pass = False
            else:
                new_params, old_params = OrderedDict(list(causal_autoencoder.named_parameters())), params
                for ((old_param_name, old_param_tensor), (new_param_name, new_param_tensor)) in zip(old_params.items(), new_params.items()):
                    assert(old_param_name == new_param_name)
                    assert(torch.equal(old_param_tensor, new_param_tensor))

