import cv2
import functools
import numpy as np
import os
import pandas as pd
import pickle
import torch

PATH_TO_TABULAR = 'datasets/participants.tsv'
PATH_TO_IMAGES = 'datasets/gstore/scratch/u/iriondoc/BHB_data/midaxial_2D'

# each sample is a list consisting of [obs_data_dict, image]
class CausalEmbeddingsDataset(torch.utils.data.Dataset):
    VARIABLE_TYPES = {
        "categorical": [
            'study',
            'sex'
        ],
        "continuous": [
            'age',
            'wmv',
            'gmv',
            'csfv',
            'tiv'
        ]
    }

    param_groups = {
        'roi': ['wmv','gmv','csfv','tiv'],
        'age': ['age'],
        'sex': ['female', 'male'],
        'study': ['study_' + str(val) for val in [1,2,3,4,6,8,9,10]],
    }

    def __init__(self):
        tab_path = PATH_TO_TABULAR
        participants = pd.read_csv(tab_path, sep='\t')

        # filter stages
        participants = participants[(participants.acquisition_setting==1.0)&(participants.magnetic_field_strength==3.0)]
        participants = participants[['participant_id', 'study', 'sex', 'age', 'wmv', 'gmv', 'csfv', 'tiv']]

        # preprocessing stages
        participants['study'] = 'study_' + participants['study'].astype(str)

        # one hot
        dfs = []
        for col_name in CausalEmbeddingsDataset.VARIABLE_TYPES["categorical"]:
            df = pd.get_dummies(participants[col_name])
            dfs.append(df)

        self.column_range = {}
        for col_name in CausalEmbeddingsDataset.VARIABLE_TYPES["continuous"]:
            df = participants[col_name]
            _min, _max = df.min(), df.max()
            df = (df - _min) / (_max - _min)
            self.column_range[col_name] = (_min, max)
            dfs.append(df)

        dfs.append(participants['participant_id'])
        self.tabular_df = pd.concat(dfs, axis=1)
        self.column_range['image'] = self.get_image_normalization_coefficients()

        with open("logs/preprocessing.pkl", "wb") as handle:
            pickle.dump(self.column_range, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.schema = {key: len(value) for key, value in self.param_groups.items()}

    def get_image_normalization_coefficients(self):
        min_val, max_val = torch.inf, -torch.inf

        for id in self.tabular_df['participant_id']:
            img = self.get_original_image(id)
            min_val = min(min_val, img.min())
            max_val = max(max_val, img.max())

        return (min_val, max_val)

    def get_original_image(self, id):
        image_name = "sub-" + str(id) + '_preproc-quasiraw_T1w.npy'
        image_path = os.path.join(PATH_TO_IMAGES, image_name)
        image = np.load(image_path)
        image = cv2.resize(image, (32,32))
        return image

    @functools.lru_cache(maxsize=250, typed=False)
    def get_transformed_image(self, idx):
        image_name = "sub-" + str(int(self.tabular_df.iloc[idx]['participant_id'])) + '_preproc-quasiraw_T1w.npy'
        image_path = os.path.join(PATH_TO_IMAGES, image_name)
        min_val, max_val = self.column_range['image'][0], self.column_range['image'][1]
        _range = max_val - min_val
        image = np.load(image_path)
        image = cv2.resize(image, (32,32))
        transformed_image = (image - min_val) / _range

        return transformed_image.reshape((1, *transformed_image.shape))

    def __len__(self):
        return len(self.tabular_df)

    def __getitem__(self, idx):
        image = self.get_transformed_image(idx)
        tab_values = self.tabular_df.drop(columns=['participant_id']).iloc[idx]

        final_tab_values = {}
        for name, group in self.param_groups.items():
            consolidated_vec = torch.tensor([tab_values[column] for column in group]).float()
            final_tab_values[name] = consolidated_vec

        return torch.tensor(image, dtype=torch.float), final_tab_values


if __name__ == '__main__':
    dataset = CausalEmbeddingsDataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    for imgs, labels in data_loader:
        print("Batch of images has shape: ", imgs.shape)
        for label, mat in labels.items():
            print(label)
            print(mat)
            print(mat.shape)
            print()

        print()



