import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

class Schizophrenia(Dataset):
    def __init__(self, data_paths, domain_name):
        super(Schizophrenia, self).__init__()
        self.data = np.load(data_paths + '/' + domain_name + '_cov.npy')
        self.label = np.load(data_paths + '/' + domain_name + '_labels.npy')


    def __getitem__(self, item):
        img = torch.from_numpy(np.expand_dims(self.data[item][0, :, :], 0))
        img = img.type(torch.FloatTensor)
        label = self.label[item]

        return img, label

    def __len__(self):
        return len(self.data)

def get_schizophrenia_dloader(base_path, domain_name, batch_size, num_workers):
    dataset_path = os.path.join(base_path, 'schizophrenia')
    domain_dataset = Schizophrenia(dataset_path, domain_name)
    data = np.load(dataset_path + '/' + domain_name + '_cov.npy')
    domain_dloader = DataLoader(domain_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return domain_dloader