import torch
import os
import numpy as np
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

class ABIDE(Dataset):
    def __init__(self, data_paths, domain_index):
        super(ABIDE, self).__init__()
        m = loadmat(data_paths)
        self.data = m['fea_net'][domain_index][0]
        self.label = m['label'][domain_index][0][0]

    def __getitem__(self, item):
        img = torch.from_numpy(np.expand_dims(self.data[:90, :90, item], 0))
        img = img.type(torch.FloatTensor)
        label = self.label[item] - 1

        return img, label

    def __len__(self):
        return self.data.shape[2]

def get_ABIDE_dloader(base_path, domain_index, batch_size, num_workers):
    dataset_path = os.path.join(base_path, 'ABIDE/rois_aal116.mat')
    domain_dataset = ABIDE(dataset_path, domain_index)
    domain_dloader = DataLoader(domain_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return domain_dloader