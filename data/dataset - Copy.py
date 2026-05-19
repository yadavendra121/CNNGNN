import numpy as np
import torch
from torch.utils.data import Dataset
from .sampler import sample_patch

class CTDataset(Dataset):
    def __init__(self, image_paths, mask_paths, config):
        self.images = image_paths
        self.masks = mask_paths
        self.config = config

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = np.load(self.images[idx], mmap_mode='r')
        mask = np.load(self.masks[idx], mmap_mode='r')

        img, mask = sample_patch(
            img,
            mask,
            self.config.patch_size
        )

        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)

        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask)

        return img, mask

