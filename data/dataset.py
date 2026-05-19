import numpy as np
import torch
from torch.utils.data import Dataset
from .sampler import sample_patch

def create_global_coordinates(x0, y0, z0,
                              patch_size,
                              full_shape):

    ph, pw, pd = patch_size
    H, W, D = full_shape

    # Create normalized global coordinates
    x_range = np.linspace(x0, x0 + ph - 1, ph) / H
    y_range = np.linspace(y0, y0 + pw - 1, pw) / W
    z_range = np.linspace(z0, z0 + pd - 1, pd) / D

    xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')

    # Stack as 3 coordinate channels
    coords = np.stack([xx, yy, zz], axis=0)  # (3, ph, pw, pd)

    # Normalize to [-1, 1]
    coords = coords * 2.0 - 1.0

    return coords.astype(np.float32)


class CTDataset(Dataset):
    def __init__(self, image_paths, mask_paths, config):
        self.images = image_paths
        self.masks = mask_paths
        self.config = config

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        # Load full volume
        img = np.load(self.images[idx], mmap_mode='r')
        mask = np.load(self.masks[idx], mmap_mode='r')

        full_shape = img.shape  # (H, W, D)

        # Sample patch
        img_patch, mask_patch, (x0, y0, z0) = sample_patch(
            img,
            mask,
            self.config.patch_size
        )

        # Create global coordinate channels
        coord_patch = create_global_coordinates(
            x0, y0, z0,
            self.config.patch_size,
            full_shape
        )

        # Make contiguous
        img_patch = np.ascontiguousarray(img_patch).astype(np.float32)
        mask_patch = np.ascontiguousarray(mask_patch).astype(np.int64)

        # Convert to tensor
        img_patch = torch.from_numpy(img_patch).unsqueeze(0)   # (1, ph, pw, pd)
        coord_patch = torch.from_numpy(coord_patch)            # (3, ph, pw, pd)
        mask_patch = torch.from_numpy(mask_patch)

        # Concatenate image + coordinates
        img_patch = torch.cat([img_patch, coord_patch], dim=0)

        return img_patch, mask_patch
