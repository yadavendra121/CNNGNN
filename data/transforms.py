import torch
import torch.nn.functional as F
import random
import numpy as np


# ============================================================
# 1️⃣ Random Flip
# ============================================================

def random_flip(x, y, p=0.5):
    # x: [C, D, H, W]
    # y: [D, H, W]
    if random.random() < p:
        x = torch.flip(x, dims=[2])  # flip H
        y = torch.flip(y, dims=[1])
    if random.random() < p:
        x = torch.flip(x, dims=[3])  # flip W
        y = torch.flip(y, dims=[2])
    return x, y


# ============================================================
# 2️⃣ Random 90° Rotation
# ============================================================

def random_rotate(x, y):
    k = random.randint(0, 3)
    x = torch.rot90(x, k, dims=[2, 3])
    y = torch.rot90(y, k, dims=[1, 2])
    return x, y


# ============================================================
# 3️⃣ Gamma Augmentation (CT Contrast Shift)
# ============================================================

def random_gamma(x, p=0.3, gamma_range=(0.7, 1.5)):
    if random.random() < p:
        gamma = random.uniform(*gamma_range)

        # Normalize to [0,1]
        min_val = x.min()
        max_val = x.max()
        x_norm = (x - min_val) / (max_val - min_val + 1e-6)

        x_gamma = x_norm ** gamma
        x = x_gamma * (max_val - min_val) + min_val

    return x


# ============================================================
# 4️⃣ Gaussian Noise (CRITICAL for Low-Dose CT)
# ============================================================

def random_gaussian_noise(x, p=0.5, std_range=(0.0, 0.05)):
    if random.random() < p:
        std = random.uniform(*std_range)
        noise = torch.randn_like(x) * std
        x = x + noise
    return x


# ============================================================
# 5️⃣ Elastic Deformation (3D)
# ============================================================

def elastic_deformation(x, y, p=0.3, alpha=15, sigma=3):

    if random.random() > p:
        return x, y

    device = x.device

    C, D, H, W = x.shape

    # Random displacement fields
    dx = torch.randn(1, 1, D, H, W, device=device)
    dy = torch.randn(1, 1, D, H, W, device=device)
    dz = torch.randn(1, 1, D, H, W, device=device)

    dx = gaussian_blur_3d(dx, sigma) * alpha
    dy = gaussian_blur_3d(dy, sigma) * alpha
    dz = gaussian_blur_3d(dz, sigma) * alpha

    # Create grid
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, D, device=device),
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )

    grid = torch.stack((xx, yy, zz), dim=-1)
    grid = grid.unsqueeze(0)

    grid[..., 0] += dx.squeeze(0).squeeze(0) / W
    grid[..., 1] += dy.squeeze(0).squeeze(0) / H
    grid[..., 2] += dz.squeeze(0).squeeze(0) / D

    # Deform image
    x = F.grid_sample(
        x.unsqueeze(0),
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(0)

    # Deform mask
    y = F.grid_sample(
        y.unsqueeze(0).unsqueeze(0).float(),
        grid,
        mode='nearest',
        padding_mode='border',
        align_corners=True
    ).squeeze(0).squeeze(0).long()

    return x, y


# ============================================================
# 6️⃣ Gaussian Blur Helper
# ============================================================

def gaussian_blur_3d(x, sigma):

    kernel_size = int(2 * round(3 * sigma) + 1)

    coords = torch.arange(kernel_size) - kernel_size // 2
    coords = coords.float()
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    g = g.to(x.device)

    x = F.conv3d(x, g.view(1, 1, -1, 1, 1), padding=(kernel_size // 2, 0, 0))
    x = F.conv3d(x, g.view(1, 1, 1, -1, 1), padding=(0, kernel_size // 2, 0))
    x = F.conv3d(x, g.view(1, 1, 1, 1, -1), padding=(0, 0, kernel_size // 2))

    return x


# ============================================================
# 7️⃣ Full Augmentation Pipeline
# ============================================================

def apply_augmentations(x, y):

    x, y = random_flip(x, y)
    x, y = random_rotate(x, y)

    x = random_gamma(x)
    x = random_gaussian_noise(x)

    x, y = elastic_deformation(x, y)

    return x, y
