import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

from configs.config import Config
from data.dataset import CTDataset
from models.network import FullModel
from losses.total_loss import TotalLoss
from training.trainer import Trainer
from training.scheduler import CosineWarmupScheduler


# ======================================================
# 🚀 RTX A5000 PERFORMANCE SETTINGS (VERY IMPORTANT)
# ======================================================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# ===========================
# Reproducibility
# ===========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===========================
# Main
# ===========================
def main():

    config = Config()
    set_seed(42)

    device = torch.device(config.device)

    # ===========================
    # Dataset paths
    # ===========================
    image_dir = "dataset/Abd/train/images_npy"
    mask_dir = "dataset/Abd/train/masks_npy"

    all_images = sorted(os.listdir(image_dir))
    all_masks = sorted(os.listdir(mask_dir))

    all_images = [os.path.join(image_dir, f) for f in all_images]
    all_masks = [os.path.join(mask_dir, f) for f in all_masks]

    assert len(all_images) == len(all_masks)

    train_images, val_images, train_masks, val_masks = train_test_split(
        all_images,
        all_masks,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    print(f"Total samples: {len(all_images)}")
    print(f"Train: {len(train_images)} | Val: {len(val_images)}")

    # ===========================
    # Dataset
    # ===========================
    train_dataset = CTDataset(train_images, train_masks, config)
    val_dataset = CTDataset(val_images, val_masks, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,  # 2 (good for 24GB)
        shuffle=True,
        num_workers=24,                 # Try 8–12 if CPU allows
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=20,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # ===========================
    # Model
    # ===========================
    model = FullModel(config).to(device)

    # 🔥 PyTorch 2.x Compiler
    #model = torch.compile(model, mode="max-autotune")
    model = torch.compile(model, mode="reduce-overhead")

    # ===========================
    # Loss
    # ===========================
    criterion = TotalLoss(config.num_classes)

    # ===========================
    # Optimizer (Good for 3D)
    # ===========================
    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
        fused=True if torch.cuda.is_available() else False,  # Faster on Ampere
    )

    # ===========================
    # Scheduler
    # ===========================
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=10,   # 20 is slow for convergence
        max_epochs=config.epochs,
    )

    # ===========================
    # Trainer (NO EMA)
    # ===========================
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        ema=None,   # 🚀 EMA removed
        config=config,
    )

    # ===========================
    # Train
    # ===========================
    trainer.fit()

    print("Training Complete!")


if __name__ == "__main__":
    main()
