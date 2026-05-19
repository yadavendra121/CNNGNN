import os
import nibabel as nib
import numpy as np
from data.preprocessing import normalize_ct
from tqdm import tqdm

image_dir = "dataset/Abd/train/images"
mask_dir = "dataset/Abd/train/masks"

save_img_dir = "dataset/Abd/train/images_npy"
save_mask_dir = "dataset/Abd/train/masks_npy"

os.makedirs(save_img_dir, exist_ok=True)
os.makedirs(save_mask_dir, exist_ok=True)

for fname in os.listdir(image_dir):
    img = nib.load(os.path.join(image_dir, fname)).get_fdata().astype(np.float32)
    mask = nib.load(os.path.join(mask_dir, fname)).get_fdata().astype(np.int64)

    img = normalize_ct(img)

    base = fname.split(".")[0]
    np.save(os.path.join(save_img_dir, base + ".npy"), img, allow_pickle=False)
    np.save(os.path.join(save_mask_dir, base + ".npy"), mask, allow_pickle=False)
    print("image has preprocessed and converted to npy", fname)

print("Preprocessing done.")
