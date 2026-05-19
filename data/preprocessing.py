import numpy as np

def normalize_ct(img):
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip(img, p1, p99)
    img = (img - img.mean()) / (img.std() + 1e-6)
    return img
