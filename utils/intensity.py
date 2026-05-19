import torch


def clip_hu(x, min_hu=-1000, max_hu=1000):
    return torch.clamp(x, min_hu, max_hu)


def normalize_hu(x, min_hu=-1000, max_hu=1000):
    x = clip_hu(x, min_hu, max_hu)
    return (x - min_hu) / (max_hu - min_hu)
