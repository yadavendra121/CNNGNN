import torch
import torch.nn.functional as F
import numpy as np


def get_gaussian_weight(patch_size):
    center = [s // 2 for s in patch_size]
    sigma = [s / 8 for s in patch_size]

    grids = np.meshgrid(
        *[np.arange(s) for s in patch_size],
        indexing="ij"
    )

    weight = np.ones(patch_size)

    for i in range(len(patch_size)):
        weight *= np.exp(
            -((grids[i] - center[i]) ** 2)
            / (2 * sigma[i] ** 2)
        )

    return torch.tensor(weight).float()


def sliding_window_inference(
    model,
    volume,
    patch_size,
    overlap,
    num_classes,
    device,
):

    model.eval()

    stride = [
        int(p * (1 - overlap))
        for p in patch_size
    ]

    B, C, D, H, W = volume.shape

    output = torch.zeros(
        (1, num_classes, D, H, W),
        device=device
    )

    weight_map = torch.zeros_like(output)

    gaussian = get_gaussian_weight(patch_size).to(device)

    with torch.no_grad():

        for z in range(0, D - patch_size[0] + 1, stride[0]):
            for y in range(0, H - patch_size[1] + 1, stride[1]):
                for x in range(0, W - patch_size[2] + 1, stride[2]):

                    patch = volume[
                        :,
                        :,
                        z:z+patch_size[0],
                        y:y+patch_size[1],
                        x:x+patch_size[2]
                    ]

                    logits, _, _, _ = model(patch)

                    weighted = logits * gaussian

                    output[
                        :,
                        :,
                        z:z+patch_size[0],
                        y:y+patch_size[1],
                        x:x+patch_size[2]
                    ] += weighted

                    weight_map[
                        :,
                        :,
                        z:z+patch_size[0],
                        y:y+patch_size[1],
                        x:x+patch_size[2]
                    ] += gaussian

    output /= weight_map + 1e-6

    return output
