import numpy as np

def sample_patch(img, mask, patch_size):

    H, W, D = img.shape
    ph, pw, pd = patch_size

    assert H >= ph and W >= pw and D >= pd, "Patch bigger than volume!"

    x = np.random.randint(0, H - ph + 1)
    y = np.random.randint(0, W - pw + 1)
    z = np.random.randint(0, D - pd + 1)

    img_patch = img[x:x+ph, y:y+pw, z:z+pd]
    mask_patch = mask[x:x+ph, y:y+pw, z:z+pd]

    return img_patch, mask_patch