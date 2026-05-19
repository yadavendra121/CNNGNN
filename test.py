import os
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm

from configs.config import Config
from models.network import FullModel
from inference.sliding_window import sliding_window_inference
from inference.tta import tta_inference
from inference.postprocessing import keep_largest_component_per_class


def main():

    config = Config()
    device = torch.device(config.device)

    model = FullModel(config).to(device)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()

    test_images = sorted(os.listdir("dataset/Abd/test/images"))
    test_images = [os.path.join("datast/Abd/test/images", f) for f in test_images]
    predictions = "pre_dataset/Abd"
    os.makedirs(predictions, exist_ok=True)

    for img_path in tqdm(test_images):

        nii = nib.load(img_path)
        volume = nii.get_fdata().astype("float32")
        affine = nii.affine

        volume = (volume - volume.mean()) / (volume.std() + 1e-6)

        volume = torch.tensor(volume).unsqueeze(0).unsqueeze(0).to(device)

        # ===============================
        # Sliding Window
        # ===============================
        logits = sliding_window_inference(
            model,
            volume,
            config.patch_size,
            config.overlap,
            config.num_classes,
            device
        )

        # ===============================
        # TTA
        # ===============================
        logits_tta = tta_inference(model, volume)

        # ===============================
        # Ensemble
        # ===============================
        logits = (logits + logits_tta) / 2.0

        probs = torch.softmax(logits, dim=1)

        pred = torch.argmax(probs, dim=1).squeeze().cpu().numpy()

        # ===============================
        # Postprocessing
        # ===============================
        pred = keep_largest_component_per_class(
            pred,
            config.num_classes
        )

        # ===============================
        # Save
        # ===============================
        output_nii = nib.Nifti1Image(pred.astype(np.uint8), affine)

        name = os.path.basename(img_path)
        nib.save(output_nii, f"{predictions}/{name}")
        print(name, "testing completed")

    print("Inference Complete!")


if __name__ == "__main__":
    main()
