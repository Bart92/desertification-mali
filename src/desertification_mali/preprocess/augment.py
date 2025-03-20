import os
import numpy as np
import rasterio
import re
from desertification_mali.preprocess.io import save_image_as_jp2

def augment_patches(input_dir: str) -> None:
    """
    Augments 512x512 patches with horizontal and vertical flipping, as well as 90°, 180°, and 270° rotations.

    Parameters:
    - input_dir (str): Path to the directory containing the original patches.
    - output_dir (str): Path to the directory where augmented patches will be saved.

    Returns:
    - None
    """
    ndvi_pattern = re.compile(r".*_NDVI\.jp2$")

    for patch_name in os.listdir(os.path.join(input_dir, "manual_labeling")):
        patch_path = os.path.join(input_dir, "manual_labeling", patch_name)

        if not os.path.isdir(patch_path):
            continue

        ndvi_files = [f for f in os.listdir(patch_path) if ndvi_pattern.match(f)]
        if not ndvi_files:
            print(f"NDVI file not found for patch: {patch_name}")
            continue
            
        for ndvi_file in ndvi_files:
            ndvi_file = os.path.join(patch_path, ndvi_file)

            with rasterio.open(ndvi_file) as src:
                ndvi = src.read(1)
                transform = src.transform
                crs = src.crs

            augmentations = {
                "original": ndvi,
                "flip_horizontal": np.fliplr(ndvi),
                "flip_vertical": np.flipud(ndvi),
                "rotate_90": np.rot90(ndvi, k=1),
                "rotate_180": np.rot90(ndvi, k=2),
                "rotate_270": np.rot90(ndvi, k=3),
            }

            for aug_name, aug_image in augmentations.items():
                aug_patch_dir = os.path.join(input_dir, "augmented", f"{patch_name}_{aug_name}")
                os.makedirs(aug_patch_dir, exist_ok=True)

                aug_ndvi_path = os.path.join(aug_patch_dir, os.path.basename(ndvi_file))
                if os.path.exists(aug_ndvi_path):
                    os.remove(aug_ndvi_path)

                save_image_as_jp2(aug_ndvi_path, aug_image, transform, crs, count=1)