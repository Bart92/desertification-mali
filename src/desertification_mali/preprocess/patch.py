import os
import random
import rasterio
from rasterio.windows import Window
from desertification_mali.preprocess.io import save_image_as_jp2
import numpy as np

def create_patches(input_dir: str, output_dir: str, dates: list, size=512, manual_labeling_count=40, weak_supervision_count=80):
    """
    Crops raster images into smaller patches and saves them as JP2 files, organized by patch.

    Parameters:
    - input_dir (str): Path to the folder containing the input raster files.
    - output_dir (str): Directory to save the cropped patches.
    - dates (list): List of dates (years) to process.
    - size (int): Size of the patches (default is 512x512).
    - manual_labeling_count (int): Number of patches for manual labeling.
    - weak_supervision_count (int): Number of patches for weak supervision.

    Returns:
    - None
    """
    def save_band_patches(src, window, patch_dir, year, band):
        output_path = os.path.join(patch_dir, f"{year}_{band}.jp2")
        image = src.read(window=window)
        save_image_as_jp2(output_path, image, src.window_transform(window), src.crs, count=src.count)

    def save_patches(patches, subdir):
        for x, y in patches:
            patch_dir = os.path.join(output_dir, subdir, f"patch_{x}_{y}")
            os.makedirs(patch_dir, exist_ok=True)
            for year in dates:
                year_dir = os.path.join(input_dir, year)
                rgb_path = os.path.join(year_dir, 'RGB.jp2')
                ndvi_path = os.path.join(year_dir, 'NDVI.jp2')
                window = Window(x, y, size, size)
                with rasterio.open(rgb_path) as src_rgb:
                    save_band_patches(src_rgb, window, patch_dir, year, 'RGB')
                    transform = src_rgb.window_transform(window)
                    crs = src_rgb.crs
                with rasterio.open(ndvi_path) as src_ndvi:
                    save_band_patches(src_ndvi, window, patch_dir, year, 'NDVI')
            # Create an empty placeholder for labeled change
            output_path_label = os.path.join(patch_dir, "labeled_change.jp2")
            empty_image = np.zeros((size, size), dtype='uint8')
            save_image_as_jp2(output_path_label, empty_image, transform=transform, crs=crs, count=1)

    # Ensure both years have the same dimensions and patches
    first_year_dir = os.path.join(input_dir, dates[0])
    second_year_dir = os.path.join(input_dir, dates[1])
    
    with rasterio.open(os.path.join(first_year_dir, 'RGB.jp2')) as src1, \
         rasterio.open(os.path.join(second_year_dir, 'RGB.jp2')) as src2:
        assert src1.width == src2.width, "Widths of the images for both years do not match."
        assert src1.height == src2.height, "Heights of the images for both years do not match."
        
        width, height = src1.width, src1.height
        # TODO: Throw out the last patch on the right and bottom if it's not a full patch
        patches = [(x, y) for x in range(0, width, size) for y in range(0, height, size)]
        random.shuffle(patches)

        manual_labeling_patches = patches[:manual_labeling_count]
        weak_supervision_patches = patches[manual_labeling_count:manual_labeling_count + weak_supervision_count]

        # Save patches to respective directories
        save_patches(manual_labeling_patches, 'manual_labeling')
        save_patches(weak_supervision_patches, 'weak_supervision')
