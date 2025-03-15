import os
import numpy as np
from desertification_mali.preprocess.io import save_image_as_jp2

def calculate_ndvi(B04: np.ndarray, B08: np.ndarray) -> np.ndarray:
    """
    Creates an NDVI image from the given bands.

    Parameters:
    - B04 (np.ndarray): Array containing the B04 band data.
    - B08 (np.ndarray): Array containing the B08 band data.

    Returns:
    - np.ndarray: Array containing the NDVI image data.
    """
    return ((B08 - B04) / (B08 + B04) * 255).astype(np.uint8)


def save_ndvi(B04, B08, date, output_dir, transform, crs):
    """
    Processes the NDVI image from the given bands and saves it to the output directory.
    
    Parameters:
    - B04 (np.ndarray): Array containing the B04 band data.
    - B08 (np.ndarray): Array containing the B08 band data.
    - date (str): Specific date being processed in the format 'YYYYMMDD'.
    - transform (rasterio.Affine): Affine transformation matrix for the image.
    - crs (str): Coordinate Reference System for the image.
    """
    ndvi = calculate_ndvi(B04, B08)
    ndvi_output_path = os.path.join(output_dir, date, 'NDVI.jp2')
    os.makedirs(os.path.join(output_dir, date), exist_ok=True)
    save_image_as_jp2(ndvi_output_path, ndvi, transform, crs, count=1)
    print(f'NDVI calculated for {date} and saved to {ndvi_output_path}')
