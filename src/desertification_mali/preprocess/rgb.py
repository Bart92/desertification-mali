import os
import numpy as np
from desertification_mali.preprocess.io import save_image_as_jp2

def create_rgb(B02: np.ndarray, B03: np.ndarray, B04: np.ndarray) -> np.ndarray:
    """
    Creates an RGB image from the given bands.

    Parameters:
    - B02 (np.ndarray): Array containing the B02 band data.
    - B03 (np.ndarray): Array containing the B03 band data.
    - B04 (np.ndarray): Array containing the B04 band data.

    Returns:
    - np.ndarray: Array containing the RGB image data.
    """
    rgb = np.dstack((B04, B03, B02))
    return (rgb / np.max(rgb) * 255).astype(np.uint8)

def save_rgb(B02, B03, B04, date, output_dir, transform, crs):
    """
    Processes the RGB image from the given bands and saves it to the output directory.
    
    Parameters:
    - B02 (np.ndarray): Array containing the B02 band data.
    - B03 (np.ndarray): Array containing the B03 band data.
    - B04 (np.ndarray): Array containing the B04 band data.
    - date (str): Specific date being processed in the format 'YYYYMMDD'.
    - date_dir (str): Directory containing the merged tiles for the specific date.
    """
    rgb_normalized = create_rgb(B02, B03, B04)
    rgb_output_path = os.path.join(output_dir, date, 'RGB.jp2')
    os.makedirs(os.path.join(output_dir, date), exist_ok=True)
    save_image_as_jp2(rgb_output_path, rgb_normalized, transform, crs, count=3)
    print(f'RGB composite created for {date} and saved to {rgb_output_path}')
