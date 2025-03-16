import os
import numpy as np
from desertification_mali.preprocess.io import save_image_as_jp2
import cv2

def gamma_correction(image, gamma=1.5):
    """Apply gamma correction to enhance brightness."""
    return np.clip(image ** (1/gamma), 0, 1)

def apply_clahe(band):
    """Apply CLAHE to enhance local contrast."""
    band = (band * 255).astype(np.uint8)  # Convert to 8-bit
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(band) / 255.0  # Rescale to [0,1]

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
    red = gamma_correction(apply_clahe(B04 / 10000), gamma=1.8)
    green = gamma_correction(apply_clahe(B03 / 10000), gamma=1.8)
    blue = gamma_correction(apply_clahe(B02 / 10000), gamma=1.8)

    rgb = np.dstack((red, green, blue))

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
