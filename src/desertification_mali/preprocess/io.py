import os
import rasterio
import numpy as np
from desertification_mali.utils import log_memory_usage

def read_band(date_dir: str, band: str) -> np.ndarray:
    """
    Reads a specific band from the given date directory using memory-mapped files.

    Parameters:
    - date_dir (str): Directory containing the merged tiles for the specific date.
    - band (str): Band to read (e.g., 'B02', 'B03', 'B04', 'B08').

    Returns:
    - np.ndarray: Array containing the band data.
    """
    band_file = os.path.join(date_dir, f'{band}.jp2')
    try:
        with rasterio.open(band_file) as src:
            return src.read(1).astype(float)
        log_memory_usage(f"After {band}")
    except rasterio.errors.RasterioIOError:
        raise FileNotFoundError(f"Band '{band}' does not exist for date directory '{date_dir}'")
    

def save_image_as_jp2(output_path: str, image: np.ndarray, transform: rasterio.Affine, crs: str, count: int) -> None:
    """
    Saves the given image as a JP2 file.

    Parameters:
    - output_path (str): Path to save the JP2 file.
    - image (np.ndarray): Array containing the image data.
    - transform (rasterio.Affine): Affine transform for the image.
    - crs (str): Coordinate reference system for the image.
    - count (int): Number of bands in the image (1 or 3).

    Returns:
    - None
    """
    
    # Ensure the image has the correct shape
    if count == 3:
        if image.ndim == 2:
            raise ValueError("Image with 3 bands should have 3 dimensions")
        elif image.shape[0] == 3:  # (bands, height, width)
            image = image.transpose(1, 2, 0)  # Convert to (height, width, bands)
    elif count == 1:
        if image.ndim == 3 and image.shape[0] == 1:  # (bands, height, width)
            image = image[0]  # Convert to (height, width)
    else:
        raise ValueError(f"Invalid count {count} for image bands. Expected 1 or 3.")
    
    with rasterio.open(
        output_path, 'w',
        driver='JP2OpenJPEG',
        width=image.shape[1] if count == 1 else image.shape[1],
        height=image.shape[0] if count == 1 else image.shape[0],
        count=count,
        dtype=image.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        if count == 3:
            dst.write(image[:, :, 0], 1)
            dst.write(image[:, :, 1], 2)
            dst.write(image[:, :, 2], 3)
        elif count == 1:
            dst.write(image, 1)