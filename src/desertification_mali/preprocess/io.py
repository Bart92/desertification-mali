import os
import rasterio
import numpy as np
from typing import List

# def read_band(date_dir: str, band: str) -> np.ndarray:
#     """
#     Reads a specific band from the given date directory.

#     Parameters:
#     - date_dir (str): Directory containing the merged tiles for the specific date.
#     - band (str): Band to read (e.g., 'B02', 'B03', 'B04', 'B08').

#     Returns:
#     - np.ndarray: Array containing the band data.
#     """
#     band_file = os.path.join(date_dir, f'{band}.jp2')
#     try:
#         return rasterio.open(band_file).read(1).astype(float)
#     except rasterio.errors.RasterioIOError:
#         raise FileNotFoundError(f"Band '{band}' does not exist for date directory '{date_dir}'")


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
    # try:
    #     with rasterio.open(band_file) as src:
    #         # Use memory-mapped file to read the band data
    #         memmap = np.memmap(band_file, dtype='float32', mode='r', shape=(src.height, src.width))
    #         return memmap
    except rasterio.errors.RasterioIOError:
        raise FileNotFoundError(f"Band '{band}' does not exist for date directory '{date_dir}'")
    

def save_image_as_jp2(output_path: str, image: np.ndarray, transform: rasterio.Affine, crs: str, count: int) -> None:
    """
    Saves the given image as a JP2 file.

    Parameters:
    - output_path (str): Path to save the JP2 file.
    - image (np.ndarray): Array containing the image data.
    - date_dir (str): Directory containing the merged tiles for the specific date.
    - count (int): Number of bands in the image (1 or 3).
    - reference_band (str): Reference band for obtaining the transform and CRS.

    Returns:
    - None
    """
    with rasterio.open(
        output_path, 'w',
        driver='JP2OpenJPEG',
        width=image.shape[1],
        height=image.shape[0],
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
        else:
            raise ValueError(f"Invalid count {count} for image bands. Expected 1 or 3.")