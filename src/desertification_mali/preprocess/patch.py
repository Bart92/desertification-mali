import os
import rasterio
from rasterio.windows import Window
from desertification_mali.preprocess.io import save_image_as_jp2

def create_patches(input_path, output_dir, size=512):
    """
    Crops a raster image into smaller patches and saves them as JP2 files.

    Parameters:
    - input_path (str): Path to the input raster file.
    - output_dir (str): Directory to save the cropped patches.
    - size (int): Size of the patches (default is 512x512).

    Returns:
    - None
    """
    with rasterio.open(input_path) as src:
        width, height = src.width, src.height
        for x in range(0, width, size):
            for y in range(0, height, size):
                window = Window(x, y, size, size)
                output_path = os.path.join(output_dir, f"patch_{x}_{y}.jp2")
                os.makedirs(output_dir, exist_ok=True)
                image = src.read(window=window)
                save_image_as_jp2(output_path, image, src.window_transform(window), src.crs, count=src.count)