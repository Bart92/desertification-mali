import os
import glob
import re
from typing import List, Tuple
import rasterio
import psutil
import os
import csv
import re


def get_unique_dates(input_dir: str) -> List[str]:
    """
    Extracts unique dates from the directory names in the input directory.
    Note that the regex is tailored to Sentinel2 level 2A data.

    Parameters:
    - input_dir (str): Directory containing the raw Sentinel-2 tiles.

    Returns:
    - list: Sorted list of unique dates in the format 'YYYYMMDD'.
    """
    search_pattern = os.path.join(input_dir, '*.SAFE')
    safe_dirs = glob.glob(search_pattern)
    
    dates = set()
    for safe_dir in safe_dirs:
        match = re.search(r'MSIL2A_(\d{8})T', os.path.basename(safe_dir))
        if match:
            dates.add(match.group(1))
    
    print(f"Unique dates found: {sorted(dates)}")
    return sorted(dates)


def get_transform_and_crs(date_dir: str, reference_band: str = "B02") -> Tuple[rasterio.Affine, str]:
    """
    Gets the transform and CRS from the reference band.

    Parameters:
    - date_dir (str): Directory containing the merged tiles for the specific date.
    - reference_band (str): Reference band to get the transform and CRS.

    Returns:
    - Tuple[rasterio.Affine, str]: The transform and CRS.
    """
    with rasterio.open(os.path.join(date_dir, f'{reference_band}.jp2')) as src:
        return src.transform, src.crs
    

def log_memory_usage(stage: str):
    """
    Logs the current memory usage.
    
    Parameters:
    - stage (str): Description of the current stage for logging purposes.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[{stage}] Memory usage: {mem_info.rss / (1024 * 1024)} MB")


def generate_initial_label_file(patch_dir: str, output_file: str) -> None:
    """
    Generates a label file for the patches in the specified directory.

    Parameters:
    - patch_dir (str): Path to the directory containing the patches.
    - output_file (str): Path to the output CSV file to store the labels.
    """
    def sort_key(patch_name: str) -> tuple:
        """
        Extracts the numerical parts of the patch name for sorting.

        Parameters:
        - patch_name (str): The name of the patch.

        Returns:
        - tuple: A tuple containing the numerical parts of the patch name.
        """
        match = re.match(r'patch_(\d+)_(\d+)', patch_name)
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            return (x, y)
        return (0, 0)

    labels = {}
    patch_names = sorted(os.listdir(patch_dir), key=sort_key)
    for patch_name in patch_names:
        labels[patch_name] = 0  # Initialize with label 0

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['patch_id', 'label'])
        for patch_id, label in labels.items():
            writer.writerow([patch_id, label])
