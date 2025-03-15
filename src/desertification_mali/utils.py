import os
import glob
import re
from typing import List, Tuple
import rasterio
import psutil

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
