import rasterio
from rasterio.merge import merge
import glob
import os
import re
from typing import List, Optional

def merge_tiles(input_dir: str, output_path: str, date: Optional[str] = None) -> None:
    """
    Merges Sentinel-2 tiles for specified bands (B02, B03, B04, B08) for a given date or all unique dates and
    writes them to a file (see merge_tiles_by_date).

    Parameters:
    - input_dir (str): Directory containing the raw Sentinel-2 tiles.
    - output_path (str): Directory to save the merged output tiles.
    - date (str, optional): Specific date to process in the format 'YYYYMMDD'. If not provided, all unique dates will be processed.

    Returns:
    - None
    """
    os.makedirs(output_path, exist_ok=True)

    if date:
        dates = [date]
    else:
        dates = get_unique_dates(input_dir)
    
    for date in dates:
        merge_tiles_by_date(input_dir, output_path, date)


def merge_tiles_by_date(input_dir: str, output_path: str, date: str, bands: List[str] = ['B02', 'B03', 'B04', 'B08']) -> None:
    """
    Merges Sentinel-2 tiles for specified bands (B02, B03, B04, B08) for a specific date and writes them to
    an output file of structure {output_path}/{band}.jp2.

    Parameters:
    - input_dir (str): Directory containing the raw Sentinel-2 tiles. Assumes a file structure of {input_dir}/{date}/GRANULE/*/IMG_DATA/R10m/*.jp2.
    - output_path (str): Directory to save the merged output tiles.
    - date (str): Specific date to process in the format 'YYYYMMDD'.

    Returns:
    - None
    """
    os.makedirs(output_path, exist_ok=True)
    
    search_pattern = os.path.join(input_dir, '*'+date+'*', 'GRANULE', '*', 'IMG_DATA', 'R10m', '*.jp2')
    q = glob.glob(search_pattern)

    for i, band in enumerate(bands):
        filtered_files = [fp for fp in q if band in fp]
        
        src_files_to_mosaic = []
        for fp in filtered_files:
            src = rasterio.open(fp)
            src_files_to_mosaic.append(src)
        
        # Merge the files. Chose to use nearest neighbor resampling to preserve the original pixel values, bilinear would interpolate the values
        mosaic, out_trans = merge(src_files_to_mosaic, method='first', resampling=rasterio.enums.Resampling.nearest)
        
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "JP2OpenJPEG",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "count": mosaic.shape[0]
        })
            
        os.makedirs(os.path.join(output_path, date), exist_ok=True)
        with rasterio.open(os.path.join(output_path, date, f"{band}.jp2"), "w", **out_meta) as dest:
            dest.write(mosaic)
        print(f"Merged the tiles of {band} and saved it to {output_path}/{band}.jp2 ({i+1}/{len(bands_to_include)})")

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