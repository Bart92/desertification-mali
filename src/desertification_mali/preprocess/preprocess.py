from desertification_mali.preprocess.merge import merge_tiles
from desertification_mali.preprocess.rgb import save_rgb
from desertification_mali.preprocess.ndvi import save_ndvi
from desertification_mali.preprocess.io import read_band
from desertification_mali.utils import get_unique_dates, get_transform_and_crs
import os
import psutil

class Preprocessor:
    """
    Preprocessor class to handle the preprocessing steps for Sentinel-2 data.
    
    Attributes:
    - input_directory (str): Directory containing the raw Sentinel-2 tiles.
    - output_directory (str): Directory to save the processed output.
    """

    def __init__(self, input_directory: str, merged_output_dir: str):
        self.input_directory = input_directory
        self.merged_output_dir = merged_output_dir
        self.dates = get_unique_dates(input_directory)

    def run(self):
        """
        Runs the preprocessing steps for Sentinel-2 data.
        """
        # Step 1: Merge tiles
        # self.merge_tiles()

        # Step 2: Calculate NDVI and store NDVI and RGB images
        self.save_rgb_ndvi()

    def merge_tiles(self):
        """
        Merges the raw Sentinel-2 tiles in the input directory and saves the merged output in the output directory.
        """
        
        merge_tiles(self.input_directory, self.merged_output_dir, self.dates)

    def save_rgb_ndvi(self):
        """
        Processes the bands for the specified dates to create RGB and NDVI images.
        
        Parameters:
        - dates (List[str]): List of dates to process in the format 'YYYYMMDD'.
        """
        for date in self.dates:
            date_dir = os.path.join(self.merged_output_dir, date)
            transform, crs = get_transform_and_crs(date_dir)

            self.log_memory_usage("Before reading bands")

            B02 = read_band(date_dir, 'B02')
            self.log_memory_usage("After B02")

            B03 = read_band(date_dir, 'B03')
            self.log_memory_usage("After B03")

            B04 = read_band(date_dir, 'B04')
            self.log_memory_usage("After B04")
            save_rgb(B02, B03, B04, date, self.merged_output_dir, transform, crs)
            # Release memory
            del B02, B03

            B08 = read_band(date_dir, 'B08')
            save_ndvi(B04, B08, date, self.merged_output_dir, transform, crs)

            # Release memory
            del B04, B08

    def log_memory_usage(self, stage: str):
        """
        Logs the current memory usage.
        
        Parameters:
        - stage (str): Description of the current stage for logging purposes.
        """
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"[{stage}] Memory usage: {mem_info.rss / (1024 * 1024)} MB")
