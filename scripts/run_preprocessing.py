"""
This script runs the preprocessing pipeline on the raw Sentinel-2 tiles.
It consists of the following steps:
 1. Merging the raw Sentinel-2 tiles
 2. Calculate NDVI over the merged tiles, and save it to the output directory
 3. Save an RGB image of the merged tiles to the output directory for manual labeling

"""

from desertification_mali.preprocess.preprocess import Preprocessor

def main():
    input_directory = 'data/raw_tiles'
    output_directory = 'data/merged_tiles'
    patch_directory = 'data/patches'

    preprocessor = Preprocessor(input_directory, output_directory, patch_directory)
    preprocessor.run()

if __name__ == "__main__":
    main()