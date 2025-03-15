from desertification_mali.preprocess.merge import merge_tiles

class Preprocessor:
    """
    Preprocessor class to handle the preprocessing steps for Sentinel-2 data.
    
    Attributes:
    - input_directory (str): Directory containing the raw Sentinel-2 tiles.
    - output_directory (str): Directory to save the processed output.
    """

    def __init__(self, input_directory: str, output_directory: str):
        self.input_directory = input_directory
        self.output_directory = output_directory

    def run(self):
        """
        Runs the preprocessing steps for Sentinel-2 data.
        """
        # Step 1: Merge tiles
        self.merge_tiles()

    def merge_tiles(self):
        """
        Merges the raw Sentinel-2 tiles in the input directory and saves the merged output in the output directory.
        """
        merge_tiles(self.input_directory, self.output_directory)