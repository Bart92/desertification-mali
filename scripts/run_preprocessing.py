from desertification_mali.preprocess.preprocess import Preprocessor

input_directory = 'data/raw_tiles'
output_directory = 'data/processed_tiles'

preprocessor = Preprocessor(input_directory, output_directory)
# preprocessor.run()