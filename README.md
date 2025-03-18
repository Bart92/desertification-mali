# Desertification Detection in Mali using Siamese Networks

## Overview

This project applies a Fully Convolutional Siamese Network (FCSN) to detect desertification in Mali using Sentinel-2 satellite imagery. The model is trained to identify land cover changes between 2020 and 2025, utilizing labeled image pairs. The primary goal is to showcase AI-based change detection capabilities within remote sensing as part of my portfolio.

## Installation

### Prerequisites

Ensure you have Python (>=3.10) installed along with the necessary dependencies.

```bash
# Clone the repository
git clone https://github.com/yourusername/desertification-mali.git
cd desertification-mali

# Install the package and its dependencies
pip install -e .
```

This will install the project in editable mode and automatically handle dependencies defined in `setup.py`.

## Usage

### Preprocessing

First, you'll have to download Sentinel-2 images for a selected region and years (the model has been trained on 2020 & 2025 in an area around Nara, Mali, so if you're using one of my trained models and just want to do inference, you'll have to make sure you select a similar region).

Run preprocessing with:

```bash
python scripts/run_preprocessing.py
```

This is going to execute three steps:
- It will merge the tiles you have selected together using nearest neighbour resampling
- Calculate the NDVI, and store an RGB and NDVI image of both time stamps
- Create 512x512 patches from the outputs

If you're training from scratch, you'll have to label a part of these 512x512 patches to train your model.

### Training the Model

Train the Siamese Network with:

```bash
python scripts/run_training.py --epochs 50 --batch_size 16 --num_trials 2 --use_multiprocessing
```

All of these parameters on the run_training.py script are optional.
The training of this Siamese network is using random search for the setting the learning rate and the L2 regularization. I chose to reduce learning rate on plateau to keep the model from oscillating around the minimum and stabilizing the training process. It is using early stopping with patience to prevent overfitting, and checkpoints with a better-than-previous performance are stored the best model is preserved even if it were to overfit.
The script will output a number of evaluation metrics (accuracy, precision, recall and F1-score) and will store the models in the models/ directory, and logs documenting the loss of the training process in logs/. 

### Evaluating & Visualizing Results

Run inference on test regions:

```bash
python scripts/run_inference.py
```

Results are stored in the `results/` folder.

## Methodology

1. **Data Collection:** Sentinel-2 imagery (bands: Red, NIR, SWIR) for 2020 and 2025.
2. **Preprocessing:** Image alignment and merging, calculation of NDVI, patch extraction, manual labeling of change masks and application of data augmentation techniques.
3. **Model Architecture:** Siamese Network trained on labeled pairs.
4. **Evaluation:** Evaluated using accuracy, precision, recall and F1 score, as common in classification tasks.

## License

This project is released under the MIT License.

---

Feel free to reach out if you have questions or suggestions!