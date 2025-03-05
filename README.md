# Desertification Detection in Mali using Siamese Networks

## Overview

This project applies a Fully Convolutional Siamese Network (FCSN) to detect desertification in Mali using Sentinel-2 satellite imagery. The model is trained to identify land cover changes between 2020 and 2025, utilizing pixel-wise labeled image pairs. The primary goal is to showcase AI-based change detection capabilities within remote sensing as part of my portfolio.

## Installation

### Prerequisites

Ensure you have Python (>=3.10) installed along with the necessary dependencies.

```bash
# Clone the repository
git clone https://github.com/yourusername/desertification-mali.git
cd desertification-mali

# Install dependencies
pip install -r requirements.txt

# Alternatively, install the package
pip install -e 
```

## Usage

### Preprocessing

1. Download Sentinel-2 images for the selected region and years (2020 & 2025).
2. Crop and align image pairs into 512x512 patches.
3. Generate binary change masks for supervised learning.

Run preprocessing with:

```bash
python src/preprocess.py
```

### Training the Model

Train the Siamese Network with:

```bash
python run_training.py --epochs 50 --batch_size 16
```

### Evaluating & Visualizing Results

Run inference on test regions:

```bash
python src/evaluate.py
```

Results are stored in the `outputs/` folder and can be visualized in a Jupyter notebook.

## Methodology

1. **Data Collection:** Sentinel-2 imagery (bands: Red, NIR, SWIR) for 2020 and 2025.
2. **Preprocessing:** Image alignment, patch extraction, manual labeling of change masks and application of data augmentation techniques.
3. **Model Architecture:** Fully Convolutional Siamese Network (FCSN) trained on pixel-wise labeled pairs.
4. **Evaluation:** Apply the trained model to unseen test areas to assess accuracy.

## License

This project is released under the MIT License.

---

Feel free to reach out if you have questions or suggestions!