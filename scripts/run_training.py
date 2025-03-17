"""
Script to run the training process for the Siamese Network on NDVI image pairs.

This script initializes the dataset, model, and trainer, and starts the training process.
"""

from desertification_mali.train.model import SiameseNetwork
from desertification_mali.train.dataset import NDVIDataset
from desertification_mali.train.train import Trainer

def main():
    dataset = NDVIDataset('data/patches/manual_labeling', 'data/patches/labels.csv')
    model = SiameseNetwork()
    
    trainer = Trainer(model, dataset)
    trainer.train()

if __name__ == "__main__":
    main()