import os
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

class NDVIDataset(Dataset):
    """
    Dataset class for loading NDVI images and their corresponding labels.

    This dataset loads NDVI images for the years 2020 and 2025 from the specified directory,
    and retrieves the corresponding labels from a CSV file.

    Attributes:
    - patch_dir (str): Path to the directory containing the patches.
    - labels (pd.DataFrame): DataFrame containing the patch IDs and their corresponding labels.
    """

    def __init__(self, patch_dir: str, label_file: pd.DataFrame) -> None:
        self.patch_dir = patch_dir
        self.labels = pd.read_csv(label_file)

    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves the NDVI images for 2020 and 2025 and the corresponding label for a given index.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - tuple: (ndvi_2020, ndvi_2025, label) where ndvi_2020 and ndvi_2025 are the NDVI images for 2020 and 2025, and label is the corresponding label.
        """
        patch_id = self.labels.iloc[idx, 0]
        label = self.labels.iloc[idx, 1]
        
        ndvi_2020_path = os.path.join(self.patch_dir, patch_id, '20200205_NDVI.jp2')
        ndvi_2025_path = os.path.join(self.patch_dir, patch_id, '20250218_NDVI.jp2')
        
        with rasterio.open(ndvi_2020_path) as src_2020:
            ndvi_2020 = src_2020.read(1).astype('float32')
        
        with rasterio.open(ndvi_2025_path) as src_2025:
            ndvi_2025 = src_2025.read(1).astype('float32')
        
        ndvi_2020 = torch.tensor(ndvi_2020).unsqueeze(0)  # Add channel dimension
        ndvi_2025 = torch.tensor(ndvi_2025).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(label).float()
        
        return ndvi_2020, ndvi_2025, label