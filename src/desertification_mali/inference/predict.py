import torch
from desertification_mali.train.model import SiameseNetwork
from desertification_mali.train.dataset import NDVIDataset

def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads the trained model from the specified path.

    Parameters:
    - model_path (str): Path to the saved model.
    - device (torch.device): The device to load the model on (CPU or GPU).

    Returns:
    - torch.nn.Module: The loaded model.
    """
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_data(data_path: str) -> torch.utils.data.Dataset:
    """
    Preprocesses the input data for prediction.

    Parameters:
    - data_path (str): Path to the input data.

    Returns:
    - torch.utils.data.Dataset: The preprocessed dataset.
    """
    dataset = NDVIDataset(data_path, train=False)
    return dataset

def predict(model: torch.nn.Module, dataset: torch.utils.data.Dataset, device: torch.device, return_raw: bool = False) -> list:
    """
    Generates predictions using the trained model.

    Parameters:
    - model (torch.nn.Module): The trained model.
    - dataset (torch.utils.data.Dataset): The dataset for prediction.
    - device (torch.device): The device to run the prediction on (CPU or GPU).
    - return_raw (bool): Whether to return raw predictions

    Returns:
    - list: The predictions.
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            tile_id, input1, input2 = batch
            input1, input2 = input1.to(device), input2.to(device)
            output = model(input1, input2)
            prediction = output.cpu().numpy()[0]

            if not return_raw:
                prediction = int(prediction > 0.5)
            predictions.append({'Tile ID': tile_id[0], 'Prediction': prediction})

    return predictions