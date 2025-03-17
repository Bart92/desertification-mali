from desertification_mali.train.model import loss_function
import torch
from torch.utils.data import DataLoader

class Trainer:
    """
    Trainer class for training a Siamese Network on NDVI image pairs.

    Attributes:
    - model (nn.Module): The Siamese Network model to be trained.
    - dataset (Dataset): The dataset containing NDVI image pairs and labels.
    - batch_size (int): The number of samples per batch.
    - learning_rate (float): The learning rate for the optimizer.
    - num_epochs (int): The number of epochs to train the model.
    - device (torch.device): The device to run the training on (CPU or GPU).
    - optimizer (torch.optim.Optimizer): The optimizer for training the model.
    - dataloader (DataLoader): The DataLoader for iterating over the dataset.
    """
    def __init__(self, model: torch.nn.Module, dataset: torch.utils.data.Dataset, batch_size: int = 8, learning_rate: int = 0.001, num_epochs: int = 10, l2_lambda: float = 0.0) -> None:
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.l2_lambda = l2_lambda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_lambda)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def train(self) -> None:
        """
        Trains the model for the specified number of epochs, monitoring loss and accuracy.
        """
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch in self.dataloader:
                input1, input2, target = batch
                input1, input2, target = input1.to(self.device), input2.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(input1, input2)
                target = target.view(-1, 1)  # Reshape target to match output shape
                loss = loss_function(output, target)
                loss.backward()
                self.optimizer.step()
                
                # TODO: Also add accuracy and F1 score
                total_loss += loss.item()
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {total_loss/len(self.dataloader)}")