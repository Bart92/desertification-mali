from desertification_mali.train.model import loss_function
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pandas as pd

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
    - scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): The learning rate scheduler.
    - early_stopping_patience (int): The number of epochs to wait for an improvement before stopping training.
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
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.early_stopping_patience = 5
        self.best_loss = float('inf')
        self.epochs_no_improve = 0


    def train(self) -> None:
        """
        Trains the model for the specified number of epochs, monitoring loss and accuracy.
        """
        self.model.train()
        training_log = []

        torch.cuda.empty_cache() # Empty CUDA cache before every training run to avoid OutOfMemory errors

        for epoch in range(self.num_epochs):
            avg_loss = self._train_epoch(epoch)

            training_log.append([epoch+1, avg_loss])

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch + 1, is_best=True)
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.early_stopping_patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break

        self._save_checkpoint(epoch + 1, is_best=False)
        self._save_training_log(training_log)


    def _train_epoch(self, epoch: int) -> float:
        """
        Trains the model for one epoch.

        Parameters:
        - epoch (int): The current epoch number.

        Returns:
        - float: The average loss for the epoch.
        """
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
            
            total_loss += loss.item()

        self.scheduler.step(total_loss)
        avg_loss = total_loss / len(self.dataloader)
        print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss}")
        return avg_loss


    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        """
        Saves the model checkpoint.

        Parameters:
        - epoch (int): The current epoch number.
        - is_best (bool): Whether this is the best model so far.
        """
        model_filename = f"model_bs{self.batch_size}_lr{self.learning_rate}_epochs{self.num_epochs}_l2{self.l2_lambda}.pth"
        if is_best:
            model_filename = f"best_{model_filename}"
        else:
            model_filename = f"final_{model_filename}"
        torch.save(self.model.state_dict(), os.path.join("models", model_filename))
        print(f"Model checkpoint saved at epoch {epoch} as {model_filename}")


    def _save_training_log(self, training_log: list) -> None:
        """
        Saves the training log to a CSV file.

        Parameters:
        - training_log (list): The training log containing epoch and loss information.
        """
        log_filename = f"training_log_bs{self.batch_size}_lr{self.learning_rate}_epochs{self.num_epochs}_l2{self.l2_lambda}.csv"
        log_filepath = os.path.join("logs", log_filename)
        df = pd.DataFrame(training_log, columns=['Epoch', 'Loss'])
        df.to_csv(log_filepath, index=False)
        print("Training log saved.")