import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    """
    Siamese Network for change detection between two NDVI images.
    
    This network takes two NDVI images as input and outputs a single value (a scalar)
    representing the probability of change between the two images.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(256 * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, 1)
        
    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one input image.
        
        Parameters:
        - x (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width)
        
        Returns:
        - torch.Tensor: Output feature vector of shape (batch_size, 1024)
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one input image.
        
        Parameters:
        - x (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width)
        
        Returns:
        - torch.Tensor: Output feature vector of shape (batch_size, 1024)
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        combined = torch.abs(output1 - output2)
        combined = self.fc2(combined)
        combined = torch.sigmoid(combined)
        return combined

def loss_function(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy loss function.
    
    Parameters:
    - output (torch.Tensor): Predicted output tensor of shape (batch_size, 1)
    - target (torch.Tensor): Ground truth tensor of shape (batch_size, 1)
    
    Returns:
    - torch.Tensor: Calculated binary cross-entropy loss. This is a scalar value.
    """
    return F.binary_cross_entropy(output, target)