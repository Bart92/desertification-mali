"""
Script to run the training process for the Siamese Network on NDVI image pairs.

This script initializes the dataset, model, and trainer, and starts the training process.
"""

import random
from desertification_mali.train.model import SiameseNetwork
from desertification_mali.train.dataset import NDVIDataset
from desertification_mali.train.train import Trainer
from sklearn.metrics import accuracy_score, f1_score

def sample_hyperparameters():
    """
    Samples hyperparameters from specified distributions.

    Returns:
    - dict: A dictionary containing sampled hyperparameters.
    """
    return {
        'batch_size': random.choice([4, 8]),
        'learning_rate': random.uniform(0.0001, 0.01),
        'num_epochs': random.choice([10, 20, 30]),
        'l2_lambda': random.uniform(0.0, 0.1)
    }

def evaluate_model(hyperparameters):
    """
    Trains and evaluates the model with the given hyperparameters.

    Parameters:
    - hyperparameters (dict): A dictionary containing hyperparameters.

    Returns:
    - float: The accuracy of the model on the validation set.
    """
    dataset = NDVIDataset('data/patches/manual_labeling', 'data/patches/labels.csv')
    model = SiameseNetwork()
    
    trainer = Trainer(
        model=model,
        dataset=dataset,
        batch_size=hyperparameters['batch_size'],
        learning_rate=hyperparameters['learning_rate'],
        num_epochs=hyperparameters['num_epochs'],
        l2_lambda=hyperparameters['l2_lambda']
    )
    trainer.train()
    
    all_targets = []
    all_predictions = []
    for batch in trainer.dataloader:
        input1, input2, target = batch
        input1, input2, target = input1.to(trainer.device), input2.to(trainer.device), target.to(trainer.device)
        output = model(input1, input2)
        all_targets.extend(target.cpu().numpy())
        all_predictions.extend((output.detach().cpu().numpy() > 0.5).astype(int))
    
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    return accuracy, f1

def main():
    num_trials = 2
    best_accuracy = 0
    best_hyperparameters = None
    
    for _ in range(num_trials):
        hyperparameters = sample_hyperparameters()
        accuracy, f1 = evaluate_model(hyperparameters)
        print(f"Accuracy: {accuracy}, F1 score: {f1}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hyperparameters = hyperparameters
    
    print(f"Best Hyperparameters: {best_hyperparameters}, Best Accuracy: {best_accuracy}")

if __name__ == "__main__":
    main()