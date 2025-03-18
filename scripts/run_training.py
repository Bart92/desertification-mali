"""
Script to run the training process for the Siamese Network on NDVI image pairs.

This script initializes the dataset, model, and trainer, and starts the training process.
"""

import random
from desertification_mali.train.model import SiameseNetwork
from desertification_mali.train.dataset import NDVIDataset
from desertification_mali.train.train import Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import multiprocessing

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
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    return accuracy, f1, precision, recall

def run_trial(trial_id: int) -> tuple:
    """
    Runs a single trial of random search for a specific set of hyperparameters.

    Parameters:
    - trial_id (int): The ID of the trial.

    Returns:
    - tuple: A tuple containing the accuracy, F1 score, precision, recall, and hyperparameters of the trial.
    """
    hyperparameters = sample_hyperparameters()
    accuracy, f1, precision, recall = evaluate_model(hyperparameters)
    print(f"Trial {trial_id}: Accuracy: {accuracy}, F1 score: {f1}, Precision: {precision}, Recall: {recall}")
    return accuracy, f1, precision, recall, hyperparameters


def run_trials(num_trials: int, use_multiprocessing: bool = False):
    """
    Runs multiple trials of random search for hyperparameter tuning.

    Parameters:
    - num_trials (int): The number of trials to run.
    - use_multiprocessing (bool): Whether to use multiprocessing for parallel trials.

    Returns:
    - tuple: The best accuracy and corresponding hyperparameters.
    """
    if use_multiprocessing:
        with multiprocessing.Pool(processes=num_trials) as pool:
            results = pool.map(run_trial, range(num_trials))
    else:
        results = [run_trial(trial_id) for trial_id in range(num_trials)]

    best_result = max(results, key=lambda x: x[0])  # x[0] is accuracy
    return best_result

def main():
    num_trials = 2
    use_multiprocessing = False

    best_accuracy, best_f1, best_precision, best_recall, best_hyperparameters = run_trials(
        num_trials, use_multiprocessing
    )

    print(f"Best Hyperparameters: {best_hyperparameters}")
    print(f"Best Accuracy: {best_accuracy}, F1 Score: {best_f1}, Precision: {best_precision}, Recall: {best_recall}")
    

if __name__ == "__main__":
    main()